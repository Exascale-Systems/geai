import pathlib
from tqdm.auto import tqdm
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from src.load import MasterDataset, collate, _worker_init_fn
from src.transform import *
from src.metrics import *
from src.nn import GravInvNet

dev = torch.device('cuda:7'); print(dev)
net = GravInvNet().to(dev)
lr=1e-3
wd=0.0
opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
crit = nn.MSELoss()
scaler = torch.amp.GradScaler('cuda')
writer = SummaryWriter()

def data_prep(ds_name: dict, split_name: dict):
    stats = compute_stats(f"datasets/{ds_name}.h5")                     # calculate distribution of data
    ds = MasterDataset(f"datasets/{ds_name}.h5")                        # load dataset
    ds.transform = make_transform(ds.shape_cells, stats, noise=(0,1))   # convert to model ready tensors
    n=len(ds); n_tr=max(1,int(0.8*n)); n_va=n-n_tr                      # 80% train, 20% val
    bs=min(8,n_tr)                                                      # batch size
    g = torch.Generator().manual_seed(0) 
    tr_ds, va_ds = random_split(ds,[n_tr,n_va],generator=g)
    pathlib.Path("splits").mkdir(parents=True, exist_ok=True); 
    np.savez(f"splits/{split_name}.npz", tr=np.array(tr_ds.indices), va=np.array(va_ds.indices))
    tr_ld = DataLoader(tr_ds,batch_size=bs,shuffle=True,num_workers=2,worker_init_fn=_worker_init_fn,collate_fn=collate, pin_memory=True)
    va_ld = DataLoader(va_ds,batch_size=bs,shuffle=False,num_workers=2,worker_init_fn=_worker_init_fn,collate_fn=collate, pin_memory=True)
    return tr_ld, va_ld, stats

def run_epoch(ld: DataLoader, train=True, ema_alpha=0.1, epoch=0):
    net.train() if train else net.eval()
    ema,tot,n=None,0.0,0
    with torch.enable_grad() if train else torch.no_grad():
        bar = tqdm(ld, leave=False, ncols=100)
        for batch_idx, (gz,tgt) in enumerate(bar):
            gz,tgt=gz.to(dev, non_blocking=True),tgt.to(dev, non_blocking=True)
            if train:
                opt.zero_grad(set_to_none=True)
            pred=net(gz)
            loss = crit(pred, tgt)
            if train:
                scaler.scale(loss).backward() 
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), float('inf'))
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                writer.add_scalar("Gradients/norm", grad_norm, epoch * len(ld) + batch_idx)
                scaler.step(opt)
                scaler.update()
            b=gz.size(0)
            li = loss.item()
            tot+=li*b
            n+=b
            ema = li if ema is None else (ema_alpha * li + (1 - ema_alpha) * ema)
            bar.set_postfix(loss=f"{li:.4f}", ema=f"{ema:.4f}")
    return tot/max(1,n)

# 
def eval_metrics(ld: DataLoader, stats=dict, threshold=0.1):
    """
    Return rmse, l1, IoU, dice for dataset.
    """
    net.eval()
    sum_se, sum_ae, n_samples = 0.0, 0.0, 0
    intersection, union, true_sum, pred_sum = 0, 0, 0, 0
    with torch.no_grad():
        for gz, tgt in ld:
            gz, tgt = gz.to(dev), tgt.to(dev)
            pred = net(gz)
            pred_denorm = denorm(pred, stats)
            tgt_denorm = denorm(tgt, stats)
            diff = tgt_denorm - pred_denorm
            sum_se += torch.sum(diff * diff).item()
            sum_ae += torch.sum(torch.abs(diff)).item()
            n_samples += diff.numel()
            true_binary = tgt_denorm > threshold
            pred_binary = pred_denorm > threshold
            intersection += torch.sum(true_binary & pred_binary).item()
            union += torch.sum(true_binary | pred_binary).item()
            true_sum += torch.sum(true_binary).item()
            pred_sum += torch.sum(pred_binary).item()
    rmse_val = (sum_se / n_samples) ** 0.5
    l1_val = sum_ae / n_samples
    iou_val = intersection / union if union > 0 else 1.0
    dice_val = (2 * intersection) / (true_sum + pred_sum) if (true_sum + pred_sum) > 0 else 1.0
    return rmse_val, l1_val, iou_val, dice_val

def train(tr_ld: DataLoader, va_ld: DataLoader, E=200, min_loss=1e-5, stats= dict):
    pbar=tqdm(range(0, E),desc="training",ncols=100)
    best = float("inf")
    for e in pbar:
        tr=run_epoch(ld=tr_ld, train=True, epoch=e)
        va=run_epoch(ld=va_ld, train=False, epoch=e)
        writer.add_scalar("Loss/train", tr, e)
        writer.add_scalar("Loss/val",   va, e)
        writer.add_scalar("Hyperparams/LR", lr, e)
        writer.add_scalar("Hyperparams/WeightDecay", wd, e)
        if e % 10 == 0:
            rmse_val, l1_val, iou_val, dice_val = eval_metrics(va_ld, stats=stats)
            writer.add_scalar("Metrics/RMSE", rmse_val, e)
            writer.add_scalar("Metrics/L1", l1_val, e)
            writer.add_scalar("Metrics/IoU", iou_val, e)
            writer.add_scalar("Metrics/Dice", dice_val, e)
            pbar.set_postfix(train=f"{tr:.4f}", val=f"{va:.4f}", rmse=f"{rmse_val:.4f}")
        else:
            pbar.set_postfix(train=f"{tr:.4f}", validation=f"{va:.4f}")
        for name, param in net.named_parameters():
            writer.add_histogram(f"Weights/{name}", param, e)
        if va < best:
            best = va
            torch.save({'model': net.state_dict()}, 'checkpoints/best.pt')
        if va < min_loss:
            print(f"Reached target loss {va:.6f} at epoch {e}")
            break
    writer.flush()
    writer.close()
    torch.save({"model": net.state_dict()}, "checkpoints/final.pt")

def main():
    tr_ld, va_ld, stats = data_prep(ds_name="single_block_overfit", split_name="test")
    train(tr_ld, va_ld, E=200, min_loss=1e-5, stats=stats)

if __name__ == "__main__":
    main()
