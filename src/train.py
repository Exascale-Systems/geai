import pathlib
import numpy as np
from tqdm.auto import tqdm
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from src.load import MasterDataset, collate, _worker_init_fn
from src.transform import *
from src.metrics import *
from src.nn import GravInvNet

# data
stats = compute_stats("datasets/sg_overfit.h5")         # calculate distribution of data
ds = MasterDataset("datasets/sg_overfit.h5")            # load dataset
ds.transform = make_transform(ds.shape_cells, stats)    # convert to model ready tensors
n=len(ds); n_tr=max(1,int(0.8*n)); n_va=n-n_tr          # 80% train, 20% val
bs=min(8,n_tr)                                          # batch size
g = torch.Generator().manual_seed(0) 
tr_ds, va_ds = random_split(ds,[n_tr,n_va],generator=g)
pathlib.Path("splits").mkdir(parents=True, exist_ok=True); 
np.savez("splits/sg_overfit.npz", tr=np.array(tr_ds.indices), va=np.array(va_ds.indices))
tr_ld = DataLoader(tr_ds,batch_size=bs,shuffle=True,num_workers=2,worker_init_fn=_worker_init_fn,collate_fn=collate, pin_memory=True)
va_ld = DataLoader(va_ds,batch_size=bs,shuffle=False,num_workers=2,worker_init_fn=_worker_init_fn,collate_fn=collate, pin_memory=True)

# device, model, optimzer, hyperparams, loss, mixed precision, tensorboard
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(dev)
net = GravInvNet().to(dev)
lr=1e-3
wd=0.0
opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
crit = nn.MSELoss()
scaler = torch.amp.GradScaler('cuda')
writer = SummaryWriter()

# single epoch (forward --> backward pass)
def run_epoch(ld, train=True, ema_alpha=0.1, epoch=0):
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
                writer.add_scalar("Gradients/norm", grad_norm, )
                scaler.step(opt)
                scaler.update()
            b=gz.size(0)
            li = loss.item()
            tot+=li*b
            n+=b
            ema = li if ema is None else (ema_alpha * li + (1 - ema_alpha) * ema)
            bar.set_postfix(loss=f"{li:.4f}", ema=f"{ema:.4f}")
    return tot/max(1,n)

# feature mapping
activations = {}
def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook
net.enc.first.register_forward_hook(save_activation('enc_first'))
net.enc.stages[2].register_forward_hook(save_activation('enc_mid'))

# evaluate rmse, l1, IoU, dice
def eval_metrics(ld, threshold=0.1):
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

# training loop
E=2000
min_loss = 1e-5
pbar=tqdm(range(0, E),desc="training",ncols=100)
best = float("inf")
for e in pbar:
    tr=run_epoch(ld=tr_ld, train=True, epoch=0)
    va=run_epoch(ld=va_ld, train=False, epoch=0)
    if e % 10 == 0:
        rmse_val, l1_val, iou_val, dice_val = eval_metrics(va_ld)
        writer.add_scalar("Metrics/RMSE", rmse_val, e)
        writer.add_scalar("Metrics/L1", l1_val, e)
        writer.add_scalar("Metrics/IoU", iou_val, e)
        writer.add_scalar("Metrics/Dice", dice_val, e)
        # for name, activation in activations.items():
        #     fmaps = activation[0, :min(8, activation.size(1))]
        #     writer.add_images(f"FeatureMaps/{name}", fmaps.unsqueeze(1), e)
        pbar.set_postfix(train=f"{tr:.4f}", val=f"{va:.4f}", rmse=f"{rmse_val:.4f}")
    else:
        pbar.set_postfix(train=f"{tr:.4f}", validation=f"{va:.4f}")
    writer.add_scalar("Loss/train", tr, e)
    writer.add_scalar("Loss/val",   va, e)
    writer.add_scalar("Hyperparams/LR", lr, e)
    writer.add_scalar("Hyperparams/WeightDecay", wd, e)
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

# save model
torch.save({"model": net.state_dict()}, "checkpoints/final.pt")
