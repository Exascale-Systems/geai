device = 'cuda:7'
lr = 1e-3
wd = 0.0
batch_size = 8
max_epochs = 200
eval_interval = 10

import pathlib
from tqdm.auto import tqdm
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from src.data import MasterDataset, collate, _worker_init_fn, data_prep
from src.utils import compute_stats, denorm
from src.data import make_transform
from src.metrics import eval_metrics
from model import GravInvNet

dev = torch.device(device); print(dev)
net = GravInvNet().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
crit = nn.MSELoss()
scaler = torch.amp.GradScaler('cuda')
writer = SummaryWriter('logs')

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

def train(tr_ld: DataLoader, va_ld: DataLoader, E=max_epochs, min_loss=1e-5, stats= dict):
    pbar=tqdm(range(0, E),desc="training",ncols=100)
    best = float("inf")
    for e in pbar:
        tr=run_epoch(ld=tr_ld, train=True, epoch=e)
        va=run_epoch(ld=va_ld, train=False, epoch=e)
        writer.add_scalar("Loss/train", tr, e)
        writer.add_scalar("Loss/val",   va, e)
        writer.add_scalar("Hyperparams/LR", lr, e)
        writer.add_scalar("Hyperparams/WeightDecay", wd, e)
        if e % eval_interval == 0:
            metrics = eval_metrics(net, va_ld, stats, dev)
            writer.add_scalar("Metrics/RMSE", metrics['rmse'], e)
            writer.add_scalar("Metrics/L1", metrics['l1'], e)
            writer.add_scalar("Metrics/IoU", metrics['iou'], e)
            writer.add_scalar("Metrics/Dice", metrics['dice'], e)
            pbar.set_postfix(train=f"{tr:.4f}", val=f"{va:.4f}", rmse=f"{metrics['rmse']:.4f}")
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
