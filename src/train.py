import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import pathlib
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from src.load import MasterDataset, _worker_init_fn
from src.transform import make_transform
from src.nn import GravInvNet
from src.normalize import compute_stats

# data
stats = compute_stats("data/singleblock.h5")
ds = MasterDataset("data/singleblock.h5")
ds.transform = make_transform(ds.shape_cells, stats)
def collate(b):
    xs, ys, _ms = zip(*[(x,y,m) for x,y,m,_ in b])
    return torch.stack(xs), torch.stack(ys)
g = torch.Generator().manual_seed(0) 
n=len(ds) 
n_tr=max(1,int(0.8*n)) # 80% train, 20% val
n_va=n-n_tr
tr_ds, va_ds = random_split(ds,[n_tr,n_va],generator=g)
pathlib.Path("training/split").mkdir(parents=True, exist_ok=True)
np.savez("training/split/idx_init.npz", tr=np.array(tr_ds.indices), va=np.array(va_ds.indices))

bs=min(8,n_tr)
tr_ld = DataLoader(tr_ds,batch_size=bs,shuffle=True,num_workers=2,worker_init_fn=_worker_init_fn,collate_fn=collate, pin_memory=True)
va_ld = DataLoader(va_ds,batch_size=bs,shuffle=False,num_workers=2,worker_init_fn=_worker_init_fn,collate_fn=collate, pin_memory=True)

# device, model, optimzer, hyperparams, loss, mixed precision, tensorboard
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = GravInvNet().to(dev)
lr=1e-3
wd=0.0
opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
crit = nn.MSELoss()
scaler = torch.amp.GradScaler('cuda')
writer = SummaryWriter()

# single epoch (forward --> backward pass)
def run_epoch(ld, train=True, ema_alpha=0.1):
    net.train() if train else net.eval()
    ema,tot,n=None,0.0,0
    with torch.enable_grad() if train else torch.no_grad():
        bar = tqdm(ld, leave=False, ncols=100)
        for gz,tgt in bar:
            gz,tgt=gz.to(dev, non_blocking=True),tgt.to(dev, non_blocking=True)
            if train:
                opt.zero_grad(set_to_none=True)
            pred=net(gz)
            loss = crit(pred, tgt
                            )
            if train:
                scaler.scale(loss).backward() 
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(),1.0)
                scaler.step(opt)
                scaler.update()
            b=gz.size(0)
            li = loss.item()
            tot+=li*b
            n+=b
            ema = li if ema is None else (ema_alpha * li + (1 - ema_alpha) * ema)
            bar.set_postfix(loss=f"{li:.4f}", ema=f"{ema:.4f}")
    return tot/max(1,n)

# training loop
E=2000; 
min_loss = 1e-5
pbar=tqdm(range(0, E),desc="training",ncols=100)
for e in pbar:
    tr=run_epoch(ld=tr_ld, train=True)
    va=run_epoch(ld=va_ld, train=False)
    writer.add_scalar("Loss/train", tr, e)
    writer.add_scalar("Loss/val",   va, e)
    writer.add_scalar("Hyperparams/LR", lr, e)
    writer.add_scalar("Hyperparams/WeightDecay", wd, e)
    torch.save({'model': net.state_dict()}, 'training/checkpoints/best.pt')
    pbar.set_postfix(train=f"{tr:.4f}")
    if va < min_loss:
        print(f"Reached target loss {tr:.6f} at epoch {e}")
        break
writer.flush()
writer.close()

# save model
torch.save({"model": net.state_dict()}, "training/checkpoints/best.pt")
