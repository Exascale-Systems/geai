import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from src.load import MasterDataset, _worker_init_fn
from src.transform import make_transform
from src.nn import GravInvNet
from src.normalize import compute_stats

# data
stats = compute_stats("data/master.h5")
ds = MasterDataset("data/master.h5")
ds.transform = make_transform(ds.shape_cells, stats)
def collate(b):
    xs, ys, _ms = zip(*[(x,y,m) for x,y,m,_ in b])
    return torch.stack(xs), torch.stack(ys)
g = torch.Generator().manual_seed(0) 
n=len(ds) 
n_tr=max(1,int(1*n))
n_va=n-n_tr
tr_ds, va_ds = random_split(ds,[n_tr,n_va],generator=g)
bs=min(4,n_tr)
tr_ld = DataLoader(tr_ds,batch_size=bs,shuffle=True,num_workers=2,worker_init_fn=_worker_init_fn,collate_fn=collate, pin_memory=True)
va_ld = DataLoader(va_ds,batch_size=bs,shuffle=False,num_workers=2,worker_init_fn=_worker_init_fn,collate_fn=collate, pin_memory=True)

# model, hyperparams
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = GravInvNet().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0)
crit = nn.MSELoss()

# single epoch (forward --> backward pass)
def run_epoch(ld, ema_alpha=0.1):
    ema=None
    tot,n=0.0,0
    net.train()
    bar = tqdm(ld, leave=False, ncols=100)
    for gz,tgt in bar:
        gz,tgt=gz.to(dev, non_blocking=True),tgt.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True) 
        pred=net(gz)
        loss = crit(pred, tgt)
        loss.backward(); 
        nn.utils.clip_grad_norm_(net.parameters(),1.0)
        opt.step()
        b=gz.size(0)
        li = loss.item()
        tot+=li*b
        n+=b
        ema = li if ema is None else (ema_alpha * li + (1 - ema_alpha) * ema)
        bar.set_postfix(loss=f"{li:.4f}", ema=f"{ema:.4f}")
    return tot/max(1,n)

# training loop
E=20000; 
tr_hist=[None]*E; 
min_loss = 1e-4
pbar=tqdm(range(0, E),desc="training",ncols=100)
for e in pbar:
    tr=run_epoch(ld=tr_ld)
    tr_hist[e]=tr
    pbar.set_postfix(train=f"{tr:.4f}")
    if tr < min_loss:
        print(f"Reached target loss {tr:.6f} at epoch {e}")
        torch.save({'model': net.state_dict()}, 'checkpoints/overfit.pt')
        break

# save model
torch.save({"model": net.state_dict()}, "checkpoints/overfit.pt")

# plot training history
plt.plot(tr_hist,label='train')
plt.xlabel('epoch'); plt.ylabel('masked MSE'); plt.legend(); plt.tight_layout(); plt.show()
