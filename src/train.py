import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from src.load import MasterDataset, _worker_init_fn
from src.transform import make_transform
from src.nn import GravInvNet

# data
ds = MasterDataset("data/master.h5")
ds.transform = make_transform(ds.shape_cells)
def collate(b): xs, ys, ms = zip(*[(x,y,m) for x,y,m,_ in b]); return torch.stack(xs), torch.stack(ys), torch.stack(ms)
g = torch.Generator().manual_seed(0) 
n=len(ds); n_tr=max(1,int(0.8*n))
n_va=n-n_tr
tr_ds, va_ds = random_split(ds,[n_tr,n_va],generator=g)
bs=min(8,n_tr)
tr_ld = DataLoader(tr_ds,batch_size=bs,shuffle=True,num_workers=2,worker_init_fn=_worker_init_fn,collate_fn=collate, pin_memory=True)
va_ld = DataLoader(va_ds,batch_size=bs,shuffle=False,num_workers=2,worker_init_fn=_worker_init_fn,collate_fn=collate, pin_memory=True)

# model, hyperparams, loss
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = GravInvNet().to(dev)
# for m in net.modules():
#     if isinstance(m,(nn.BatchNorm2d,nn.BatchNorm3d)): m.eval()
opt = torch.optim.SGD(net.parameters(),lr=1e-2,momentum=0.9,weight_decay=1e-4)
# masked_mse=lambda p,t,m:((p-t)[m]**2).mean()
def masked_mse(pred, tgt, mask, eps=1e-12):
    if mask.dtype != torch.bool:
        mask = mask > 0.5  # accept 0/1 or float masks
    diff2 = (pred - tgt)**2
    num = (diff2 * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den

# single epoch (forward --> backward pass)
def run_epoch(ld,train=True, ema_alpha=0.1):
    ema=None
    net.train(train)
    tot,n=0.0,0
    bar = tqdm(ld, leave=False, desc=("train" if train else "valid"), ncols=100)
    for gz,tgt,mask in bar:
        gz,tgt,mask=gz.to(dev, non_blocking=True),tgt.to(dev, non_blocking=True),mask.to(dev, non_blocking=True)
        pred=net(gz)
        loss=masked_mse(pred,tgt,mask)
        if train:
            opt.zero_grad(set_to_none=True); 
            loss.backward(); 
            nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        b=gz.size(0)
        tot+=loss.item()*b
        n+=b
        val = loss.item()
        ema = val if ema is None else (ema_alpha*val + (1-ema_alpha)*ema)
        bar.set_postfix(loss=f"{val:.4f}", ema=f"{ema:.4f}")
    return tot/max(1,n)

# training loop
E=10; tr_hist,va_hist=[],[]
pbar=tqdm(range(1, E+1),desc="training",ncols=100)
for e in pbar:
    tr=run_epoch(tr_ld,True)
    tr_hist.append(tr)
    pbar.set_postfix(train=f"{tr:.4f}")

# save model
torch.save({"model": net.state_dict()}, "checkpoints/best.pt")

# plot training history
plt.plot(tr_hist,label='train')
plt.xlabel('epoch'); plt.ylabel('masked MSE'); plt.legend(); plt.tight_layout(); plt.show()
