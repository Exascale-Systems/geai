import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.load import MasterDataset, _worker_init_fn
from src.transforms import make_transform
from src.model import GravInvNet

# data
ds = MasterDataset("data/master.h5")
ds.transform = make_transform(ds.shape_cells)
def collate(b): xs, ys, ms = zip(*[(x,y,m) for x,y,m,_ in b]); return torch.stack(xs), torch.stack(ys), torch.stack(ms)
g = torch.Generator().manual_seed(0) 
n=len(ds); n_tr=max(1,int(0.8*n))
n_va=n-n_tr
tr_ds, va_ds = random_split(ds,[n_tr,n_va],generator=g)
bs=min(4,n_tr)
tr_ld = DataLoader(tr_ds,batch_size=bs,shuffle=True,num_workers=2,worker_init_fn=_worker_init_fn,collate_fn=collate)
va_ld = DataLoader(va_ds,batch_size=bs,shuffle=False,num_workers=2,worker_init_fn=_worker_init_fn,collate_fn=collate)

# model, hyperparams, loss
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = GravInvNet().to(dev)
for m in net.modules():
    if isinstance(m,(nn.BatchNorm2d,nn.BatchNorm3d)): m.eval()
opt = torch.optim.SGD(net.parameters(),lr=1e-2,momentum=0.9,weight_decay=1e-4)
masked_mse=lambda p,t,m:((p-t)[m]**2).mean()

# single epoch (forward --> backward pass)
def run_epoch(ld,train=True):
    net.train(train)
    for m in net.modules():
        if isinstance(m,(nn.BatchNorm2d,nn.BatchNorm3d)): m.eval()
    tot,n=0.0,0
    for gz,tgt,mask in ld:
        gz,tgt,mask=gz.to(dev),tgt.to(dev),mask.to(dev)
        pred=net(gz); loss=masked_mse(pred,tgt,mask)
        if train:
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        b=gz.size(0); tot+=loss.item()*b; n+=b
    return tot/max(1,n)

E=100; tr_hist,va_hist=[],[]
pbar=tqdm(range(E),desc="training",ncols=100)
for e in pbar:
    tr=run_epoch(tr_ld,True); va=run_epoch(va_ld,False)
    tr_hist.append(tr); va_hist.append(va)
    pbar.set_postfix(train=f"{tr:.4f}",val=f"{va:.4f}")

# plot training history
plt.plot(tr_hist,label='train'); plt.plot(va_hist,label='val')
plt.xlabel('epoch'); plt.ylabel('masked MSE'); plt.legend(); plt.tight_layout(); plt.show()
