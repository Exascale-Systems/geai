import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.load import MasterDataset, _worker_init_fn
from src.transforms import make_transform
from src.model import GravInvNet

# ----- Dataset + transform -----
ds = MasterDataset("data/master.h5")
ds.transform = make_transform(ds.shape_cells)

def collate(batch):
    xs, ys, masks = [], [], []
    for x, y, mask, _ in batch:
        xs.append(x)     # (1,H,W)
        ys.append(y)     # (Z,H,W)
        masks.append(mask)
    return (
        torch.stack(xs, dim=0),     # (B,1,H,W)
        torch.stack(ys, dim=0),     # (B,Z,H,W)
        torch.stack(masks, dim=0),  # (B,Z,H,W)
    )

# ----- Train/val split (10 samples -> 8/2) -----
n_total = len(ds)                 # 10
n_train = max(1, int(0.8 * n_total))
n_val   = n_total - n_train
train_ds, val_ds = random_split(ds, [n_train, n_val])

# Small batches (BatchNorm hates B=1; try B=4 if you can)
batch_size = min(4, n_train)     # e.g., 4
num_workers = 2                   # set 0 on Windows if you see issues

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, worker_init_fn=_worker_init_fn, collate_fn=collate
)

val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, worker_init_fn=_worker_init_fn, collate_fn=collate
)

# ----- Device / model / optimizer -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

net = GravInvNet().to(device)
opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# Optional: freeze BatchNorm stats (helps with tiny batches)
for m in net.modules():
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
        m.eval()                 # use running stats, don’t update them

def masked_mse(pred, target, mask):
    return ((pred - target)[mask] ** 2).mean()

# ----- One epoch helper -----
def run_epoch(loader, train=True):
    net.train(train)
    # keep BN in eval if you froze it above
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

    total, n = 0.0, 0
    for gz, tgt, mask in loader:
        gz, tgt, mask = gz.to(device), tgt.to(device), mask.to(device)
        pred = net(gz)
        loss = masked_mse(pred, tgt, mask)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        bs = gz.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(1, n)

# ----- Training loop -----
for epoch in range(5):
    tr = run_epoch(train_loader, train=True)
    va = run_epoch(val_loader,   train=False)
    print(f"epoch {epoch:02d} | train {tr:.6f} | val {va:.6f}")
