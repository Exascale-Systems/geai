import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp.grad_scaler import GradScaler
from pathlib import Path
from src.training.evaluation import eval_nn


def _find_latest_epoch_checkpoint(model_name):
    """Find the latest epoch checkpoint for a model. Returns (epoch, path) or (None, None)."""
    checkpoints = list(Path("checkpoints").glob(f"{model_name}_epoch_*.pt"))
    if not checkpoints:
        return None, None

    epoch_nums = []
    for ckpt in checkpoints:
        try:
            epoch_num = int(ckpt.stem.split("_")[-1])
            epoch_nums.append((epoch_num, ckpt))
        except (ValueError, IndexError):
            pass

    if not epoch_nums:
        return None, None

    epoch, path = max(epoch_nums, key=lambda x: x[0])
    return epoch, path


def _load_checkpoint_state(checkpoint_path, net, device):
    """Load checkpoint and return the states needed to resume."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model"])
    resume_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]
    opt_state = checkpoint["optimizer"]
    scaler_state = checkpoint["scaler"]
    return resume_epoch, best_loss, opt_state, scaler_state


def _cleanup_epoch_checkpoints(model_name):
    """Remove all per-epoch checkpoints after training completes."""
    checkpoints = list(Path("checkpoints").glob(f"{model_name}_epoch_*.pt"))
    for ckpt in checkpoints:
        ckpt.unlink()


def run_epoch(
    net,
    opt,
    scaler,
    crit,
    writer,
    device,
    ld: DataLoader,
    train=True,
    ema_alpha=0.1,
    epoch=0,
):
    net.train() if train else net.eval()
    ema, tot, n = None, 0.0, 0
    with torch.enable_grad() if train else torch.no_grad():
        bar = tqdm(ld, leave=False, ncols=100)
        for batch_idx, (gz, tgt) in enumerate(bar):
            gz, tgt = (
                gz.to(device, non_blocking=True),
                tgt.to(device, non_blocking=True),
            )
            if train:
                opt.zero_grad(set_to_none=True)
            pred = net(gz)
            loss = crit(pred, tgt)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), float("inf")
                )
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                writer.add_scalar(
                    "Gradients/norm", grad_norm, epoch * len(ld) + batch_idx
                )
                scaler.step(opt)
                scaler.update()
            b = gz.size(0)
            li = loss.item()
            tot += li * b
            n += b
            ema = li if ema is None else (ema_alpha * li + (1 - ema_alpha) * ema)
            bar.set_postfix(loss=f"{li:.4f}", ema=f"{ema:.4f}")
    return tot / max(1, n)


def train_model(net, tr_ld: DataLoader, va_ld: DataLoader, stats: dict, config: dict):
    device = config.get("device", "cuda:0")
    lr = config.get("lr", 3e-4)
    wd = config.get("wd", 0.0)
    max_epochs = config.get("max_epochs", 200)
    min_loss = config.get("min_loss", 1e-6)
    eval_interval = config.get("eval_interval", 10)
    model_name = config.get("model_name", "default_model")

    dev = torch.device(device)
    print(dev)
    net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()
    scaler = GradScaler()
    writer = SummaryWriter("logs")

    # Check for incomplete run and resume if found
    start_epoch = 0
    best = float("inf")
    epoch, ckpt_path = _find_latest_epoch_checkpoint(model_name)
    if epoch is not None:
        print(f"Found checkpoint at epoch {epoch}. Resuming...")
        start_epoch, best, opt_state, scaler_state = _load_checkpoint_state(
            ckpt_path, net, dev
        )
        opt.load_state_dict(opt_state)
        scaler.load_state_dict(scaler_state)

    pbar = tqdm(range(start_epoch, max_epochs), desc="training", ncols=100)

    for e in pbar:
        tr = run_epoch(
            net, opt, scaler, crit, writer, dev, ld=tr_ld, train=True, epoch=e
        )
        va = run_epoch(
            net, opt, scaler, crit, writer, dev, ld=va_ld, train=False, epoch=e
        )

        writer.add_scalar("Loss/train", tr, e)
        writer.add_scalar("Loss/val", va, e)
        writer.add_scalar("Hyperparams/LR", lr, e)
        writer.add_scalar("Hyperparams/WeightDecay", wd, e)

        if e % eval_interval == 0:
            metrics = eval_nn(net, va_ld, stats, dev)
            writer.add_scalar("Metrics/RMSE", metrics["rmse"], e)
            writer.add_scalar("Metrics/L1", metrics["l1"], e)
            writer.add_scalar("Metrics/IoU", metrics["iou"], e)
            writer.add_scalar("Metrics/Dice", metrics["dice"], e)
            pbar.set_postfix(
                train=f"{tr:.4f}", val=f"{va:.4f}", rmse=f"{metrics['rmse']:.4f}"
            )
        else:
            pbar.set_postfix(train=f"{tr:.4f}", validation=f"{va:.4f}")

        for name, param in net.named_parameters():
            writer.add_histogram(f"Weights/{name}", param, e)
        writer.flush()

        # Save per-epoch checkpoint
        checkpoint_state = {
            "model": net.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": e,
            "best_loss": best,
        }
        current_ckpt = f"checkpoints/{model_name}_epoch_{e:03d}.pt"
        torch.save(checkpoint_state, current_ckpt)

        # Delete previous epoch checkpoint to save disk space
        if e > 0:
            prev_ckpt = Path(f"checkpoints/{model_name}_epoch_{e - 1:03d}.pt")
            if prev_ckpt.exists():
                prev_ckpt.unlink()

        if va < best:
            best = va
            torch.save({"model": net.state_dict()}, f"checkpoints/{model_name}_best.pt")
        if va < min_loss:
            print(f"Reached target loss {va:.6f} at epoch {e}")
            break

    writer.flush()
    writer.close()
    torch.save({"model": net.state_dict()}, f"checkpoints/{model_name}_final.pt")

    # Cleanup per-epoch checkpoints since training is complete
    _cleanup_epoch_checkpoints(model_name)
