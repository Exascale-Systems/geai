import logging

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.evaluation.nn import eval_nn
from src.nn.loss_functions import DiceLoss

logger = logging.getLogger(__name__)

LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "mse": nn.MSELoss,
    "dice": DiceLoss,
}


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
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
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
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    lr = config.get("lr", 3e-4)
    wd = config.get("wd", 0.0)
    max_epochs = config.get("max_epochs", 200)
    min_loss = config.get("min_loss", 1e-6)
    eval_interval = config.get("eval_interval", 10)
    loss_fn_name = config.get("loss_function", "mse")
    log_dir = config.get("log_dir", "logs")
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    model_name = config.get("model_name", "model")

    dev = torch.device(device)
    logger.info("Device: %s", dev)
    net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    crit_cls = LOSS_REGISTRY.get(loss_fn_name)
    if crit_cls is None:
        raise ValueError(f"Unknown loss function '{loss_fn_name}'. Choose from: {list(LOSS_REGISTRY)}")
    crit = crit_cls()
    logger.info("Loss function: %s", loss_fn_name)

    scaler = GradScaler()
    writer = SummaryWriter(log_dir)

    pbar = tqdm(range(0, max_epochs), desc="training", ncols=100)
    best = float("inf")

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

        if va < best:
            best = va
            torch.save({"model": net.state_dict()}, f"{checkpoint_dir}/best.pt")
        if va < min_loss:
            logger.info("Reached target loss %.6f at epoch %d", va, e)
            break

    writer.flush()
    writer.close()
    torch.save({"model": net.state_dict()}, f"{checkpoint_dir}/{model_name}_final.pt")

    final_metrics = eval_nn(net, va_ld, stats, dev)
    final_metrics["best_val_loss"] = best
    return final_metrics
