import os
import pathlib
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from typing import Optional, Dict, Any

from src.load import MasterDataset, collate, _worker_init_fn
from src.transform import *
from src.metrics import *
from src.nn import GravInvNet


@dataclass
class TrainerConfig:
    # model
    device: str = 'cuda:0'
    
    # optimization
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 200
    min_loss: float = 1e-5
    grad_norm_clip: float = 1.0
    
    # data
    dataset_name: str = "single_block_overfit"
    split_name: str = "test"
    train_split: float = 0.8
    batch_size: Optional[int] = None  # auto-computed if None
    num_workers: int = 2
    
    # logging
    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    eval_interval: int = 10
    
    # misc
    seed: int = 42


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs("splits", exist_ok=True)
        
        # setup model and optimization
        self.model = GravInvNet().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.MSELoss()
        self.scaler = torch.amp.GradScaler('cuda' if 'cuda' in config.device else 'cpu')
        
        # setup logging
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # metrics tracking
        self.best_val_loss = float('inf')
        self.step = 0
        
    def setup_data(self):
        """Setup data loaders and statistics"""
        # compute stats and create dataset
        dataset_path = f"datasets/{self.config.dataset_name}.h5"
        self.stats = compute_stats(dataset_path)
        
        ds = MasterDataset(dataset_path)
        ds.transform = make_transform(ds.shape_cells, self.stats, noise=(0,1))
        
        # train/val split
        n = len(ds)
        n_train = max(1, int(self.config.train_split * n))
        n_val = n - n_train
        
        batch_size = self.config.batch_size or min(8, n_train)
        
        # deterministic split
        generator = torch.Generator().manual_seed(self.config.seed)
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=generator)
        
        # save split indices
        split_path = f"splits/{self.config.split_name}.npz"
        np.savez(split_path, tr=np.array(train_ds.indices), va=np.array(val_ds.indices))
        
        # create data loaders
        common_kwargs = {
            'num_workers': self.config.num_workers,
            'worker_init_fn': _worker_init_fn,
            'collate_fn': collate,
            'pin_memory': True
        }
        
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, **common_kwargs
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, **common_kwargs
        )
        
        print(f"Dataset: {self.config.dataset_name}")
        print(f"Train samples: {n_train}, Val samples: {n_val}")
        print(f"Batch size: {batch_size}")
        
    def run_epoch(self, data_loader: DataLoader, is_train: bool = True) -> float:
        """Run a single epoch"""
        self.model.train(is_train)
        
        total_loss = 0.0
        total_samples = 0
        ema_loss = None
        ema_alpha = 0.1
        
        context = torch.enable_grad if is_train else torch.no_grad
        
        with context():
            pbar = tqdm(data_loader, leave=False, ncols=100)
            for batch_idx, (gz, target) in enumerate(pbar):
                gz = gz.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)
                
                # forward pass
                with torch.amp.autocast('cuda' if 'cuda' in str(self.device) else 'cpu'):
                    pred = self.model(gz)
                    loss = self.criterion(pred, target)
                
                if is_train:
                    # backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    
                    # gradient clipping and logging
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_norm_clip
                    )
                    self.writer.add_scalar("train/grad_norm", grad_norm, self.step)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.step += 1
                
                # update metrics
                batch_size = gz.size(0)
                loss_item = loss.item()
                total_loss += loss_item * batch_size
                total_samples += batch_size
                
                # exponential moving average for display
                ema_loss = loss_item if ema_loss is None else (
                    ema_alpha * loss_item + (1 - ema_alpha) * ema_loss
                )
                
                pbar.set_postfix(loss=f"{loss_item:.4f}", ema=f"{ema_loss:.4f}")
        
        return total_loss / max(1, total_samples)
    
    def evaluate_metrics(self, data_loader: DataLoader, threshold: float = 0.1) -> Dict[str, float]:
        """Evaluate comprehensive metrics"""
        self.model.eval()
        
        sum_se = sum_ae = n_samples = 0.0
        intersection = union = true_sum = pred_sum = 0
        
        with torch.no_grad():
            for gz, target in data_loader:
                gz = gz.to(self.device)
                target = target.to(self.device)
                
                pred = self.model(gz)
                
                # denormalize for metric computation
                pred_denorm = denorm(pred, self.stats)
                target_denorm = denorm(target, self.stats)
                
                # regression metrics
                diff = target_denorm - pred_denorm
                sum_se += torch.sum(diff * diff).item()
                sum_ae += torch.sum(torch.abs(diff)).item()
                n_samples += diff.numel()
                
                # segmentation metrics
                true_binary = target_denorm > threshold
                pred_binary = pred_denorm > threshold
                intersection += torch.sum(true_binary & pred_binary).item()
                union += torch.sum(true_binary | pred_binary).item()
                true_sum += torch.sum(true_binary).item()
                pred_sum += torch.sum(pred_binary).item()
        
        rmse = (sum_se / n_samples) ** 0.5
        l1 = sum_ae / n_samples
        iou = intersection / union if union > 0 else 1.0
        dice = (2 * intersection) / (true_sum + pred_sum) if (true_sum + pred_sum) > 0 else 1.0
        
        return {"rmse": rmse, "l1": l1, "iou": iou, "dice": dice}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # save latest
        torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, 'latest.pt'))
        
        # save best
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, 'best.pt'))
    
    def train(self):
        """Main training loop"""
        self.setup_data()
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        pbar = tqdm(range(self.config.max_epochs), desc="Training")
        
        for epoch in pbar:
            # training
            train_loss = self.run_epoch(self.train_loader, is_train=True)
            val_loss = self.run_epoch(self.val_loader, is_train=False)
            
            # logging
            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], epoch)
            
            # detailed evaluation
            if epoch % self.config.eval_interval == 0:
                metrics = self.evaluate_metrics(self.val_loader)
                
                for name, value in metrics.items():
                    self.writer.add_scalar(f"metrics/{name}", value, epoch)
                
                pbar.set_postfix(
                    train=f"{train_loss:.4f}",
                    val=f"{val_loss:.4f}", 
                    rmse=f"{metrics['rmse']:.4f}"
                )
            else:
                pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")
            
            # checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, is_best)
            
            # early stopping
            if val_loss < self.config.min_loss:
                print(f"Reached target loss {val_loss:.6f} at epoch {epoch}")
                break
        
        self.writer.close()
        print(f"Training completed. Best validation loss: {self.best_val_loss:.6f}")


def main():
    config = TrainerConfig(
        dataset_name="single_block_overfit",
        split_name="test",
        max_epochs=200,
        min_loss=1e-5,
        device='cuda:7'
    )
    
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()