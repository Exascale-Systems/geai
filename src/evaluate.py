import pathlib
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from src.load import MasterDataset, collate, _worker_init_fn
from src.transform import *
from src.nn import GravInvNet

def evaluate_model(model_path, dataset_path, splits_path=None, threshold=0.1, batch_size=8, device=None):
    """
    Evaluate a trained model on validation and training data.
    """
    if device is None:
        device = torch.device('cuda:7')
    
    print(f"Using device: {device}")
    
    # Load dataset and compute stats
    stats = compute_stats(dataset_path)
    ds = MasterDataset(dataset_path)
    ds.transform = make_transform(ds.shape_cells, stats, noise=(0.1, 0.95))
    
    # Load model
    net = GravInvNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint['model'])
    net.eval()
    
    # Create train/val splits
    if splits_path and pathlib.Path(splits_path).exists():
        # Load existing splits
        splits = np.load(splits_path)
        tr_indices, va_indices = splits['tr'], splits['va']
        print(f"Loaded existing splits: {len(tr_indices)} train, {len(va_indices)} val")
    else:
        # Create new splits
        n = len(ds)
        n_tr = max(1, int(0.8 * n))
        n_va = n - n_tr
        g = torch.Generator().manual_seed(0)
        tr_ds, va_ds = torch.utils.data.random_split(ds, [n_tr, n_va], generator=g)
        tr_indices, va_indices = tr_ds.indices, va_ds.indices
        print(f"Created new splits: {len(tr_indices)} train, {len(va_indices)} val")
    
    # Create datasets and dataloaders
    tr_ds = Subset(ds, tr_indices)
    va_ds = Subset(ds, va_indices)
    
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=False, 
                       num_workers=2, worker_init_fn=_worker_init_fn, 
                       collate_fn=collate, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                       num_workers=2, worker_init_fn=_worker_init_fn,
                       collate_fn=collate, pin_memory=True)
    
    def eval_metrics(ld, split_name):
        """Evaluate metrics on a dataloader"""
        sum_se, sum_ae, n_samples = 0.0, 0.0, 0
        intersection, union, true_sum, pred_sum = 0, 0, 0, 0
        
        with torch.no_grad():
            pbar = tqdm(ld, desc=f"Evaluating {split_name}", leave=False, ncols=80)
            for gz, tgt in pbar:
                gz, tgt = gz.to(device), tgt.to(device)
                pred = net(gz)
                
                # Denormalize predictions and targets
                pred_denorm = denorm(pred, stats)
                tgt_denorm = denorm(tgt, stats)
                
                # Calculate RMSE and L1
                diff = tgt_denorm - pred_denorm
                sum_se += torch.sum(diff * diff).item()
                sum_ae += torch.sum(torch.abs(diff)).item()
                n_samples += diff.numel()
                
                # Calculate IoU and Dice
                true_binary = tgt_denorm > threshold
                pred_binary = pred_denorm > threshold
                intersection += torch.sum(true_binary & pred_binary).item()
                union += torch.sum(true_binary | pred_binary).item()
                true_sum += torch.sum(true_binary).item()
                pred_sum += torch.sum(pred_binary).item()
        
        # Compute final metrics
        rmse_val = (sum_se / n_samples) ** 0.5
        l1_val = sum_ae / n_samples
        iou_val = intersection / union if union > 0 else 1.0
        dice_val = (2 * intersection) / (true_sum + pred_sum) if (true_sum + pred_sum) > 0 else 1.0
        
        return {
            'rmse': rmse_val,
            'l1': l1_val,
            'iou': iou_val,
            'dice': dice_val,
            'n_samples': len(ld.dataset)
        }
    
    # Evaluate on both splits
    print("\nEvaluating model performance...")
    results = {}
    
    if len(tr_indices) > 0:
        results['train'] = eval_metrics(tr_ld, "training")
    
    if len(va_indices) > 0:
        results['validation'] = eval_metrics(va_ld, "validation")
    
    return results

def print_results(results):
    """Print evaluation results in a formatted table"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for split_name, metrics in results.items():
        print(f"\n{split_name.upper()} SET ({metrics['n_samples']} samples):")
        print("-" * 40)
        print(f"RMSE:  {metrics['rmse']:.6f}")
        print(f"L1:    {metrics['l1']:.6f}")
        print(f"IoU:   {metrics['iou']:.6f}")
        print(f"Dice:  {metrics['dice']:.6f}")
    
    print("\n" + "="*60)

def main():
    # Run evaluation with default parameters
    results = evaluate_model(
        model_path='weights/single_block.pt',
        dataset_path='datasets/single_block.h5',
        splits_path='splits/single_block.npz',
        threshold=0.1,
        batch_size=8,
        device=None
    )
    
    # Print results
    print_results(results)
    
    # Save results to file
    output_file = 'evaluation_results.npz'
    np.savez(output_file, **{f"{split}_{metric}": value 
                            for split, metrics in results.items() 
                            for metric, value in metrics.items()})
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()