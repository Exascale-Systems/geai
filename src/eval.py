import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.data import data_prep
from src.utils import load_model
from src.transform import denorm

def eval_metrics(model, dataloader: DataLoader, stats: dict, device, threshold=0.1, split_name="", show_progress=False):
    """
    Evaluate metrics on a dataloader using the given model.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader to evaluate on
        stats: Statistics dict for denormalization
        device: Device to run evaluation on
        threshold: Threshold for binary classification metrics
        split_name: Name of split for progress bar description
        show_progress: Whether to show progress bar
    
    Returns:
        dict: Dictionary containing rmse, l1, iou, dice metrics and n_samples
    """
    model.eval()
    sum_se, sum_ae, n_samples = 0.0, 0.0, 0
    intersection, union, true_sum, pred_sum = 0, 0, 0, 0
    iterator = dataloader
    if show_progress:
        desc = f"Evaluating {split_name}" if split_name else "Evaluating"
        iterator = tqdm(dataloader, desc=desc, leave=False, ncols=80)
    with torch.no_grad():
        for gz, tgt in iterator:
            gz, tgt = gz.to(device), tgt.to(device)
            pred = model(gz)
            pred_denorm = denorm(pred, stats)
            tgt_denorm = denorm(tgt, stats)
            diff = tgt_denorm - pred_denorm
            sum_se += torch.sum(diff * diff).item()
            sum_ae += torch.sum(torch.abs(diff)).item()
            n_samples += diff.numel()            
            true_binary = tgt_denorm > threshold
            pred_binary = pred_denorm > threshold
            intersection += torch.sum(true_binary & pred_binary).item()
            union += torch.sum(true_binary | pred_binary).item()
            true_sum += torch.sum(true_binary).item()
            pred_sum += torch.sum(pred_binary).item()
    rmse_val = (sum_se / n_samples) ** 0.5
    l1_val = sum_ae / n_samples
    iou_val = intersection / union if union > 0 else 1.0
    dice_val = (2 * intersection) / (true_sum + pred_sum) if (true_sum + pred_sum) > 0 else 1.0
    return {
        'rmse': rmse_val,
        'l1': l1_val,
        'iou': iou_val,
        'dice': dice_val,
        'n_samples': len(dataloader.dataset)
    }

def evaluate_model(model_path: str = None, dataset_name: str = None, 
                   threshold: float = 0.1, batch_size: int = 8, device: str ="cpu"):
    """
    Evaluate a trained model on validation and training data.
    """
    net, device = load_model(model_path, torch.device(device))
    tr_ld, va_ld, stats = data_prep(dataset_name, dataset_name, batch_size, load_splits=True)
    print("\nEvaluating model performance...")
    results = {}
    results['train'] = eval_metrics(net, tr_ld, stats, device, threshold, "training", show_progress=True)
    results['validation'] = eval_metrics(net, va_ld, stats, device, threshold, "validation", show_progress=True)
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
    results = evaluate_model(
        model_path='models/single_block.pt',
        dataset_name='single_block',
        threshold=0.1,
        batch_size=8,
        device="cpu"
    )
    print_results(results)

if __name__ == "__main__":
    main()