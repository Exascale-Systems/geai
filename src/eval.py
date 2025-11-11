import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.data import data_prep
from src.gen.StructuralGeo_gen import gravity_survey, init_model, create_mesh
from src.utils import denorm
from src.utils import load_model
from simpeg.potential_fields import gravity
from simpeg import data, inverse_problem, regularization, optimization, directives, inversion, data_misfit

def eval_metrics_nn(model, dataloader: DataLoader, stats: dict, device: str = "cuda:7", threshold=0.1, split_name="", show_progress=False, sample_idx=None):
    """
    Evaluate metrics on a dataloader using the given neural network model.
    If sample_idx is provided, evaluates only that specific sample.
    """
    model.eval()
    sum_se, sum_ae, n_samples = 0.0, 0.0, 0
    intersection, union, true_sum, pred_sum = 0, 0, 0, 0
    if sample_idx is not None:
        dataset = dataloader.dataset
        # if hasattr(dataset, 'indices'):
        #     actual_idx = dataset.indices[sample_idx]
        #     gz, tgt, _, _ = dataset.dataset[actual_idx]
        # else:
        gz, tgt, _, _ = dataset[sample_idx]
        gz, tgt = gz.unsqueeze(0).to(device), tgt.unsqueeze(0).to(device)
        with torch.no_grad():
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
    else:
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
        'n_samples': 1 if sample_idx is not None else len(dataloader.dataset)
    }

def eval_metrics_bayesian(dataloader: DataLoader, stats: dict, threshold=0.1, split_name="", show_progress=False, max_samples=None):
    """
    Evaluate metrics on a dataloader using Bayesian inversion.
    """
    sum_se, sum_ae, n_samples = 0.0, 0.0, 0
    intersection, union, true_sum, pred_sum = 0, 0, 0, 0
    iterator = dataloader
    if show_progress:
        desc = f"Evaluating {split_name}" if split_name else "Evaluating"
        iterator = tqdm(dataloader, desc=desc, leave=False, ncols=80)
    for batch_idx, (gz, tgt) in enumerate(iterator):
        if max_samples is not None and batch_idx >= max_samples:
            break
        dataset = dataloader.dataset.dataset
        sample_idx = dataloader.dataset.indices[batch_idx]
        original_transform = dataset.transform
        dataset.transform = None
        sample_data = dataset[sample_idx]
        dataset.transform = original_transform
        rx = sample_data['receiver_locations'].cpu().numpy()
        ind = sample_data['ind_active'].cpu().numpy().astype(bool)
        mesh = create_mesh(bounds=((0, dataset.shape_cells[0]*dataset.hx[0]), 
                                  (0, dataset.shape_cells[1]*dataset.hy[0]), 
                                  (0, dataset.shape_cells[2]*dataset.hz[0])), 
                          resolution=dataset.shape_cells)
        _, survey = gravity_survey(rx, components=("gz",))
        _, _, model_map, _ = init_model(mesh, rx, 0)
        sim = gravity.simulation.Simulation3DIntegral(
            survey=survey, mesh=mesh, rhoMap=model_map, active_cells=ind, engine="choclo"
        )
        gz_denorm = denorm(gz.squeeze(), stats, data_type="gz").cpu().numpy().flatten()
        tgt_denorm = denorm(tgt.squeeze(), stats).cpu().numpy().flatten()
        y = np.zeros(int(ind.sum()))
        data_object = data.Data(survey, dobs=gz_denorm, noise_floor=1e-6)
        dmis = data_misfit.L2DataMisfit(data=data_object, simulation=sim)
        reg = regularization.WeightedLeastSquares(mesh, active_cells=ind, mapping=model_map)
        opt = optimization.ProjectedGNCG(maxIter=20, lower=-1.0, upper=1.0, maxIterLS=20, cg_maxiter=10, cg_atol=1e-3)
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        directives_list = [
            directives.UpdateSensitivityWeights(every_iteration=False),
            directives.BetaEstimate_ByEig(beta0_ratio=1e1),
            directives.BetaSchedule(coolingFactor=5, coolingRate=1),
            directives.SaveOutputEveryIteration(on_disk=False),
            directives.UpdatePreconditioner(),
            directives.TargetMisfit(chifact=1),
        ]
        inv = inversion.BaseInversion(inv_prob, directives_list)
        try:
            pred = inv.run(y)
            diff = tgt_denorm - pred
            sum_se += np.sum(diff * diff)
            sum_ae += np.sum(np.abs(diff))
            n_samples += diff.size
            true_binary = tgt_denorm > threshold
            pred_binary = pred > threshold
            intersection += np.sum(true_binary & pred_binary)
            union += np.sum(true_binary | pred_binary)
            true_sum += np.sum(true_binary)
            pred_sum += np.sum(pred_binary)
        except:
            continue
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

def evaluate_model(method: str = "nn", model_path: str = None, dataset_name: str = None, 
                   threshold: float = 0.1, batch_size: int = None, device: str = "cpu", max_samples: int = None, sample_idx: int = None):
    """
    Evaluate either a neural network model or Bayesian inversion on validation and training data.
    If sample_idx is provided, evaluates only that specific sample.
    """
    if batch_size is None:
        batch_size = 8 if method == "nn" else 1
    tr_ld, va_ld,  stats = data_prep(dataset_name, dataset_name, batch_size, load_splits=True)
    if method == "nn":
        if model_path is None:
            raise ValueError("model_path is required for neural network evaluation")
        net, device = load_model(model_path, torch.device(device))
        print("\nEvaluating neural network model performance...")
        results = {}
        if sample_idx is not None:
            results['sample'] = eval_metrics_nn(net, va_ld, stats, device, threshold, f"sample {sample_idx}", show_progress=True, sample_idx=sample_idx)
        else:
            results['train'] = eval_metrics_nn(net, tr_ld, stats, device, threshold, "training", show_progress=True)
            results['validation'] = eval_metrics_nn(net, va_ld, stats, device, threshold, "validation", show_progress=True)
    elif method == "bayesian":
        print("\nEvaluating Bayesian inversion performance...")
        results = {}
        results['train'] = eval_metrics_bayesian(tr_ld, stats, threshold, "training", show_progress=True, max_samples=max_samples or 5)
        results['validation'] = eval_metrics_bayesian(va_ld, stats, threshold, "validation", show_progress=True, max_samples=max_samples or 5)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'nn' or 'bayesian'")
    return results

def print_results(results):
    """
    Print evaluation results in a formatted table
    """
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
    """
    Main function to run evaluation. By default runs neural network evaluation.
    Change method parameter to "bayesian" to run Bayesian evaluation.
    """
    results = evaluate_model(
        method="nn",
        model_path='models/single_block.pt',
        dataset_name='single_block',
        threshold=0.5,
        batch_size=8,
        device="cuda:7",
        sample_idx=3
    )
    # results = evaluate_model(
    #     method="bayesian",
    #     dataset_name='single_block',
    #     threshold=0.5,
    #     max_samples=1
    # )
    print_results(results)

if __name__ == "__main__":
    main()