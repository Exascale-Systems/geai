import numpy as np
import torch


class TorchMetrics:
    def __init__(self, stats, threshold=0.1):
        self.stats = stats
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self.sum_se = 0.0
        self.sum_ae = 0.0
        self.intersection = 0.0
        self.union = 0.0
        self.true_sum = 0.0
        self.pred_sum = 0.0
        self.n_samples = 0

    @torch.no_grad()
    def update(self, net, gz, tgt, denorm_fn):
        """Accumulate metrics for a single batch."""
        pred = net(gz)
        # Skip denormalization since data is not normalized
        pred_denorm = pred  # denorm_fn(pred, self.stats)
        tgt_denorm = tgt  # denorm_fn(tgt, self.stats)
        diff = tgt_denorm - pred_denorm
        self.sum_se += torch.sum(diff**2).item()
        self.sum_ae += torch.sum(torch.abs(diff)).item()
        self.n_samples += diff.numel()
        true_binary = tgt_denorm > self.threshold
        pred_binary = pred_denorm > self.threshold
        self.intersection += torch.sum(true_binary & pred_binary).item()
        self.union += torch.sum(true_binary | pred_binary).item()
        self.true_sum += torch.sum(true_binary).item()
        self.pred_sum += torch.sum(pred_binary).item()

    def compute(self):
        """Return final metrics: RMSE, L1, IoU, Dice."""
        rmse = (self.sum_se / self.n_samples) ** 0.5 if self.n_samples > 0 else 0
        l1 = self.sum_ae / self.n_samples if self.n_samples > 0 else 0
        iou = self.intersection / self.union if self.union > 0 else 1.0
        dice = (
            (2 * self.intersection) / (self.true_sum + self.pred_sum)
            if (self.true_sum + self.pred_sum) > 0
            else 1.0
        )
        return {"RMSE": rmse, "L1": l1, "IoU": iou, "Dice": dice}


class NumpyMetrics:
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self.sum_se = 0.0
        self.sum_ae = 0.0
        self.intersection = 0
        self.union = 0
        self.true_sum = 0
        self.pred_sum = 0
        self.n_samples = 0

    def update(self, true: np.ndarray, pred: np.ndarray):
        """Accumulate metrics for true vs predicted arrays."""
        diff = true - pred
        self.sum_se += np.sum(diff * diff)
        self.sum_ae += np.sum(np.abs(diff))
        self.n_samples += diff.size

        true_binary = true > self.threshold
        pred_binary = pred > self.threshold
        self.intersection += np.sum(true_binary & pred_binary)
        self.union += np.sum(true_binary | pred_binary)
        self.true_sum += np.sum(true_binary)
        self.pred_sum += np.sum(pred_binary)

    def compute(self):
        """Return final metrics: RMSE, L1, IoU, Dice."""
        rmse = (self.sum_se / self.n_samples) ** 0.5 if self.n_samples > 0 else 0
        l1 = self.sum_ae / self.n_samples if self.n_samples > 0 else 0
        iou = self.intersection / self.union if self.union > 0 else 1.0
        dice = (
            (2 * self.intersection) / (self.true_sum + self.pred_sum)
            if (self.true_sum + self.pred_sum) > 0
            else 1.0
        )
        return {"RMSE": rmse, "L1": l1, "IoU": iou, "Dice": dice}
