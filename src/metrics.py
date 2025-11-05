import numpy as np

def rmse(true: np.ndarray, pred: np.ndarray):
    """Return RMSE (g/cc)"""
    return np.sqrt(np.mean((true - pred) ** 2))

def l1(true: np.ndarray, pred: np.ndarray):
    """Return L1 error (g/cc)"""
    return np.mean(np.abs(true - pred))

def iou_dice(true: np.ndarray, pred: np.ndarray, threshold: float):
    """Return IoU & Dice coefficient"""
    true_binary = true > threshold
    pred_binary = pred > threshold
    intersection = np.sum(true_binary & pred_binary)
    union = np.sum(true_binary | pred_binary)
    true_sum = np.sum(true_binary)
    pred_sum = np.sum(pred_binary)
    iou = intersection / union if union > 0 else 1.0
    dice = (2 * intersection) / (true_sum + pred_sum) if (true_sum + pred_sum) > 0 else 1.0
    return iou, dice