import numpy as np
import torch
from .config import MAX_SCORE


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y, raw_y in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x) * MAX_SCORE
            preds.extend(pred.cpu().tolist())
            targets.extend(raw_y.tolist())

    preds = np.array(preds)
    targets = np.array(targets)
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    acc5 = float(np.mean(np.abs(preds - targets) <= 5.0))
    return {
        "mae": mae,
        "rmse": rmse,
        "acc_5pt": acc5,
        "preds": preds,
        "targets": targets
    }