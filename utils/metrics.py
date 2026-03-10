"""
Evaluation metrics for paratope prediction (imbalanced binary classification).
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Compute binary classification metrics.
    
    Args:
        logits: (N, 2) raw model outputs
        labels: (N,) ground truth (0 or 1)
    
    Returns:
        dict with accuracy, f1, precision, recall, auc
    """
    probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    y_true = labels.detach().cpu().numpy()

    acc = (preds == y_true).mean()
    f1 = f1_score(y_true, preds, zero_division=0)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)

    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = 0.0

    return {
        'accuracy': float(acc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'auc': float(auc)
    }


def class_weights_from_dataset(dataset) -> torch.Tensor:
    """Compute inverse-frequency class weights to handle imbalance."""
    all_labels = []
    for data in dataset:
        all_labels.append(data.y.numpy())
    all_labels = np.concatenate(all_labels)
    counts = np.bincount(all_labels, minlength=2).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * 2  # normalize
    return torch.tensor(weights, dtype=torch.float)