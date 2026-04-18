"""
Classification and severity heads for aircraft damage CausalVAE.

MultiLabelHead  — binary classification for 4 observable damage concepts
SeverityHead    — regression for total damage instance count (0, 1, 2, …)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


class MultiLabelHead(nn.Module):
    """
    MLP classification head for 4 binary damage concepts.

    Args:
        z_dim:      int  input dimension (total latent dim, e.g. 32)
        n_classes:  int  number of output classes (default 4)
        hidden_dim: int  hidden layer width
    """
    def __init__(self, z_dim: int, n_classes: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), -1))


class ConceptHeads(nn.Module):
    """One linear head per observable concept, reading only its z sub-vector."""
    def __init__(self, z2_dim: int, n_observable: int):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(z2_dim, 1) for _ in range(n_observable)
        ])

    def forward(self, z_dag: torch.Tensor) -> torch.Tensor:
        # z_dag: (B, Z1_DIM, Z2_DIM) — use only first n_observable sub-vectors
        logits = [self.heads[i](z_dag[:, i, :]) for i in range(len(self.heads))]
        return torch.cat(logits, dim=-1)  # (B, n_observable)


class SeverityHead(nn.Module):
    """
    Regression head for total damage instance count.

    Predicts the raw count of individual damage detections across all types
    (e.g. 3 cracks = 3, 1 crack + 1 dent = 2).  Output is an unbounded float;
    apply max(0, round(pred)) for integer interpretation.

    Args:
        z_dim: int  input dimension (total latent dim, e.g. 32)
    """
    def __init__(self, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, z_dim) flattened latent code
        Returns:
            (batch,) predicted instance counts (float)
        """
        return self.net(z.view(z.size(0), -1)).squeeze(-1)


# ── Loss functions ─────────────────────────────────────────────────────────────

def multilabel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor = None,
) -> torch.Tensor:
    """Binary cross-entropy for multi-label concept classification."""
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight
    )


def severity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Smooth L1 (Huber) loss for damage instance count regression.
    Robust to occasional high-count outliers.
    """
    return F.smooth_l1_loss(pred, target.float())


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    class_names: list = None,
    sev_pred: torch.Tensor = None,
    sev_target: torch.Tensor = None,
) -> dict:
    """
    Per-class and macro F1, precision, recall for damage concepts.
    Optionally adds severity MAE and accuracy if sev_pred/sev_target are given.

    Args:
        logits:     (N, n_classes)  raw logits from MultiLabelHead
        targets:    (N, n_classes)  binary labels
        threshold:  float           sigmoid threshold
        class_names: list[str]      for named keys in output dict
        sev_pred:   (N,) float      predicted instance counts
        sev_target: (N,) float      true instance counts

    Returns:
        dict with f1_macro, precision_macro, recall_macro, per-class scores,
        and optionally sev_mae, sev_acc.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(int)
        tgts  = targets.cpu().numpy().astype(int)

    n_classes    = preds.shape[1]
    default_names = class_names or [f'class_{i}' for i in range(n_classes)]

    f1_per   = f1_score(tgts, preds, average=None, zero_division=0)
    prec_per = precision_score(tgts, preds, average=None, zero_division=0)
    rec_per  = recall_score(tgts, preds, average=None, zero_division=0)

    result = {
        'f1_macro':        float(np.mean(f1_per)),
        'precision_macro': float(np.mean(prec_per)),
        'recall_macro':    float(np.mean(rec_per)),
        'f1_per_class':    f1_per.tolist(),
    }
    for i, name in enumerate(default_names):
        result[f'f1_{name}']        = float(f1_per[i])
        result[f'precision_{name}'] = float(prec_per[i])
        result[f'recall_{name}']    = float(rec_per[i])

    # Severity metrics
    if sev_pred is not None and sev_target is not None:
        sp = sev_pred.cpu().float().numpy()
        st = sev_target.cpu().float().numpy()
        result['sev_mae'] = float(np.mean(np.abs(sp - st)))
        result['sev_acc'] = float(np.mean(np.round(sp) == np.round(st)))

    return result
