import torch
import pandas as pd
from typing import List

# accuracy
def compute_acc(gt_df, predicted_scores, device):
    # Convert inputs to tensors on specified device
    g = torch.tensor(gt_df['score'].values, dtype=torch.float32, device=device)
    s = torch.tensor(predicted_scores, dtype=torch.float32, device=device)

    # Compute pairwise differences
    g_diff = g.unsqueeze(0) - g.unsqueeze(1)
    s_diff = s.unsqueeze(0) - s.unsqueeze(1)

    # Create masks
    gt_mask = g_diff > 0
    pred_mask = s_diff > 0

    # Compute correct and total comparisons
    correct = torch.sum(gt_mask & pred_mask)
    total = torch.sum(gt_mask)

    if total.item() == 0:
        return 0.0
    return (correct / total).item()


# weighted accuracy
def compute_weighted_acc(gt_df: pd.DataFrame,
                         predicted_scores: List[float],
                         device: torch.device,
                         alpha: float = 1.0) -> float:
    """
    Compute weighted pairwise accuracy.

    Args:
        gt_df: DataFrame with column 'score' for ground-truth rewards.
        predicted_scores: list or 1D tensor of model scores, length n.
        device: torch device to run on.
        alpha: exponent on |reward difference| to form weights.

    Returns:
        Weighted pairwise accuracy in [0,1].
    """
    # Ground-truth rewards (n,)
    g = torch.tensor(gt_df['score'].values,
                     dtype=torch.float32,
                     device=device)
    # Predicted scores as a tensor on device (n,)
    s = torch.tensor(predicted_scores,
                     dtype=torch.float32,
                     device=device)

    # Pairwise difference matrices (n,n)
    g_diff = g.unsqueeze(0) - g.unsqueeze(1)
    s_diff = s.unsqueeze(0) - s.unsqueeze(1)

    # Masks: true ordering vs. predicted ordering
    gt_mask   = g_diff > 0     # i > j in ground truth
    pred_mask = s_diff > 0     # i > j in prediction

    # Weights based on reward gap
    w = torch.abs(g_diff) ** alpha

    # Create an i<j mask to count each pair only once
    n = g.shape[0]
    idxs = torch.triu_indices(n, n, offset=1, device=device)  # all i<j
    pair_mask = torch.zeros_like(w, dtype=torch.bool)
    pair_mask[idxs[0], idxs[1]] = True

    # Apply both the i<j constraint and the gt_mask
    valid_mask = pair_mask & gt_mask

    weighted_correct = torch.sum(w[valid_mask & pred_mask])
    weighted_total   = torch.sum(w[valid_mask])

    if weighted_total.item() == 0:
        return 0.0
    return (weighted_correct / weighted_total).item()
