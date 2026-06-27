import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy.stats import kendalltau

def crowd_bt_dist(r, beta):
    """
    Returns a sample function and the flattened probability tensor for CrowdBT.
    
    Parameters
    ----------
    r : torch.Tensor, shape (N,)
        True item scores (latent utilities).
    beta : torch.Tensor, shape (K,)
        Worker reliabilities (0 <= beta <= 1).

    Returns
    -------
    sample_fn : callable
        Function sample_fn(n) returns n samples of shape (n, 3) with
        columns (winner_index, loser_index, worker_index)
    flat_probs : torch.Tensor
        Flattened probabilities for all (winner, loser, worker) triplets.
    """
    r = r.view(-1)       # (N,)
    beta = beta.view(-1) # (K,)
    N = r.shape[0]
    K = beta.shape[0]

    # Compute pairwise probabilities for all item pairs (Bradley-Terry component)
    r_i = r.view(N, 1)      # (N,1)
    r_j = r.view(1, N)      # (1,N)
    p_win = torch.sigmoid(r_i - r_j)  # (N,N)

    # Broadcast with beta for worker reliability (CrowdBT probability)
    # P(i > j | worker k) = beta_k * P(i > j) + (1 - beta_k) * P(j > i)
    p_win_workers = beta.view(1, 1, K) * p_win.unsqueeze(-1) \
                  + (1 - beta).view(1, 1, K) * (1 - p_win).unsqueeze(-1)  # (N,N,K)

    # Mask self-comparisons
    mask = torch.eye(N, device=r.device).unsqueeze(-1)  # (N,N,1)
    probs = p_win_workers * (1 - mask)

    # Flatten probabilities for categorical sampling
    flat_probs = probs.flatten()
    total = flat_probs.sum()
    if total.item() == 0:
        raise RuntimeError("All sampling probabilities are zero (check r/beta).")
    flat_probs = flat_probs / total

    cat = torch.distributions.Categorical(flat_probs)

    def sample_fn(n):
        samples = cat.sample((n,))  # indices in [0, N*N*K)
        # Convert flat index back to (winner, loser, worker) indices
        winner = samples // (N * K)
        loser = (samples % (N * K)) // K
        worker = samples % K
        return torch.stack([winner, loser, worker], dim=-1)

    return sample_fn, flat_probs


def comparisons_to_df(comparisons):
    """
    comparisons: iterable of [win, lose, worker] (indices)
    returns DataFrame with columns left (winner), right (loser), worker, label (=left)
    """
    df = pd.DataFrame(comparisons, columns=['left', 'right', 'worker']).assign(label=lambda d: d['left'])
    return df

# def comparisons_to_df_equally_balanced(comparisons, seed=None):
#     """
#     Makes EXACTLY balanced dataset:
#     50% cases where winner is left, 50% where winner is right.
#     """

#     rng = np.random.default_rng(seed)
#     comparisons = list(comparisons)
#     n = len(comparisons)

#     # Create exactly balanced assignment
#     flags = np.array([0]*(n//2) + [1]*(n - n//2))  # 0: winner left, 1: winner right
#     rng.shuffle(flags)

#     rows = []
#     for (win, lose, worker), f in zip(comparisons, flags):
#         if f == 0:
#             left, right = win, lose
#         else:
#             left, right = lose, win

#         rows.append({
#             "left": left,
#             "right": right,
#             "worker": worker,
#             "label": win  # true winner item id
#         })

#     return pd.DataFrame(rows)

def comparisons_to_df_equally_balanced(samples, seed=None):
    rng = np.random.default_rng(seed)

    if torch.is_tensor(samples):
        samples = samples.cpu().numpy()

    n = samples.shape[0]

    flags = np.zeros(n, dtype=bool)
    flags[n // 2:] = True
    rng.shuffle(flags)

    left = samples[:, 0].copy()
    right = samples[:, 1].copy()

    left[flags] = samples[flags, 1]
    right[flags] = samples[flags, 0]

    return pd.DataFrame({
        "left": left,
        "right": right,
        "worker": samples[:, 2],
        "label": samples[:, 0]
    })

def safe_kendalltau(x, y):
    """Wrap kendalltau and replace nan with 0.0"""
    try:
        tau, p = kendalltau(x, y)
        if np.isnan(tau):
            return 0.0
        return tau
    except Exception:
        # Catch other errors, e.g., if input arrays are too short/constant
        return 0.0

def to_numpy(x):
    """Convert tensor/scalar/list to numpy array"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.array(x)
    return np.array(x)


def logistic_preference_dist(r, beta):
    """
    Constructs a sampling function from the logistic preference distribution (CrowdBT probability),
    excluding self-comparisons (i.e., where a == b).

    Args:
        r (torch.Tensor): A 1D tensor of item rewards (latent utilities) of shape (N,).
        beta (torch.Tensor): A 1D tensor of worker competencies of shape (K,).

    Returns:
        sample_fn (function): Function that samples (a, b, c) tuples where a ≠ b.
        flat_probs (torch.Tensor): Flattened sampling probabilities of shape (N*N*K,), with 0 for a == b entries.
    """
    r = r.view(-1)
    beta = beta.view(-1)
    N = r.shape[0]
    K = beta.shape[0]

    # Compute logits: beta_c * (r_a - r_b)
    r_a = r.view(N, 1, 1)
    r_b = r.view(1, N, 1)
    beta_c = beta.view(1, 1, K)

    logits = beta_c * (r_a - r_b)  # (N, N, K)
    probs = torch.sigmoid(logits)  # P(a > b | worker c)

    # Set probs[a == b] = 0 to exclude self-comparisons
    mask = torch.eye(N, device=r.device).unsqueeze(-1)  # (N, N, 1)
    probs = probs * (1 - mask)  # zero out diagonal

    # Normalize
    probs = probs / probs.sum()
    flat_probs = probs.flatten()
    cat = torch.distributions.Categorical(flat_probs)

    def sample_fn(n):
        """
        Samples n (a, b, c) comparisons where a ≠ b.

        Returns:
            torch.Tensor: (n, 3) with rows (winner_index, loser_index, worker_index)
        """
        samples = cat.sample((n,))
        # Convert flat index back to (winner, loser, worker) indices
        a = samples // (N * K)
        b = (samples % (N * K)) // K
        c = samples % K
        return torch.stack([a, b, c], dim=-1)

    return sample_fn, flat_probs


import torch
import random

def memory_efficient_sample(r, beta, n_samples, seed=None):
    """
    Samples comparisons without creating a massive N*N*K probability table.
    
    Args:
        r: tensor of rewards, shape (N,)
        beta: tensor of worker competencies, shape (K,)
        n_samples: number of comparisons to sample
        seed: optional int for reproducibility
    """

    # --- Set seeds for reproducibility ---
    if seed is not None:
        random.seed(seed)                     # Python RNG
        torch.manual_seed(seed)              # PyTorch CPU RNG
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)    # current GPU
            torch.cuda.manual_seed_all(seed)  # all GPUs

    N = r.shape[0]
    K = beta.shape[0]

    # 1. Randomly pick pairs (a, b) and workers (c)
    a = torch.randint(0, N, (n_samples,), device=r.device)
    b = torch.randint(0, N, (n_samples,), device=r.device)

    # Ensure a != b
    mask = (a == b)
    while mask.any():
        b[mask] = torch.randint(0, N, (mask.sum(),), device=r.device)
        mask = (a == b)

    c = torch.randint(0, K, (n_samples,), device=r.device)

    # 2. Get rewards and competencies
    r_a = r[a]
    r_b = r[b]
    beta_c = beta[c]

    # 3. Compute probabilities
    logits = beta_c * (r_a - r_b)
    probs = torch.sigmoid(logits)

    # 4. Sample winner
    random_vals = torch.rand(n_samples, device=r.device)
    winner_is_a = random_vals < probs

    winners = torch.where(winner_is_a, a, b)
    losers = torch.where(winner_is_a, b, a)

    return torch.stack([winners, losers, c], dim=-1)

def memory_efficient_sample_crowdbt(r, beta, n_samples, seed=None):
    """
    Memory-efficient sampling for CrowdBT model.

    Args:
        r: tensor of rewards, shape (N,)
        beta: tensor of worker reliabilities in [0,1], shape (K,)
        n_samples: number of comparisons
        seed: optional seed
    """

    # --- Reproducibility ---
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    N = r.shape[0]
    K = beta.shape[0]

    # 1. Sample item pairs (a, b) and workers (c)
    a = torch.randint(0, N, (n_samples,), device=r.device)
    b = torch.randint(0, N, (n_samples,), device=r.device)

    # Ensure a != b
    mask = (a == b)
    while mask.any():
        b[mask] = torch.randint(0, N, (mask.sum(),), device=r.device)
        mask = (a == b)

    c = torch.randint(0, K, (n_samples,), device=r.device)

    # 2. Get values
    r_a = r[a]
    r_b = r[b]
    beta_c = beta[c]

    # 3. True Bradley-Terry probability
    p_true = torch.sigmoid(r_a - r_b)

    # 4. Decide if worker is correct or flips
    rand_worker = torch.rand(n_samples, device=r.device)
    is_correct = rand_worker < beta_c   # with prob beta_k → correct

    # 5. Sample outcome
    rand_outcome = torch.rand(n_samples, device=r.device)
    a_beats_b = rand_outcome < p_true

    # If worker flips → invert outcome
    final_a_wins = torch.where(is_correct, a_beats_b, ~a_beats_b)

    winners = torch.where(final_a_wins, a, b)
    losers  = torch.where(final_a_wins, b, a)

    return torch.stack([winners, losers, c], dim=-1)

from collections import defaultdict
from pathlib import Path

# def build_synthetic_pc_dict_for_noisybt(
#     df,
#     n_items,
#     worker_col="worker",
#     left_col="left",
#     right_col="right",
#     label_col="label",
# ):
#     unique_workers = df[worker_col].unique()
#     performer_label_dict = {
#         worker: i for i, worker in enumerate(unique_workers)
#     }

#     # Identity mapping
#     item_label_dict = {i: i for i in range(n_items)}

#     pc_dict = defaultdict(list)

#     for worker, group in df.groupby(worker_col):
#         wid = performer_label_dict[worker]

#         for _, row in group.iterrows():
#             left = int(row[left_col])
#             right = int(row[right_col])
#             winner = int(row[label_col])

#             pc_dict[wid].append((left, right, winner))

#     return dict(pc_dict), performer_label_dict, item_label_dict

from collections import defaultdict
import torch

def build_synthetic_pc_dict_for_noisybt(samples, n_items, seed=None):
    """
    samples: torch.Tensor (GPU/CPU) or numpy array of shape (m,3)
             columns = (winner, loser, worker)
    """

    if torch.is_tensor(samples):
        samples = samples.cpu().numpy()

    rng = np.random.default_rng(seed)

    n = samples.shape[0]

    # exactly balanced
    flags = np.zeros(n, dtype=bool)
    flags[n // 2:] = True
    rng.shuffle(flags)

    pc_dict = defaultdict(list)

    for (winner, loser, worker), flip in zip(samples, flags):
        if flip:
            left, right = loser, winner
        else:
            left, right = winner, loser

        pc_dict[int(worker)].append((int(left), int(right), int(winner)))

    performer_label_dict = {i: i for i in range(samples[:, 2].max() + 1)}
    item_label_dict = {i: i for i in range(n_items)}

    return dict(pc_dict), performer_label_dict, item_label_dict

from collections import defaultdict
import numpy as np

def pc_passage_to_noisybt(PC_passage, n_items, seed=None):
    rng = np.random.default_rng(seed)

    pc_dict = defaultdict(list)

    # count total comparisons
    total = sum(len(v) for v in PC_passage.values())

    # exactly balanced left/right
    flags = np.zeros(total, dtype=bool)
    flags[total // 2:] = True
    rng.shuffle(flags)

    idx = 0

    for worker, comps in PC_passage.items():
        for winner, loser in comps:

            if flags[idx]:
                left, right = loser, winner
            else:
                left, right = winner, loser

            pc_dict[worker].append((left, right, winner))
            idx += 1

    performer_label_dict = {w: w for w in PC_passage.keys()}
    item_label_dict = {i: i for i in range(n_items)}

    return dict(pc_dict), performer_label_dict, item_label_dict