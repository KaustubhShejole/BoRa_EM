## Code for HTCV method

import torch
import torch.nn as nn
from tqdm import tqdm
import random
import numpy as np
import math

# -----------------------------
# Utility: Normal CDF (Phi)
# -----------------------------
def normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

def log_normal_cdf(x):
    # numerically stable log Phi
    return torch.log(normal_cdf(x) + 1e-12)


# -----------------------------
# Wrapper
# -----------------------------
class HTCVWrapper:
    def __init__(self, df_by_worker, lr=0.01, random_seed=45, device=None, max_iter=200, init_beta=1.0, tol=1e-6):
        self.lr = lr
        self.random_seed = random_seed
        self.init_beta = init_beta
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        self.comparisons = self._get_compatible_data(df_by_worker)

        self.num_workers = len(df_by_worker)
        max_index = max(max(w, l) for w, l, _ in self.comparisons)
        self.num_items = max_index + 1
        self.max_iter = max_iter
        self.tol = tol

    def _get_compatible_data(self, data):
        comparisons = []
        worker_id = 0
        for worker_id, cc in data.items():
            for winner, loser in cc:
                comparisons.append([winner, loser, worker_id])
            worker_id += 1
        return np.array(comparisons, dtype=np.int64)

    def run_algorithm(self):
        raw_data = self.comparisons

        winners = torch.tensor(raw_data[:, 0], device=self.device)
        losers = torch.tensor(raw_data[:, 1], device=self.device)
        annotators = torch.tensor(raw_data[:, 2], device=self.device)

        data_tensors = (winners, losers, annotators)

        model = HTCV(self.num_items, self.num_workers, self.random_seed, self.init_beta)
        model.to(self.device)

        opt_r = torch.optim.Adam([model.item_rewards.weight], lr=self.lr)
        opt_beta = torch.optim.Adam([model.worker_betas.weight], lr=self.lr)

        r, beta = train_with_convergence_htcv(
            model, data_tensors, opt_r, opt_beta, max_epochs=self.max_iter, tol=self.tol,
        )

        return r.cpu(), beta.cpu()


# -----------------------------
# Model
# -----------------------------
class HTCV(nn.Module):
    def __init__(self, num_items, num_workers, random_seed=42, init_beta=1.0):
        super().__init__()

        torch.manual_seed(random_seed)

        self.item_rewards = nn.Embedding(num_items, 1)
        self.worker_betas = nn.Embedding(num_workers, 1)

        with torch.no_grad():
            # Initialize item scores
            self.item_rewards.weight.fill_(0.0)

            # Initialize worker reliabilities (positive)
            self.worker_betas.weight.fill_(init_beta)

    def forward(self, winners, losers, annotators):
        r_w = self.item_rewards(winners)
        r_l = self.item_rewards(losers)
        beta = self.worker_betas(annotators)

        # HTCV scaling
        logits = beta * (r_w - r_l) / math.sqrt(2)

        return logits.squeeze()


# -----------------------------
# Loss (Probit / HTCV)
# -----------------------------
def probit_loss(logits):
    return -log_normal_cdf(logits).mean()


# -----------------------------
# Training Step
# -----------------------------
def train_step_htcv(model, data, optimizer_r, optimizer_beta):
    winners, losers, annotators = data

    # ---- Update item scores ----
    optimizer_r.zero_grad()
    logits = model(winners, losers, annotators)
    loss_r = probit_loss(logits)
    loss_r.backward()
    optimizer_r.step()

    # Center scores (identifiability)
    with torch.no_grad():
        model.item_rewards.weight -= model.item_rewards.weight.mean()

    # ---- Update worker reliabilities ----
    optimizer_beta.zero_grad()
    logits = model(winners, losers, annotators)
    loss_beta = probit_loss(logits)
    loss_beta.backward()
    optimizer_beta.step()

    # Enforce beta >= 0 (important for HTCV)
    with torch.no_grad():
        model.worker_betas.weight.clamp_(0.0, 5.0)

    return loss_r.item()


# -----------------------------
# Training Loop
# -----------------------------
def train_with_convergence_htcv(model, data, opt_r, opt_beta, tol=1e-6, max_epochs=200, verbose=True):
    prev_loss = float("inf")

    for epoch in tqdm(range(max_epochs), disable=not verbose):
        current_loss = train_step_htcv(model, data, opt_r, opt_beta)

        if abs(prev_loss - current_loss) < tol:
            print(f"\n HTCV: Converged at epoch {epoch} | Loss: {current_loss:.6f}")
            break

        prev_loss = current_loss
    else:
        print("\nReached max_epochs without full convergence.")

    r = model.item_rewards.weight.data.flatten()
    beta = model.worker_betas.weight.data.flatten()

    return r, beta