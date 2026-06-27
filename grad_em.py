## This is code for HBTL Method

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np

class GradientEMWrapper:
    def __init__(self, df_by_worker, lr=0.01, random_seed=45, device=None, max_iter=200, init_beta=0.8, tol=1e-6):
        self.lr = lr
        self.random_seed = random_seed
        self.tol=tol
        
        # Set device: user provided -> cuda (if available) -> cpu
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        random.seed(self.random_seed)
        self.comparisons = self._get_compatible_data(df_by_worker)
        

        self.num_workers = len(df_by_worker)
        max_index = max(
            max(winner, loser) for winner, loser, _ in self.comparisons
        )
        self.num_items = max_index + 1
        self.max_iter = max_iter
        self.init_beta = init_beta

    def _get_compatible_data(self, data):
        comparisons = []
        worker_id = 0
        for worker_id, cc in data.items():
            for comp in cc:
                winner, loser = comp
                comparisons.append([winner, loser, worker_id])
            worker_id += 1

#         random.shuffle(comparisons)
        return np.array(comparisons, dtype=np.int64)
    
    def run_algorithm(self):
        raw_data = self.comparisons
        
        # 1. Convert to tensors and move to self.device
        winners = torch.tensor([d[0] for d in raw_data], device=self.device)
        losers = torch.tensor([d[1] for d in raw_data], device=self.device)
        annotators = torch.tensor([d[2] for d in raw_data], device=self.device)
        data_tensors = (winners, losers, annotators)

        # 2. Initialize model and move to self.device
        model = GradientEM(self.num_items, self.num_workers, random_seed=self.random_seed, init_beta=self.init_beta)
        model.to(self.device)

        opt_r = torch.optim.Adam(model.item_rewards.parameters(), lr=self.lr)
        opt_beta = torch.optim.Adam(model.worker_betas.parameters(), lr=self.lr)

        r, beta = train_with_convergence_hbtl(model, data_tensors, opt_r, opt_beta, max_epochs=self.max_iter, tol=self.tol)
        
        # Return as CPU tensors or numpy for easier downstream processing
        return r.cpu(), beta.cpu()

class GradientEM(nn.Module):
    def __init__(self, num_items, num_workers, random_seed=42, init_beta=0.8):
        super().__init__()
        
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        
        self.item_rewards = nn.Embedding(num_items, 1)
        with torch.no_grad():
#             torch.nn.init.normal_(self.item_rewards.weight, mean=0.0, std=1.0)
            self.item_rewards.weight.fill_(0.0)
            self.item_rewards.weight -= self.item_rewards.weight.mean()
        
        self.worker_betas = nn.Embedding(num_workers, 1)
        with torch.no_grad():
#             torch.nn.init.uniform_(self.worker_betas.weight, a=-1.0, b=1.0)
            self.worker_betas.weight.fill_(init_beta)

    def forward(self, winners, losers, annotators):
        r_w = self.item_rewards(winners)
        r_l = self.item_rewards(losers)
        beta_s = self.worker_betas(annotators)
        
        logits = beta_s * (r_w - r_l)
        return logits.squeeze()

def train_step_hbtl(model, data, optimizer_r, optimizer_beta):
    winners, losers, annotators = data
#     print("model:", model.item_rewards.weight.device)
#     print("winners:", winners.device)
#     print("losers:", losers.device)
#     print("annotators:", annotators.device)
    
    # 1. Update Rewards (r)
    optimizer_r.zero_grad()
    logits = model(winners, losers, annotators)
    labels = torch.ones_like(logits)
    loss_r = F.binary_cross_entropy_with_logits(logits, labels)
    loss_r.backward()
    optimizer_r.step()
    
    with torch.no_grad():
        mean_r = model.item_rewards.weight.mean()
        model.item_rewards.weight.sub_(mean_r)
    
    # 2. Update Competencies (beta)
    optimizer_beta.zero_grad()
    logits = model(winners, losers, annotators)
    loss_beta = F.binary_cross_entropy_with_logits(logits, labels)
    loss_beta.backward()
    optimizer_beta.step()
    
    with torch.no_grad():
        model.worker_betas.weight.clamp_(-1.0, 1.0)
        
    return loss_r.item()

def train_with_convergence_hbtl(model, data, opt_r, opt_beta, tol=1e-6, max_epochs=200, verbose=True):
#     print("model:", model.item_rewards.weight.device)
    prev_loss = float('inf')
    
    for epoch in tqdm(range(max_epochs), disable=not verbose):
        current_loss = train_step_hbtl(model, data, opt_r, opt_beta)
        loss_diff = abs(prev_loss - current_loss)
        
        if loss_diff < tol:
            print(f"HBTL: Converged at epoch {epoch} | Loss: {current_loss:.6f}")
            break
            
        prev_loss = current_loss
    else:
        print("\nReached max_epochs without full convergence.")
        
    return model.item_rewards.weight.data.flatten(), model.worker_betas.weight.data.flatten()
