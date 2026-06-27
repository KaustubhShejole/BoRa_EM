import numpy as np
import pandas as pd
import os
import torch
import scipy.sparse.linalg
from tqdm import tqdm
import random
import torch

# Helper functions
def estimate_labels(true_r, r_est, comparisons):
    labels = []
    for a,b,_ in comparisons:
        if true_r[a] > true_r[b]:
            labels.append(1 if r_est[a] > r_est[b] else 0)
        else:
            labels.append(1 if r_est[a] < r_est[b] else 0)
    return labels

def get_original_label(true_r, comparisons):
    labels = []
    for a,b,_ in comparisons:
        labels.append(1 if true_r[a] > true_r[b] else 0)
    return labels


class EMWrapper:
    def __init__(self, df_by_worker, max_iter, device, random_seed=45, num_items=None, num_workers=None, verbose=True, init_beta=0.8, tol=1e-6):
#         if device not in ("cuda", "cpu"):
#             raise ValueError(f"Unsupported device: {device}. Use 'cuda' or 'cpu'.")
        self.device = device
        print(self.device)
        self.random_seed = random_seed
        random.seed(self.random_seed)
        self.comparisons = self._get_compatible_data(df_by_worker)
#         print(self.comparisons[:5])
        self.max_iter = max_iter
        self.verbose = verbose
        self.init_beta = init_beta
        self.tol = tol
        

        # Number of workers
        if num_workers == None:
            self.num_workers = len(df_by_worker)
        else:
            self.num_workers = num_workers

        # Number of items = 1 + max index seen
        max_index = max(
            max(winner, loser) for winner, loser, _ in self.comparisons
        )
        if num_items == None:
            self.num_items = max_index + 1
        else:
            self.num_items = num_items

    def _get_compatible_data(self, data):
        comparisons = []
        # Change this: instead of while worker_id in data
        for worker_id, cc in data.items(): 
            for comp in cc:
                winner, loser = comp
                comparisons.append([winner, loser, worker_id])

        if not comparisons:
            raise ValueError("No comparisons found in the input data!")

        random.shuffle(comparisons)
        return np.array(comparisons, dtype=np.int64)
    
    def run_algorithm(self):
        pgem = PolyaGamma_EM(self.num_items, self.num_workers, max_iter=self.max_iter, device=self.device, random_seed=self.random_seed, verbose=self.verbose, init_beta=self.init_beta, epsilon=self.tol)
        r_est, b_est, ll = pgem.fit(self.comparisons)
        return r_est, b_est, ll

def to_tensor(x, dtype=None, device=None):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


class EMWrapper_3_0:
    def __init__(self, df_by_worker, max_iter, device, random_seed=45, num_items=None, num_workers=None):
        self.device = device
        print(self.device)

        self.random_seed = random_seed
        random.seed(self.random_seed)

        self.comparisons = self._get_compatible_data(df_by_worker)
        self.max_iter = max_iter

        # Number of workers
        if num_workers is None:
            self.num_workers = len(df_by_worker)
        else:
            self.num_workers = num_workers

        # Number of items
        max_index = max(
            max(winner, loser) for winner, loser, _ in self.comparisons
        )
        if num_items is None:
            self.num_items = max_index + 1
        else:
            self.num_items = num_items

    def _get_compatible_data(self, data):
        comparisons = []

        for worker_id, cc in data.items():
            for comp in cc:
                winner, loser = comp
                comparisons.append([winner, loser, worker_id])

        if not comparisons:
            raise ValueError("No comparisons found in the input data!")

        random.shuffle(comparisons)
        return np.array(comparisons, dtype=np.int64)

    def run_algorithm(self):
        pgem = PolyaGamma_EM_4_0(
            num_items=self.num_items,
            num_workers=self.num_workers,
            max_iter=self.max_iter,
            device=self.device,
            random_seed=self.random_seed
        )

        r_est, b_est, ll = pgem.fit(self.comparisons)
        return r_est, b_est, ll

    
class EMWrapper_random_r:
    def __init__(self, df_by_worker, max_iter, device, random_seed=45, num_items=None, num_workers=None):
        self.device = device
        print(self.device)

        self.random_seed = random_seed
        random.seed(self.random_seed)

        self.comparisons = self._get_compatible_data(df_by_worker)
        self.max_iter = max_iter

        # Number of workers
        if num_workers is None:
            self.num_workers = len(df_by_worker)
        else:
            self.num_workers = num_workers

        # Number of items
        max_index = max(
            max(winner, loser) for winner, loser, _ in self.comparisons
        )
        if num_items is None:
            self.num_items = max_index + 1
        else:
            self.num_items = num_items

    def _get_compatible_data(self, data):
        comparisons = []

        for worker_id, cc in data.items():
            for comp in cc:
                winner, loser = comp
                comparisons.append([winner, loser, worker_id])

        if not comparisons:
            raise ValueError("No comparisons found in the input data!")

        random.shuffle(comparisons)
        return np.array(comparisons, dtype=np.int64)

    def run_algorithm(self):
        pgem = PolyaGamma_EM_flip_order(
            num_items=self.num_items,
            num_workers=self.num_workers,
            max_iter=self.max_iter,
            device=self.device,
            random_seed=self.random_seed
        )

        r_est, b_est, ll = pgem.fit(self.comparisons)
        return r_est, b_est, ll


import torch
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F

import torch
import numpy as np
from tqdm import tqdm

class PolyaGamma_EM:
    def __init__(self, num_items, num_workers, max_iter=500, epsilon=1e-6, device='cuda', random_seed=45, verbose=True, init_beta=0.8):
        self.device = device
        print(self.device)
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)  # For multi-GPU

        np.random.seed(self.random_seed)

        self.num_items = num_items
        self.num_workers = num_workers
        self.max_iter = max_iter
        self.epsilon = epsilon

        # --- new hyperparams for regularization (minimal additions) ---
        self.sigma_beta = 1.0    # prior std for beta (Gaussian prior). Tune this.
        self.lambda_r_base = 1e-2  # base multiplier for ridge on rewards. Tune this.
        # ----------------------------------------------------------------
        self.beta = torch.full((num_workers,), init_beta, device=device)
#         self.beta = torch.ones(num_workers, device=device)
        self.r = torch.zeros(num_items, device=device)
        self.verbose = verbose

#         # Initialize parameters with identifiability constraints
#         self.r = torch.randn(num_items, device=device)
#         self.r -= self.r.mean()  # Zero-mean initialization

#         if initial_r is None:
#             self.r = torch.zeros(num_items, device=device)
#         else:
#             self.r = torch.as_tensor(initial_r, device=device, dtype=torch.float32).clone().detach()

        # Pre-allocated tensors for comparison processing
        self.winner_idx = None
        self.loser_idx = None
        self.worker_idx = None

    def _log_likelihood(self):
        logits = self.beta[self.worker_idx] * (self.r[self.winner_idx] - self.r[self.loser_idx])
        return torch.log(torch.sigmoid(logits) + 1e-8).mean()

    def _compute_pg_expectations(self):
        deltas = self.r[self.winner_idx] - self.r[self.loser_idx]
        x = self.beta[self.worker_idx] * deltas

        abs_x = x.abs()

        kappas = torch.where(
            abs_x < 1e-8,
            0.25 - (x**2)/48.0,                # Taylor approx for small x
            torch.tanh(x/2) / (2*x)            # exact for other x
        )

        return kappas

    def _update_competencies(self, kappas):
        deltas = self.r[self.winner_idx] - self.r[self.loser_idx]

        # Vectorized contributions
        num_contrib = 0.5 * deltas
        denom_contrib = kappas * deltas**2

        # Aggregate by worker
        numerator = torch.zeros(self.num_workers, device=self.device)
        denominator = torch.zeros(self.num_workers, device=self.device)
        numerator.index_add_(0, self.worker_idx, num_contrib)
        denominator.index_add_(0, self.worker_idx, denom_contrib)

        # --- Bayesian Gaussian prior on beta (ridge / L2) ---
        prior_prec = 1.0 / (self.sigma_beta ** 2)  # precision = 1/sigma^2
        # add prior precision to every worker's denominator (acts like extra observations)
        denominator = denominator + prior_prec
        # ----------------------------------------------------

        # Update competencies with stability checks
        valid = denominator > 1e-12
        self.beta[valid] = numerator[valid] / denominator[valid]
        # workers with tiny denominator will be shrunk toward zero by the prior

    import torch

    def _update_rewards(self, kappas):
        num_items = self.num_items
        device = self.device
        beta = self.beta

        # 1. Faster Sparse Construction
        beta_sq = beta[self.worker_idx] ** 2
        summands = beta_sq * kappas

        rows = torch.cat([self.winner_idx, self.loser_idx, self.winner_idx, self.loser_idx])
        cols = torch.cat([self.winner_idx, self.loser_idx, self.loser_idx, self.winner_idx])
        vals = torch.cat([summands, summands, -summands, -summands])

        # Keep H as a sparse tensor to save memory and avoid dense conversion overhead
        indices = torch.stack([rows, cols], dim=0)
        H_sparse = torch.sparse_coo_tensor(indices, vals, (num_items, num_items), device=device).coalesce()

        # 2. Build RHS vector b (Vectorized)
        b = torch.zeros(num_items, device=device)
        b_i_vals = 0.5 * beta[self.worker_idx]
        b.index_add_(0, self.winner_idx, b_i_vals)
        b.index_add_(0, self.loser_idx, -b_i_vals)

        # --- Data-dependent ridge regularization (statistical, not just numeric) ---
        lambda_r = self.lambda_r_base * max(1e-12, kappas.mean().item())
        # Solve (H + lambda_r * I) r = b via CG
        r = self._torch_cg(H_sparse, b, r0=self.r, max_iter=500, tol=1e-5, reg=lambda_r)
        # -------------------------------------------------------------------------

        # Apply zero-mean constraint
        self.r = r - r.mean()

    def _torch_cg_copy(self, A_sparse, b, r0=None, max_iter=500, tol=1e-5, reg=1e-8):
        """
        Pure PyTorch implementation of Conjugate Gradient.
        Solves (A + reg*I)x = b
        """
        x = r0 if r0 is not None else torch.zeros_like(b)

        # Helper for matrix-vector product (A + reg*I) @ x
        def mvp(v):
            return torch.matmul(A_sparse, v.unsqueeze(1)).squeeze(1) + reg * v

        r = b - mvp(x)
        if torch.norm(r) < tol:
            return x

        p = r.clone()
        rdotr = torch.dot(r, r)

        for i in range(max_iter):
            Ap = mvp(p)
            alpha = rdotr / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap

            new_rdotr = torch.dot(r, r)
            if torch.sqrt(new_rdotr) < tol:
                break

            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def _torch_cg(self, A_sparse, b, r0=None, max_iter=500, tol=1e-5, reg=1e-8):
        # 1. High-precision promotion
        orig_dtype = b.dtype
        b_64 = b.detach().to(torch.float64)
        A_64 = A_sparse.detach().to(torch.float64)
        x = r0.to(torch.float64) if r0 is not None else torch.zeros_like(b_64)

        def mvp(v):
            if A_64.is_sparse:
                return torch.sparse.mm(A_64, v.unsqueeze(1)).squeeze(1) + reg * v
            return torch.matmul(A_64, v.unsqueeze(1)).squeeze(1) + reg * v

        # 2. Construct Diagonal Preconditioner (Jacobi)
        # Extract diagonal: A_ii + reg
        if A_64.is_sparse:
            # For sparse COO, find indices where row == col
            indices = A_64._indices()
            values = A_64._values()
            mask = (indices[0] == indices[1])
            diag = torch.zeros(b_64.size(0), device=b_64.device, dtype=torch.float64)
            diag[indices[0][mask]] = values[mask]
            diag += reg
        else:
            diag = torch.diag(A_64) + reg

        # M_inv scales the residual to be near 1.0 based on the matrix scale
        M_inv = 1.0 / (diag + 1e-12)

        r = b_64 - mvp(x)
        z = M_inv * r  # This is your "scaled" residual
        p = z.clone()
        rdotz = torch.dot(r, z)

        if torch.sqrt(rdotz) < tol:
            return x.to(orig_dtype)

        for i in range(max_iter):
            Ap = mvp(p)

            denom = torch.dot(p, Ap)
            if denom <= 1e-16: # Stability break
                break

            alpha = rdotz / denom
            x = x + alpha * p

            # Periodic refresh for stability
            if i % 50 == 0:
                r = b_64 - mvp(x)
            else:
                r = r - alpha * Ap

            z = M_inv * r  # Apply scaling/preconditioning
            new_rdotz = torch.dot(r, z)

            if torch.norm(r) < tol:
                break

            beta = new_rdotz / (rdotz + 1e-16)
            p = z + beta * p
            rdotz = new_rdotz

            if torch.isnan(x).any():
                break

        return x.to(orig_dtype)

    def _check_convergence(self, prev_r, prev_beta):
        r_diff = torch.norm(self.r - prev_r)
        beta_diff = torch.norm(self.beta - prev_beta)
        return r_diff < self.epsilon and beta_diff < self.epsilon

    def fit(self, comparisons):
        # Convert comparisons to tensor indices
        comparisons_tensor = torch.tensor(comparisons, dtype=torch.long, device=self.device)
        self.winner_idx = comparisons_tensor[:, 0]
        self.loser_idx = comparisons_tensor[:, 1]
        self.worker_idx = comparisons_tensor[:, 2]

        prev_r = self.r.clone()
        prev_beta = self.beta.clone()
        prev_ll = -float("inf")

        for iter in tqdm(range(self.max_iter), disable=not self.verbose):
            # E-step: Compute Polya-Gamma expectations
            kappas = self._compute_pg_expectations()

            # M-step: Update parameters
            for _ in range(10):
                self._update_rewards(kappas)      # r vector
                self._update_competencies(kappas) # beta vector

            # Enforce identifiability constraints: zero-mean then scale-normalize
            self.r -= self.r.mean()
            # scale normalization: set RMS of r to 1
            scale = torch.sqrt(torch.mean(self.r ** 2) + 1e-12)
            if scale > 0:
                self.r = self.r / scale
                self.beta = self.beta * scale

            # Removed hard clamp on beta; prior handles shrinkage

            # Compute log-likelihood
            ll = self._log_likelihood().item()
            if iter % 100 == 0:
                print(f"Iter {iter:03d}: Log-likelihood = {ll:.6f}")

            # Convergence based on log-likelihood change
            if abs(ll - prev_ll) < self.epsilon:
                print(f"Converged at iter {iter}, Log-likelihood change = {ll - prev_ll:.6e}")
                break

            prev_ll = ll
            prev_r = self.r.clone()
            prev_beta = self.beta.clone()

        return self.r.cpu().numpy(), self.beta.cpu().numpy(), self._log_likelihood().item()

    
    
import torch
import numpy as np
from tqdm import tqdm

class PolyaGamma_EM_3_0:
    def __init__(self, num_items, num_workers, max_iter=500, epsilon=1e-6, device='cuda', random_seed=45):
        self.device = device
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)

        self.num_items = num_items
        self.num_workers = num_workers
        self.max_iter = max_iter
        self.epsilon = epsilon

        self.sigma_beta = 1.0
        self.lambda_r_base = 1e-2

        self.beta = torch.full((num_workers,), 0.8, device=device)
        self.r = torch.zeros(num_items, device=device)

        self.winner_idx = None
        self.loser_idx = None
        self.worker_idx = None

    @torch.no_grad()
    def _log_likelihood(self):
        logits = self.beta[self.worker_idx] * (self.r[self.winner_idx] - self.r[self.loser_idx])
        return torch.log(torch.sigmoid(logits) + 1e-8).mean()

    @torch.no_grad()
    def _compute_pg_expectations(self):
        deltas = self.r[self.winner_idx] - self.r[self.loser_idx]
        x = self.beta[self.worker_idx] * deltas
        abs_x = x.abs()

        kappas = torch.where(
            abs_x < 1e-8,
            0.25 - (x ** 2) / 48.0,
            torch.tanh(x / 2) / (2 * x)
        )
        return kappas

    @torch.no_grad()
    def _update_competencies(self, kappas):
        deltas = self.r[self.winner_idx] - self.r[self.loser_idx]

        numerator = torch.zeros(self.num_workers, device=self.device, dtype=self.r.dtype)
        denominator = torch.zeros(self.num_workers, device=self.device, dtype=self.r.dtype)

        numerator.index_add_(0, self.worker_idx, 0.5 * deltas)
        denominator.index_add_(0, self.worker_idx, kappas * deltas ** 2)

        prior_prec = 1.0 / (self.sigma_beta ** 2)
        denominator = denominator + prior_prec

        valid = denominator > 1e-12
        self.beta[valid] = numerator[valid] / denominator[valid]

    @torch.no_grad()
    def _laplacian_mv(self, weights, v, reg=0.0):
        """
        Exact matvec for the weighted Laplacian-like system:
        H = sum_e w_e * [[1,-1],[-1,1]]
        over each comparison edge e=(winner, loser).
        """
        diff = v[self.winner_idx] - v[self.loser_idx]
        msg = weights * diff

        out = torch.zeros_like(v)
        out.index_add_(0, self.winner_idx, msg)
        out.index_add_(0, self.loser_idx, -msg)

        if reg != 0.0:
            out.add_(v, alpha=reg)
        return out

    @torch.no_grad()
    def _torch_cg(self, weights, b, r0=None, max_iter=500, tol=1e-5, reg=1e-8):
        """
        Preconditioned CG for (H + reg I)x = b
        where H is represented implicitly by the comparisons.
        """
        orig_dtype = b.dtype
        b = b.to(torch.float64)
        weights = weights.to(torch.float64)

        x = r0.to(torch.float64) if r0 is not None else torch.zeros_like(b)

        # Jacobi preconditioner from exact diagonal
        diag = torch.zeros(self.num_items, device=b.device, dtype=torch.float64)
        diag.index_add_(0, self.winner_idx, weights)
        diag.index_add_(0, self.loser_idx, weights)
        diag = diag + reg
        M_inv = 1.0 / (diag + 1e-12)

        def mvp(v):
            return self._laplacian_mv(weights, v, reg=reg)

        r = b - mvp(x)
        z = M_inv * r
        p = z.clone()
        rdotz = torch.dot(r, z)

        if torch.sqrt(rdotz) < tol:
            return x.to(orig_dtype)

        for _ in range(max_iter):
            Ap = mvp(p)
            denom = torch.dot(p, Ap)
            if denom <= 1e-16:
                break

            alpha = rdotz / denom
            x = x + alpha * p
            r = r - alpha * Ap

            if torch.norm(r) < tol:
                break

            z = M_inv * r
            new_rdotz = torch.dot(r, z)

            beta = new_rdotz / (rdotz + 1e-16)
            p = z + beta * p
            rdotz = new_rdotz

            if torch.isnan(x).any():
                break

        return x.to(orig_dtype)

    @torch.no_grad()
    def _update_rewards(self, kappas):
        beta_w = self.beta[self.worker_idx]
        weights = (beta_w ** 2) * kappas

        b = torch.zeros(self.num_items, device=self.device, dtype=self.r.dtype)
        contrib = 0.5 * beta_w
        b.index_add_(0, self.winner_idx, contrib)
        b.index_add_(0, self.loser_idx, -contrib)

        lambda_r = self.lambda_r_base * max(1e-12, kappas.mean().item())
        r = self._torch_cg(weights, b, r0=self.r, max_iter=500, tol=1e-5, reg=lambda_r)
        self.r = r - r.mean()

    @torch.no_grad()
    def fit(self, comparisons):
        comparisons_tensor = torch.tensor(comparisons, dtype=torch.long, device=self.device)
        self.winner_idx = comparisons_tensor[:, 0]
        self.loser_idx = comparisons_tensor[:, 1]
        self.worker_idx = comparisons_tensor[:, 2]

        prev_ll = -float("inf")

        with torch.inference_mode():
            for iter in tqdm(range(self.max_iter)):
                kappas = self._compute_pg_expectations()

                # Keep your current update schedule unchanged
                for _ in range(10):
                    self._update_rewards(kappas)
                    self._update_competencies(kappas)

                self.r -= self.r.mean()
                scale = torch.sqrt(torch.mean(self.r ** 2) + 1e-12)
                if scale > 0:
                    self.r = self.r / scale
                    self.beta = self.beta * scale

                ll = self._log_likelihood().item()
                if iter % 100 == 0:
                    print(f"Iter {iter:03d}: Log-likelihood = {ll:.6f}")

                if abs(ll - prev_ll) < self.epsilon:
                    print(f"Converged at iter {iter}, Log-likelihood change = {ll - prev_ll:.6e}")
                    break

                prev_ll = ll

        return self.r.cpu().numpy(), self.beta.cpu().numpy(), self._log_likelihood().item()


import torch
import numpy as np
from tqdm import tqdm

class PolyaGamma_EM_4_0:
    def __init__(self, num_items, num_workers, max_iter=500, epsilon=1e-6, device='cuda', random_seed=45):
        self.device = device
        self.random_seed = random_seed

        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)

        self.num_items = num_items
        self.num_workers = num_workers
        self.max_iter = max_iter
        self.epsilon = epsilon

        self.sigma_beta = 1.0
        self.lambda_r_base = 1e-2

        self.beta = torch.full((num_workers,), 0.8, device=device)
        self.r = torch.zeros(num_items, device=device)

        self.winner_idx = None
        self.loser_idx = None
        self.worker_idx = None

    @torch.no_grad()
    def _log_likelihood(self):
        r_w = self.r[self.winner_idx]
        r_l = self.r[self.loser_idx]
        beta_w = self.beta[self.worker_idx]

        logits = beta_w * (r_w - r_l)
        return torch.log(torch.sigmoid(logits) + 1e-8).mean()

    @torch.no_grad()
    def _compute_pg_expectations(self):
        r_w = self.r[self.winner_idx]
        r_l = self.r[self.loser_idx]
        beta_w = self.beta[self.worker_idx]

        deltas = r_w - r_l
        x = beta_w * deltas
        abs_x = x.abs()

        kappas = torch.where(
            abs_x < 1e-8,
            0.25 - (x ** 2) / 48.0,
            torch.tanh(x / 2) / (2 * x)
        )
        return kappas

    @torch.no_grad()
    def _update_competencies(self, kappas):
        r_w = self.r[self.winner_idx]
        r_l = self.r[self.loser_idx]
        deltas = r_w - r_l

        numerator = torch.zeros(self.num_workers, device=self.device, dtype=self.r.dtype)
        denominator = torch.zeros(self.num_workers, device=self.device, dtype=self.r.dtype)

        numerator.index_add_(0, self.worker_idx, 0.5 * deltas)
        denominator.index_add_(0, self.worker_idx, kappas * deltas ** 2)

        prior_prec = 1.0 / (self.sigma_beta ** 2)
        denominator = denominator + prior_prec

        valid = denominator > 1e-12
        self.beta[valid] = numerator[valid] / denominator[valid]

    @torch.no_grad()
    def _laplacian_mv(self, weights, v, reg=0.0):
        diff = v[self.winner_idx] - v[self.loser_idx]
        msg = weights * diff

        out = torch.zeros_like(v)
        out.index_add_(0, self.winner_idx, msg)
        out.index_add_(0, self.loser_idx, -msg)

        if reg != 0.0:
            out.add_(v, alpha=reg)
        return out

    @torch.no_grad()
    def _torch_cg(self, weights, b, r0=None, max_iter=500, tol=1e-5, reg=1e-8):
        orig_dtype = b.dtype
        b = b.to(torch.float64)
        weights = weights.to(torch.float64)

        x = r0.to(torch.float64) if r0 is not None else torch.zeros_like(b)

        # Jacobi preconditioner
        diag = torch.zeros(self.num_items, device=b.device, dtype=torch.float64)
        diag.index_add_(0, self.winner_idx, weights)
        diag.index_add_(0, self.loser_idx, weights)
        diag = diag + reg
        M_inv = 1.0 / (diag + 1e-12)

        def mvp(v):
            return self._laplacian_mv(weights, v, reg=reg)

        r = b - mvp(x)
        z = M_inv * r
        p = z.clone()
        rdotz = torch.dot(r, z)

        if torch.sqrt(rdotz) < tol:
            return x.to(orig_dtype)

        for _ in range(max_iter):
            Ap = mvp(p)
            denom = torch.dot(p, Ap)
            if denom <= 1e-16:
                break

            alpha = rdotz / denom
            x = x + alpha * p
            r = r - alpha * Ap

            if torch.norm(r) < tol:
                break

            z = M_inv * r
            new_rdotz = torch.dot(r, z)

            beta = new_rdotz / (rdotz + 1e-16)
            p = z + beta * p
            rdotz = new_rdotz

            if torch.isnan(x).any():
                break

        return x.to(orig_dtype)

    @torch.no_grad()
    def _update_rewards(self, kappas):
        beta_w = self.beta[self.worker_idx]
        weights = (beta_w ** 2) * kappas

        b = torch.zeros(self.num_items, device=self.device, dtype=self.r.dtype)
        contrib = 0.5 * beta_w
        b.index_add_(0, self.winner_idx, contrib)
        b.index_add_(0, self.loser_idx, -contrib)

        lambda_r = self.lambda_r_base * torch.clamp(kappas.mean(), min=1e-12)

        r = self._torch_cg(weights, b, r0=self.r, max_iter=500, tol=1e-5, reg=lambda_r)
        self.r = r - r.mean()

    @torch.no_grad()
    def fit(self, comparisons):
        comparisons_tensor = torch.tensor(comparisons, dtype=torch.int32, device=self.device)

        self.winner_idx = comparisons_tensor[:, 0]
        self.loser_idx  = comparisons_tensor[:, 1]
        self.worker_idx = comparisons_tensor[:, 2]

        prev_ll = -float("inf")

        with torch.inference_mode():
            for iter in tqdm(range(self.max_iter)):
                kappas = self._compute_pg_expectations()

                for _ in range(1):
                    self._update_rewards(kappas)
                    self._update_competencies(kappas)

                self.r -= self.r.mean()
                scale = torch.sqrt(torch.mean(self.r ** 2) + 1e-12)
                if scale > 0:
                    self.r = self.r / scale
                    self.beta = self.beta * scale

                ll = self._log_likelihood().item()
                if iter % 100 == 0:
                    print(f"Iter {iter:03d}: Log-likelihood = {ll:.6f}")

                if abs(ll - prev_ll) < self.epsilon:
                    print(f"Converged at iter {iter}, ΔLL = {ll - prev_ll:.6e}")
                    break

                prev_ll = ll

        return self.r.cpu().numpy(), self.beta.cpu().numpy(), self._log_likelihood().item()
    


class PolyaGamma_EM_flip_order:
    def __init__(self, num_items, num_workers, max_iter=500, epsilon=1e-6, device='cuda', random_seed=45):
        self.device = device
        self.random_seed = random_seed

        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)

        self.num_items = num_items
        self.num_workers = num_workers
        self.max_iter = max_iter
        self.epsilon = epsilon

        self.sigma_beta = 1.0
        self.lambda_r_base = 1e-2

        self.beta = torch.full((num_workers,), 0.8, device=device)
#         self.r = torch.zeros(num_items, device=device)
        self.r = (torch.rand(num_items, device=device) * 4) - 2
        self.r = self.r - self.r.mean()

        self.winner_idx = None
        self.loser_idx = None
        self.worker_idx = None

    @torch.no_grad()
    def _log_likelihood(self):
        r_w = self.r[self.winner_idx]
        r_l = self.r[self.loser_idx]
        beta_w = self.beta[self.worker_idx]

        logits = beta_w * (r_w - r_l)
        return torch.log(torch.sigmoid(logits) + 1e-8).mean()

    @torch.no_grad()
    def _compute_pg_expectations(self):
        r_w = self.r[self.winner_idx]
        r_l = self.r[self.loser_idx]
        beta_w = self.beta[self.worker_idx]

        deltas = r_w - r_l
        x = beta_w * deltas
        abs_x = x.abs()

        kappas = torch.where(
            abs_x < 1e-8,
            0.25 - (x ** 2) / 48.0,
            torch.tanh(x / 2) / (2 * x)
        )
        return kappas

    @torch.no_grad()
    def _update_competencies(self, kappas):
        r_w = self.r[self.winner_idx]
        r_l = self.r[self.loser_idx]
        deltas = r_w - r_l

        numerator = torch.zeros(self.num_workers, device=self.device, dtype=self.r.dtype)
        denominator = torch.zeros(self.num_workers, device=self.device, dtype=self.r.dtype)

        numerator.index_add_(0, self.worker_idx, 0.5 * deltas)
        denominator.index_add_(0, self.worker_idx, kappas * deltas ** 2)

        prior_prec = 1.0 / (self.sigma_beta ** 2)
        denominator = denominator + prior_prec

        valid = denominator > 1e-12
        self.beta[valid] = numerator[valid] / denominator[valid]

    @torch.no_grad()
    def _laplacian_mv(self, weights, v, reg=0.0):
        diff = v[self.winner_idx] - v[self.loser_idx]
        msg = weights * diff

        out = torch.zeros_like(v)
        out.index_add_(0, self.winner_idx, msg)
        out.index_add_(0, self.loser_idx, -msg)

        if reg != 0.0:
            out.add_(v, alpha=reg)
        return out

    @torch.no_grad()
    def _torch_cg(self, weights, b, r0=None, max_iter=500, tol=1e-5, reg=1e-8):
        orig_dtype = b.dtype
        b = b.to(torch.float64)
        weights = weights.to(torch.float64)

        x = r0.to(torch.float64) if r0 is not None else torch.zeros_like(b)

        # Jacobi preconditioner
        diag = torch.zeros(self.num_items, device=b.device, dtype=torch.float64)
        diag.index_add_(0, self.winner_idx, weights)
        diag.index_add_(0, self.loser_idx, weights)
        diag = diag + reg
        M_inv = 1.0 / (diag + 1e-12)

        def mvp(v):
            return self._laplacian_mv(weights, v, reg=reg)

        r = b - mvp(x)
        z = M_inv * r
        p = z.clone()
        rdotz = torch.dot(r, z)

        if torch.sqrt(rdotz) < tol:
            return x.to(orig_dtype)

        for _ in range(max_iter):
            Ap = mvp(p)
            denom = torch.dot(p, Ap)
            if denom <= 1e-16:
                break

            alpha = rdotz / denom
            x = x + alpha * p
            r = r - alpha * Ap

            if torch.norm(r) < tol:
                break

            z = M_inv * r
            new_rdotz = torch.dot(r, z)

            beta = new_rdotz / (rdotz + 1e-16)
            p = z + beta * p
            rdotz = new_rdotz

            if torch.isnan(x).any():
                break

        return x.to(orig_dtype)

    @torch.no_grad()
    def _update_rewards(self, kappas):
        beta_w = self.beta[self.worker_idx]
        weights = (beta_w ** 2) * kappas

        b = torch.zeros(self.num_items, device=self.device, dtype=self.r.dtype)
        contrib = 0.5 * beta_w
        b.index_add_(0, self.winner_idx, contrib)
        b.index_add_(0, self.loser_idx, -contrib)

        # 🚀 removed .item() (no CPU sync)
        lambda_r = self.lambda_r_base * torch.clamp(kappas.mean(), min=1e-12)

        r = self._torch_cg(weights, b, r0=self.r, max_iter=500, tol=1e-5, reg=lambda_r)
        self.r = r - r.mean()

    @torch.no_grad()
    def fit(self, comparisons):
        comparisons_tensor = torch.tensor(comparisons, dtype=torch.int32, device=self.device)

        # 🚀 int32 indices
        self.winner_idx = comparisons_tensor[:, 0]
        self.loser_idx  = comparisons_tensor[:, 1]
        self.worker_idx = comparisons_tensor[:, 2]

        prev_ll = -float("inf")

        with torch.inference_mode():
            for iter in tqdm(range(self.max_iter)):
                kappas = self._compute_pg_expectations()

                for _ in range(10):
                    self._update_rewards(kappas)
                    self._update_competencies(kappas)

                self.r -= self.r.mean()
                scale = torch.sqrt(torch.mean(self.r ** 2) + 1e-12)
                if scale > 0:
                    self.r = self.r / scale
                    self.beta = self.beta * scale

                ll = self._log_likelihood().item()
                if iter % 100 == 0:
                    print(f"Iter {iter:03d}: Log-likelihood = {ll:.6f}")

                if abs(ll - prev_ll) < self.epsilon:
                    print(f"Converged at iter {iter}, ΔLL = {ll - prev_ll:.6e}")
                    break

                prev_ll = ll

        return self.r.cpu().numpy(), self.beta.cpu().numpy(), self._log_likelihood().item()