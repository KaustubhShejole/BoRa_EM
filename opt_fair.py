'''
Implementations following https://github.com/Ambress92/Bias-Aware-Ranker-from-Pairwise-comparisons:
1. Bias-aware ranking from pairwise comparisons (BARP) -- class BARP
2. CrowdBT -- class CrowdBT_3_0
3. Rank Centrality (RC) -- class RC

We made them faster and efficient for utilizing GPU.

4. NoisyBT or FactorBT is adapted from https://github.com/Toloka/crowd-kit/blob/main/crowdkit/aggregation/pairwise/noisy_bt.py -- class NoisyBT_3_0

5. Bradley-Terry-Luce (BTL) is adapted from https://github.com/lucasmaystre/choix/blob/master/choix/opt.py -- class PairwiseFctsGPU.

'''


import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy import linalg
import torch
import torch.nn.functional as F
from tqdm import tqdm

def _safe_exp(x):
    """Safe exponential that preserves gradients"""
    return torch.exp(torch.clamp(x, max=50))

import torch
import random
import numpy as np


import numpy as np
import torch
import torch.nn.functional as F

## Code modified to make it faster and efficient on GPU.
class BARP:

    def __init__(
        self,
        data,
        penalty,
        classes,
        device="cuda",
        dtype=torch.float32,
    ):
        self.device = device
        self.dtype = dtype
        self._penalty = penalty

        self._classes = torch.as_tensor(
            classes,
            device=device,
            dtype=dtype,
        )

        wins = []
        losses = []
        reviewers = []

        for reviewer_id, pairs in data.items():

            if not pairs:
                continue

            # Fast conversion from set[(w,l)] -> ndarray
            arr = np.array(list(pairs), dtype=np.int64)

            wins.append(arr[:, 0])
            losses.append(arr[:, 1])

            reviewers.append(
                np.full(
                    arr.shape[0],
                    reviewer_id,
                    dtype=np.int64,
                )
            )

        if len(wins) == 0:
            self.win_idx = torch.empty(
                0,
                dtype=torch.long,
                device=device,
            )
            self.los_idx = torch.empty(
                0,
                dtype=torch.long,
                device=device,
            )
            self.rev_idx = torch.empty(
                0,
                dtype=torch.long,
                device=device,
            )
        else:
            self.win_idx = torch.from_numpy(
                np.concatenate(wins)
            ).to(device)

            self.los_idx = torch.from_numpy(
                np.concatenate(losses)
            ).to(device)

            self.rev_idx = torch.from_numpy(
                np.concatenate(reviewers)
            ).to(device)

        # Cached class difference
        self.class_diff = (
            self._classes[self.win_idx]
            - self._classes[self.los_idx]
        )

    def objective(self, params, rev_params):

        delta = (
            params[self.win_idx]
            - params[self.los_idx]
            + rev_params[self.rev_idx] * self.class_diff
        )

        return (
            self._penalty * torch.dot(params, params)
            + F.softplus(-delta).sum()
        )

    def gradient_scores(self, params, rev_params):

        delta = (
            params[self.win_idx]
            - params[self.los_idx]
            + rev_params[self.rev_idx] * self.class_diff
        )

        z = torch.sigmoid(-delta)

        grad = 2.0 * self._penalty * params.clone()

        grad.index_add_(0, self.win_idx, -z)
        grad.index_add_(0, self.los_idx, z)

        return grad

    def gradient_revs(self, params, rev_params):

        delta = (
            params[self.win_idx]
            - params[self.los_idx]
            + rev_params[self.rev_idx] * self.class_diff
        )

        z = -torch.sigmoid(-delta) * self.class_diff

        grad = torch.zeros_like(rev_params)

        grad.index_add_(0, self.rev_idx, z)

        return grad


import torch


class RankCentrality:
    def __init__(self, device):
        self.device = device

    def matrix_of_comparisons(self, size, comparisons, reg=1.0, dtype=torch.float32):
        """
        PyTorch version of _matrix_of_comparisons.

        B_ij = fraction of times object j was preferred to object i
               plus reg*(1 - I)
        """

        A = torch.zeros((size, size), dtype=dtype, device=self.device)

        # accumulate counts: preserve original semantics A[j,i] += 1
        if isinstance(comparisons, torch.Tensor):
            if comparisons.numel() > 0:
                idx_i = comparisons[:, 0].long().to(self.device)
                idx_j = comparisons[:, 1].long().to(self.device)

                flat_idx = idx_j * size + idx_i
                A.view(-1).index_add_(
                    0,
                    flat_idx,
                    torch.ones_like(flat_idx, dtype=dtype),
                )

        else:
            try:
                comp_t = torch.as_tensor(
                    comparisons,
                    dtype=torch.long,
                    device=self.device,
                )

                if comp_t.numel() > 0:
                    idx_i = comp_t[:, 0]
                    idx_j = comp_t[:, 1]

                    flat_idx = idx_j * size + idx_i
                    A.view(-1).index_add_(
                        0,
                        flat_idx,
                        torch.ones_like(flat_idx, dtype=dtype),
                    )

            except Exception:
                # exact original fallback
                for i, j in comparisons:
                    A[j, i] += 1.0

        # compute pairwise ratios
        denom = A + A.t()

        B = torch.where(
            denom > 0,
            A / denom,
            torch.zeros_like(A),
        )

        # add reg*(1-I) without allocating eye/ones matrices
        B += reg
        diag_idx = torch.arange(size, device=self.device)
        B[diag_idx, diag_idx] -= reg

        return B

    def trans_prob(self, A):
        """
        Produce row-stochastic transition matrix.
        """

        dtype = A.dtype
        n = A.shape[0]

        counts = torch.count_nonzero(A, dim=1).to(dtype)
        d_max = counts.amax()

        if d_max == 0:
            return torch.eye(
                n,
                dtype=dtype,
                device=self.device,
            )

        P = A / d_max

        sum_by_row = P.sum(dim=1)

        diag_values = 1.0 - sum_by_row

        # no clone needed
        P.view(-1)[:: n + 1] = diag_values

        # numerical guard
        row_sums = P.sum(dim=1, keepdim=True)
        P = P / row_sums

        return P

    def stationary_dist(
        self,
        P,
        tol=1e-6,
        max_iter=10000,
        dtype=None,
    ):
        """
        Compute stationary distribution using power iteration.
        """

        if dtype is None:
            dtype = P.dtype

        n = P.shape[0]

        pi = torch.full(
            (1, n),
            1.0 / n,
            dtype=dtype,
            device=self.device,
        )

        for i in range(max_iter):
            pi_next = pi @ P

            diff = torch.sum(torch.abs(pi_next - pi))

            if diff < tol:
                print(f'RC converged at iteration {i}')
                pi = pi_next
                break

            pi = pi_next

        pi = pi.view(-1)

        pi = torch.clamp(pi, min=0.0)

        s = pi.sum()

        if s <= 0:
            pi = torch.full(
                (n,),
                1.0 / n,
                dtype=dtype,
                device=self.device,
            )
        else:
            pi = pi / s
        
        print('RC finished')

        return pi



################# OLD #################
def _sample_pairs(scores, n_pairs ):
    pairs = []
    numbers = np.arange(len(scores))
    for i in range(n_pairs):
        
        a, b = np.random.choice(numbers, size=2, replace=False) #replace = False to ensure a != b
        #while (a, b) in pairs or (b, a) in pairs: #sample the pair, in this version the reviewer can evaluate the same pair only once
            #a, b = np.random.choice(numbers, size=2, replace=False)

        #make them play
        if np.random.rand() < (np.exp(scores[a])/(np.exp(scores[a])+np.exp(scores[b]))):
        #if scores[a]>scores[b]: #deterministic version
        # this block of code will be executed with probability p
            pairs.append((a, b)) #who win is the first of the pair!!! i.e. a won
        else:
        # this block of code will be executed with probability 1-p   
            pairs.append((b, a)) #b won
    return(pairs)

def _create_matrix_biased_scores(original,rev_bias,classes):
    '''this matrix represents how much bias each reviewer has
    original: the original scores for the items
    rev_bias: the vector with the reviewers' biases
    classes: the items' classes 
    return:
    biases_scores: the matrix with the scores as 'seen' by each reviewer'''
    #matrix of biased scores, each reviewer correspond to a column
    biased_scores = np.zeros((len(original),len(rev_bias)))
    for col,bias in enumerate(rev_bias):
        for row,value in enumerate(classes):
            if value == 1:
                biased_scores[row,col] = original[row] + bias #add bias to reviewers ranking
                                                                                    
            elif value == 0:
                biased_scores[row,col] = original[row] 
    #biased_scores[biased_scores <= 0] = 0.00001
    return biased_scores

def _create_pc_set_for_reviewers(biased_scores,pair_per_reviewer):
    revs_set = {}
    for i in range(np.shape(biased_scores)[1]):
        revs_set.update({i:_sample_pairs(biased_scores[:,i], n_pairs = pair_per_reviewer )})
        
    return revs_set

def create_pc_set_for_reviewers_custom(biased_scores,pair_per_reviewer):
    revs_set = {}
    for i in range(np.shape(biased_scores)[1]):
        revs_set.update({i:_sample_pairs(biased_scores[:,i], n_pairs = pair_per_reviewer[i] )})
        
    return revs_set

def _pc_without_reviewers(revs_set):
    ''' input: the set of pc for each reviewer
        output: pc without the reviewer info'''
    return [[val1, val2] for sublist in revs_set.values() for val1, val2 in sublist]


def _alternate_optim(size, num_reviewers, pc_with_revs, iters = 101, tol = 1e-6, gtol = 1e-5):
    '''x0 is the estimated scores
       y0 is the estimated bias for each reviewer'''
    x0 = np.zeros(size)
    y0 = np.zeros(num_reviewers)
    for i in range(iters):
        # minimize with x fixed and update y
        res_y = minimize(lambda y: pc_with_revs.objective(x0, y), y0,tol = tol, jac=lambda y: pc_with_revs.gradient_revs(x0, y), options={"gtol": gtol,'maxiter': 1})
        y0 = res_y.x

        # minimize with y fixed and update x
        res_x = minimize(lambda x: pc_with_revs.objective(x, y0), x0,tol = tol, jac=lambda x: pc_with_revs.gradient_scores(x, y0), options={"gtol": gtol,'maxiter': 1})
        x0 = res_x.x

        if res_x.success and res_y.success:
            break
    return x0,y0

def _alternate_optim_torch(
    size,
    num_reviewers,
    pc_with_revs,
    iters=100,
    lr_x=1e-2,
    lr_y=1e-2,
    device="cuda",
    tol=1e-6,
    patience=5,
    verbose=True
):
    """
    Alternate optimization using PyTorch.

    Stops when relative loss improvement is below `tol`
    for `patience` consecutive iterations.
    """

    x0 = torch.zeros(size, device=device, requires_grad=True)
    y0 = torch.zeros(num_reviewers, device=device, requires_grad=True)

    optimizer_x = torch.optim.Adam([x0], lr=lr_x)
    optimizer_y = torch.optim.Adam([y0], lr=lr_y)

    prev_loss = float("inf")
    stall_count = 0

    for i in tqdm(range(iters), disable=not verbose):

        # ----- optimize y -----
        optimizer_y.zero_grad(set_to_none=True)
        loss_y = pc_with_revs.objective(x0.detach(), y0)
        loss_y.backward()
        optimizer_y.step()

        # ----- optimize x -----
        optimizer_x.zero_grad(set_to_none=True)
        loss_x = pc_with_revs.objective(x0, y0.detach())
        loss_x.backward()
        optimizer_x.step()

        # ----- convergence check -----
        with torch.no_grad():
            curr_loss = pc_with_revs.objective(x0, y0).item()

        rel_improvement = abs(prev_loss - curr_loss) / max(abs(prev_loss), 1.0)

        if rel_improvement < tol:
            stall_count += 1
            if stall_count >= patience:
                break
        else:
            stall_count = 0

        prev_loss = curr_loss

    return x0.detach(), y0.detach()


class CrowdBT_3_0:
    """
    Fast, vectorized CrowdBT implementation
    """

    def __init__(
        self,
        data,
        device,
        random_seed=42,
        penalty=0.0,
        dtype=torch.float32,
        clamp_scores=20.0,
        verbose=True,
        init_beta=0.7,
    ):
        """
        data: dict mapping reviewer i -> iterable of (winner, loser)
        """
        self.device = device
        self.dtype = dtype
        self.penalty = penalty
        self.random_seed = random_seed
        self.clamp_scores = clamp_scores
        self.verbose = verbose
        self.init_beta = init_beta

        # -------- Preprocess data once --------
        win_all = []
        los_all = []
        rev_all = []

        for i, pairs in data.items():
            if len(pairs) == 0:
                continue
            pairs = torch.tensor(
                list(pairs), device=device, dtype=torch.long
            )
            n = pairs.shape[0]
            win_all.append(pairs[:, 0])
            los_all.append(pairs[:, 1])
            rev_all.append(torch.full((n,), i, device=device))

        self.win_idx = torch.cat(win_all)
        self.los_idx = torch.cat(los_all)
        self.rev_idx = torch.cat(rev_all)

        self.num_pairs = self.win_idx.numel()

    # ----------------------------------------------------
    # Objective (fully vectorized)
    # ----------------------------------------------------
    def crowdbt_objective(self, scores, reliabilities):
        """
        Negative penalized log-likelihood
        """
        # Optional clamp for numerical stability
        scores = torch.clamp(scores, -self.clamp_scores, self.clamp_scores)

        pw = torch.exp(scores[self.win_idx])
        pl = torch.exp(scores[self.los_idx])
        denom = pw + pl

        r = reliabilities[self.rev_idx]
        prob = r * pw / denom + (1.0 - r) * pl / denom

        loss = -torch.sum(torch.log(prob + 1e-12))

        if self.penalty > 0:
            loss = loss + self.penalty * torch.sum(scores ** 2)

        return loss

    # ----------------------------------------------------
    # Alternating optimization
    # ----------------------------------------------------
    def alternate_optim(
        self,
        num_items,
        num_reviewers,
        iters=100,
        lr_x=0.05,
        lr_y=0.05,
        tol=1e-6,
        verbose=True,
        init_beta=0.7,
    ):
        """
        Returns:
            scores: (num_items,)
            reliabilities: (num_reviewers,)
        """
        torch.manual_seed(self.random_seed)

        # Initialize parameters
        scores = torch.zeros(
            num_items,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

#         reliabilities = torch.rand(
#             num_reviewers,
#             device=self.device,
#             dtype=self.dtype,
#             requires_grad=True,
#         )
        init_beta = self.init_beta
        reliabilities = torch.full((num_reviewers,), init_beta, device=self.device, dtype=self.dtype, requires_grad=True)

        # Optimizers (created ONCE)
        opt_x = torch.optim.Adam([scores], lr=lr_x)
        opt_y = torch.optim.Adam([reliabilities], lr=lr_y)

        prev_loss = None

        loop = tqdm(range(iters), disable=not verbose)
        for i in loop:

            # ---- Update item scores ----
            opt_x.zero_grad()
            loss_x = self.crowdbt_objective(scores, reliabilities.detach())
            loss_x.backward()
            opt_x.step()

            # ---- Update reviewer reliabilities ----
            opt_y.zero_grad()
            loss_y = self.crowdbt_objective(scores.detach(), reliabilities)
            loss_y.backward()
            opt_y.step()

            # Enforce [0, 1] constraint
            reliabilities.data.clamp_(0.0, 1.0)

            loss_val = loss_x.item()
            if verbose and (i % 50 == 0 or i == iters - 1):
                loop.set_postfix(loss=f"{loss_val:.6f}")

            # Convergence check
            if prev_loss is not None and abs(prev_loss - loss_val) < tol:
                print(f'CrowdBT: Converged at iteration {i} | loss_val: {loss_val}')
                break
            prev_loss = loss_val

        return scores.detach(), reliabilities.detach()

import torch
from tqdm import tqdm


class NoisyBT_3_0:
    """
    Fast, vectorized NoisyBradleyTerry implementation.

    Like CrowdBT, this models pairwise comparisons via Bradley-Terry item
    scores. Unlike CrowdBT, each worker has TWO parameters instead of one:

      - skill (reliability), in [0, 1]: how often the worker behaves like a
        genuine Bradley-Terry comparer (i.e. actually looks at item scores)
        vs. just picking based on their own innate bias.
      - bias, in [0, 1]: when a worker is "not skilled" for a given
        comparison, this is the probability they pick the `left` item
        regardless of content (e.g. a worker who just always clicks the
        first option).

    For a comparison with winner/loser (left/right is encoded via the sign
    of y, matching crowd-kit's convention: y = +1 if left wins, else -1):

        P(label) = skill * sigmoid(y * (s_left - s_right))
                   + (1 - skill) * sigmoid(y * bias_logit)

    skill and bias are stored/optimized in raw (logit) space and squashed
    with sigmoid inside the objective, exactly like crowd-kit's `gamma`/`q`.
    """

    def __init__(
        self,
        data,
        device,
        random_seed=42,
        penalty=0.0,
        dtype=torch.float32,
        clamp_scores=20.0,
        init_beta=0.7,
    ):
        """
        data: dict mapping reviewer i -> iterable of (left, right, winner)
              triples, where `winner` is whichever of `left`/`right` was
              chosen by that reviewer for that comparison.
        """
        self.device = device
        self.dtype = dtype
        self.penalty = penalty
        self.random_seed = random_seed
        self.clamp_scores = clamp_scores
        self.init_beta = init_beta

        # -------- Preprocess data once --------
        left_all = []
        right_all = []
        y_all = []  # +1 if left won, -1 if right won
        rev_all = []

        for i, triples in data.items():
            triples = list(triples)
            if len(triples) == 0:
                continue
            triples = torch.tensor(triples, device=device, dtype=torch.long)
            n = triples.shape[0]

            left = triples[:, 0]
            right = triples[:, 1]
            winner = triples[:, 2]

            y = torch.where(
                winner == left,
                torch.ones(n, device=device, dtype=dtype),
                -torch.ones(n, device=device, dtype=dtype),
            )

            left_all.append(left)
            right_all.append(right)
            y_all.append(y)
            rev_all.append(torch.full((n,), i, device=device, dtype=torch.long))

        self.left_idx = torch.cat(left_all)
        self.right_idx = torch.cat(right_all)
        self.y = torch.cat(y_all)
        self.rev_idx = torch.cat(rev_all)
        self.num_pairs = self.left_idx.numel()

    # ----------------------------------------------------
    # Objective (fully vectorized)
    # ----------------------------------------------------
    def noisybt_objective(self, scores, skill_logits, bias_logits, ref_score):
        """
        Negative penalized log-likelihood.

        scores: (num_items,) raw item scores
        skill_logits: (num_reviewers,) raw skill logits (sigmoid -> [0,1] skill)
        bias_logits: (num_reviewers,) raw bias logits (sigmoid -> [0,1] bias)
        ref_score: scalar reference score used for the regularization term
                   (mirrors crowd-kit's x[0])
        """
        scores_c = torch.clamp(scores, -self.clamp_scores, self.clamp_scores)

        s_left = scores_c[self.left_idx]
        s_right = scores_c[self.right_idx]
        y = self.y

        skill = torch.sigmoid(skill_logits[self.rev_idx])
        bias_logit = bias_logits[self.rev_idx]

        prob_bt = torch.sigmoid(y * (s_left - s_right))
        prob_bias = torch.sigmoid(y * bias_logit)

        prob = skill * prob_bt + (1.0 - skill) * prob_bias

        loss = -torch.sum(torch.log(prob + 1e-12))

        if self.penalty > 0:
            # Pulls every item's score toward a shared reference score,
            # mirroring crowd-kit's `reg` term on x[0] vs the rest.
            reg = torch.sum(
                torch.log(torch.sigmoid(ref_score - scores_c) + 1e-12)
            ) + torch.sum(torch.log(torch.sigmoid(scores_c - ref_score) + 1e-12))
            loss = loss + self.penalty * (-reg)

        return loss

    # ----------------------------------------------------
    # Alternating optimization
    # ----------------------------------------------------
    def alternate_optim(
        self,
        num_items,
        num_reviewers,
        iters=100,
        lr_x=0.05,
        lr_y=0.05,
        tol=1e-6,
        verbose=True,
        init_beta=0.7,
    ):
        """
        Returns:
            scores: (num_items,) item scores (raw, unsquashed)
            skills: (num_reviewers,) worker skills in [0, 1]
            biases: (num_reviewers,) worker biases in [0, 1]
        """
        torch.manual_seed(self.random_seed)

        # Initialize parameters
        scores = torch.zeros(
            num_items, device=self.device, dtype=self.dtype, requires_grad=True
        )
        ref_score = torch.zeros(
            1, device=self.device, dtype=self.dtype, requires_grad=True
        )
        init_beta=self.init_beta
        skill_init = torch.logit(torch.tensor(init_beta, dtype=self.dtype))
        skill_logits = torch.full(
            (num_reviewers,),
            skill_init.item(),
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        bias_logits = torch.zeros(
            num_reviewers, device=self.device, dtype=self.dtype, requires_grad=True
        )

        # Optimizers (created ONCE)
        opt_x = torch.optim.Adam([scores, ref_score], lr=lr_x)
        opt_y = torch.optim.Adam([skill_logits, bias_logits], lr=lr_y)

        prev_loss = None
        loop = tqdm(range(iters), disable=not verbose)

        for i in loop:
            # ---- Update item scores (+ reference score) ----
            opt_x.zero_grad()
            loss_x = self.noisybt_objective(
                scores,
                skill_logits.detach(),
                bias_logits.detach(),
                ref_score,
            )
            loss_x.backward()
            opt_x.step()

            # ---- Update reviewer skill/bias ----
            opt_y.zero_grad()
            loss_y = self.noisybt_objective(
                scores.detach(),
                skill_logits,
                bias_logits,
                ref_score.detach(),
            )
            loss_y.backward()
            opt_y.step()

            loss_val = loss_x.item()
            if verbose and (i % 50 == 0 or i == iters - 1):
                loop.set_postfix(loss=f"{loss_val:.6f}")

            # Convergence check
            if prev_loss is not None and abs(prev_loss - loss_val) < tol:
                print(f'FactorBT converged at iteration {i} | Loss value is {loss_val}')
                break
            prev_loss = loss_val

        skills = torch.sigmoid(skill_logits.detach())
        biases = torch.sigmoid(bias_logits.detach())

        return scores.detach(), skills, biases

"""GPU implementation of plain pairwise Bradley-Terry MLE.

This mirrors `PairwiseFcts` / `opt_pairwise` exactly in terms of the model
(plain Bradley-Terry, no skill/bias mixture) and the loss being minimized
(penalized negative log-likelihood), but runs on GPU via PyTorch tensors
and uses `torch.optim.LBFGS` in place of `scipy.optimize`'s "Newton-CG",
since scipy cannot operate on GPU tensors directly. LBFGS is the standard
GPU-native stand-in for a second-order method: it builds a curvature
approximation from gradient history, the same way Newton-CG uses
Hessian-vector products without forming the full Hessian.
"""

from typing import Optional

import torch

from collections.abc import Sequence

PairwiseData = Sequence[tuple[int, int]]

class PairwiseFctsGPU:
    """GPU-vectorized objective/gradient for plain Bradley-Terry data.

    Equivalent to `PairwiseFcts`, but:
      - `data` is preprocessed once into index tensors (win_idx, los_idx)
        instead of being iterated as a Python list of tuples per call.
      - The objective is evaluated via PyTorch ops so autograd supplies
        the gradient; there's no hand-written gradient or Hessian, since
        LBFGS only needs the objective (and PyTorch's backward pass gives
        the gradient for free).
    """

    def __init__(
        self,
        data: PairwiseData,
        penalty: float,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ):
        self.penalty = penalty
        self.device = device
        self.dtype = dtype

        win_idx = torch.tensor([w for w, _ in data], device=device, dtype=torch.long)
        los_idx = torch.tensor([l for _, l in data], device=device, dtype=torch.long)

        self.win_idx = win_idx
        self.los_idx = los_idx

    def objective(self, params: torch.Tensor) -> torch.Tensor:
        """Penalized negative log-likelihood, identical to PairwiseFcts.objective."""
        diff = params[self.win_idx] - params[self.los_idx]
        # logaddexp(0, -diff) == softplus(-diff), computed in a numerically
        # stable way by torch (equivalent to np.logaddexp(0, -diff)).
        nll = torch.nn.functional.softplus(-diff).sum()
        reg = self.penalty * torch.sum(params**2)
        return nll + reg


def _opt_gpu(
    n_items: int,
    fcts: PairwiseFctsGPU,
    initial_params: Optional[torch.Tensor],
    max_iter: Optional[int],
    tol: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if initial_params is not None:
        x0 = initial_params.to(device=device, dtype=dtype)
    else:
        x0 = torch.zeros(n_items, device=device, dtype=dtype)

    params = x0.clone().requires_grad_(True)

    # LBFGS is the GPU-native stand-in for Newton-CG: both are second-order
    # methods that avoid forming a full Hessian (Newton-CG via Hessian-vector
    # products, LBFGS via a low-rank curvature approximation built from
    # gradient history). `max_iter` here maps to LBFGS's outer `max_iter`;
    # `tol` maps to its gradient-norm tolerance, mirroring the `gtol`/`xtol`
    # semantics used for the scipy methods.
    n_iter = max_iter if max_iter is not None else 100
    optimizer = torch.optim.LBFGS(
        [params],
        max_iter=n_iter,
        tolerance_grad=tol,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = fcts.objective(params)
        loss.backward()
        return loss

    optimizer.step(closure)
    print('BT finished')

    return params.detach()


def opt_pairwise_gpu(
    n_items: int,
    data: PairwiseData,
    alpha: float = 1e-6,
    initial_params: Optional[torch.Tensor] = None,
    max_iter: Optional[int] = None,
    tol: float = 1e-6,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Compute the ML estimate of plain Bradley-Terry parameters on GPU.

    Same model and same penalized log-likelihood as `opt_pairwise`
    (`method="Newton-CG"`), but runs the optimization on GPU using
    `torch.optim.LBFGS` instead of `scipy.optimize.minimize`.

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of (winner, loser) index pairs
        Pairwise-comparison data, same format as `opt_pairwise`.
    alpha : float, optional
        Regularization strength (same role as `opt_pairwise`'s `alpha`).
    initial_params : torch.Tensor, optional
        Parameters used to initialize the iterative procedure.
    max_iter : int, optional
        Maximum number of LBFGS iterations.
    tol : float, optional
        Gradient-norm tolerance for termination.
    device : torch.device, optional
        Device to run on. Defaults to CUDA if available, else CPU.
    dtype : torch.dtype, optional
        Floating-point precision. float64 by default to match scipy's
        default precision; float32 is faster but less precise.

    Returns
    -------
    params : torch.Tensor
        The (penalized) ML estimate of model parameters, shape (n_items,).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fcts = PairwiseFctsGPU(data, alpha, device=device, dtype=dtype)
    return _opt_gpu(n_items, fcts, initial_params, max_iter, tol, device, dtype)