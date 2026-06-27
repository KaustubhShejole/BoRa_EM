#!/usr/bin/env python3
# cleaned_experiment.py

import os
# -------------------------
# Environment / seeds
# -------------------------
# If you want to force a specific GPU, set CUDA_VISIBLE_DEVICES before torch touches CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import csv
import gc
import random
from collections import defaultdict
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the parent directory (one directory up)
# e.g., '/path/to/my_project_root'
parent_dir = os.path.join(current_dir, '..')
root_dir = os.path.join(parent_dir, '..')

# Insert the parent directory path at the beginning of sys.path
sys.path.insert(0, parent_dir)
sys.path.insert(0, root_dir)

# Project-specific imports (assumed available in your env)
from pgem_initialize_beta_1 import EMWrapper
from metrics import compute_acc, compute_weighted_acc
import opt_fair
from crowdkit.aggregation import NoisyBradleyTerry
import choix
from distribution_utils import crowd_bt_dist, logistic_preference_dist, comparisons_to_df, safe_kendalltau, to_numpy, build_synthetic_pc_dict_for_noisybt

import sys
sys.path.insert(0, "../")
sys.path.insert(0, "../../")


import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

from metrics import compute_acc, compute_weighted_acc
import opt_fair
from crowdkit.aggregation import NoisyBradleyTerry
import choix


from distribution_utils import crowd_bt_dist, memory_efficient_sample, comparisons_to_df, safe_kendalltau, to_numpy, comparisons_to_df_equally_balanced
from grad_em import *
from htcv_m import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


# -------------------------
# Parameters
# -------------------------
# Make sure N_array and K_array have the same length
m = 20000
K = 1000
N_array = [50, 100, 250, 500, 1000, 2000, 2250, 2500]
# N_array = [50]
# K_array = [2]

# number of comparisons per experiment
# m = 5000
max_iter = 30000
lr = 1e-3
tol = 1e-6
# CSV output
csv_file = "results/vary_n_0_1.csv"
os.makedirs("results", exist_ok=True)

# -------------------------
# Helper functions
# -------------------------
def get_ground_truth_df(true_r):
    return pd.DataFrame({
        'label': list(range(len(true_r))),
        'score': true_r
    })

# -------------------------
# Write CSV header
# -------------------------
header = [
    "N", "K", "M", 
    # PGEM (mean, std) for acc, wacc, tau
    "PGEM_acc_mean", "PGEM_acc_std",
    "PGEM_wacc_mean", "PGEM_wacc_std",
    "PGEM_tau_mean", "PGEM_tau_std",
    # BT single-run
    "BT_acc_mean", "BT_acc_std",
    "BT_wacc_mean", "BT_wacc_std",
    "BT_tau_mean", "BT_tau_std",
    # BARP single-run
    "BARP_acc_mean", "BARP_acc_std",
    "BARP_wacc_mean", "BARP_wacc_std",
    "BARP_tau_mean", "BARP_tau_std",
    # RankCentrality single-run
    "RC_acc_mean", "RC_acc_std",
    "RC_wacc_mean", "RC_wacc_std",
    "RC_tau_mean", "RC_tau_std",
    # FactorBT single-run
    "FactorBT_acc_mean", "FactorBT_acc_std",
    "FactorBT_wacc_mean", "FactorBT_wacc_std",
    "FactorBT_tau_mean", "FactorBT_tau_std",
    # CrowdBT (mean,std) for acc, wacc, tau
    "CrowdBT_acc_mean", "CrowdBT_acc_std",
    "CrowdBT_wacc_mean", "CrowdBT_wacc_std",
    "CrowdBT_tau_mean", "CrowdBT_tau_std",
    # HBTL and HTCV
    "HTCV_acc_mean", "HTCV_acc_std",
    "HTCV_wacc_mean", "HTCV_wacc_std",
    "HTCV_tau_mean", "HTCV_tau_std",
    
    "HBTL_acc_mean", "HBTL_acc_std",
    "HBTL_wacc_mean", "HBTL_wacc_std",
    "HBTL_tau_mean", "HBTL_tau_std",
]

with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    
def mean_std(x):
    return (0, 0) if len(x) == 0 else (np.mean(x), np.std(x))

verbose = False
# -------------------------
# Main experiment loop
# -------------------------
for N in N_array:
    # create GT rewards and worker betas
    PGEM_accs, PGEM_waccs, PGEM_taus = [], [], []
    FactorBT_accs, FactorBT_waccs, FactorBT_taus = [], [], []
    BT_accs, BT_waccs, BT_taus = [], [], []
    BARP_accs, BARP_waccs, BARP_taus = [], [], []
    CrowdBT_accs, CrowdBT_waccs, CrowdBT_taus = [], [], []
    RC_accs, RC_waccs, RC_taus = [], [], []
    HTCV_accs, HTCV_waccs, HTCV_taus = [], [], []
    HBTL_accs, HBTL_waccs, HBTL_taus = [], [], []
    
    SEEDS = range(20, 30)
    iters = 0
    for sd in tqdm(SEEDS):
        iters = iters+1
        torch.manual_seed(sd)
        np.random.seed(sd)
        random.seed(sd)

        true_r = (6 * torch.rand(N, device=device) - 3).clone()
        true_r = true_r - true_r.mean()
        gt_df = get_ground_truth_df(true_r.detach().cpu().numpy())
        gt_scores = gt_df['score'].to_numpy()
        true_beta = torch.rand(K, device=device)

#         sample_fn, flat_probs = logistic_preference_dist(true_r, true_beta)
#         samples = sample_fn(m)
        
    
        samples = memory_efficient_sample(true_r, true_beta, m)

        winners = samples[:, 0]
        losers = samples[:, 1]
        annotators = samples[:, 2]

        data_tensors = (winners, losers, annotators)

#         df = comparisons_to_df_equally_balanced(samples)

        PC_passage = defaultdict(list)
        
        sample_np = samples.cpu().numpy()

        for winner, loser, worker in sample_np:
            PC_passage[int(worker)].append((int(winner), int(loser)))

        PC_faceage = PC_passage
        all_pc_faceage = opt_fair._pc_without_reviewers(PC_faceage)
        size = N
        classes = [0] * size
        
        for seed in range(1):
            model = GradientEM(N, K, random_seed=seed)
            model = model.to(device)

            # Separate optimizers for alternating updates
            opt_r = torch.optim.Adam(model.item_rewards.parameters(), lr=lr)
            opt_beta = torch.optim.Adam(model.worker_betas.parameters(), lr=lr)

            r, beta = train_with_convergence_hbtl(model, data_tensors, opt_r, opt_beta, max_epochs=max_iter, tol=tol, verbose=verbose)

            # Skip if ANY element is NaN
            if torch.isnan(r).any() or torch.isnan(beta).any():
                print("Skipping nan")
                continue

            # Convert the estimated item scores (r_est) to a NumPy array immediately.
            r_est_np = to_numpy(r)
             # Check rank correlation between estimated scores and ground truth scores.
            # If the correlation (tau) is negative, the scores are flipped (non-identifiability).
            # This is more stable than checking if accuracy is below 0.5.
            current_tau = safe_kendalltau(r_est_np, gt_scores)

            if current_tau < 0:
                # Flip the sign of the scores to match the convention (higher score = better item)
                r_est_np = -r_est_np

            # Since r_est_np is the final, correctly-signed array, use it directly for all metrics.
            HBTL_accs.append(compute_acc(gt_df, r_est_np, device))
            HBTL_waccs.append(compute_weighted_acc(gt_df, r_est_np, device))
            # Recalculate Tau for the correctly-signed scores (it should now be positive)
            HBTL_taus.append(safe_kendalltau(r_est_np, gt_scores))

            del model
        
        for seed in range(10, 11):
            model = HTCV(N, K, random_seed=seed)
            model = model.to(device)

            # Separate optimizers for alternating updates
            opt_r = torch.optim.Adam(model.item_rewards.parameters(), lr=lr)
            opt_beta = torch.optim.Adam(model.worker_betas.parameters(), lr=lr)

            r, beta = train_with_convergence_htcv(model, data_tensors, opt_r, opt_beta, max_epochs=max_iter, tol=tol, verbose=verbose)

            # Skip if ANY element is NaN
            if torch.isnan(r).any() or torch.isnan(beta).any():
                print("Skipping nan")
                continue

            # Convert the estimated item scores (r_est) to a NumPy array immediately.
            r_est_np = to_numpy(r)

             # Check rank correlation between estimated scores and ground truth scores.
            # If the correlation (tau) is negative, the scores are flipped (non-identifiability).
            # This is more stable than checking if accuracy is below 0.5.
            current_tau = safe_kendalltau(r_est_np, gt_scores)

            if current_tau < 0:
                # Flip the sign of the scores to match the convention (higher score = better item)
                r_est_np = -r_est_np

            HTCV_accs.append(compute_acc(gt_df, r_est_np, device))
            HTCV_waccs.append(compute_weighted_acc(gt_df, r_est_np, device))
            HTCV_taus.append(safe_kendalltau(r_est_np, gt_scores))
    #                 print(HTCV_accs)
            del model
        
        # === PGEM (averaged over seeds) ===
        
        for seed in range(1):
            # --- 1. Run the EM algorithm (Keep this section) ---
            pg = EMWrapper(PC_faceage, max_iter=max_iter, device=device, random_seed=seed, num_items=N, num_workers=K, verbose=verbose)
            r_est_tensor, beta_est_tensor, ll = pg.run_algorithm()
            
            # Skip if ANY element is NaN
            
            
            # --- 2. Convert to NumPy once for unified processing (Improvement) ---
            # Convert the estimated item scores (r_est) to a NumPy array immediately.
            r_est_np = to_numpy(r_est_tensor)

            # --- 3. Robust Sign-Flipping based on Kendall's Tau (Major Improvement) ---
            # Check rank correlation between estimated scores and ground truth scores.
            # If the correlation (tau) is negative, the scores are flipped (non-identifiability).
            # This is more stable than checking if accuracy is below 0.5.
            current_tau = safe_kendalltau(r_est_np, gt_scores)

            if current_tau < 0:
                # Flip the sign of the scores to match the convention (higher score = better item)
                r_est_np = -r_est_np

            # --- 4. Calculate and Append Metrics (Simplified and Cleaned) ---
            # Since r_est_np is the final, correctly-signed array, use it directly for all metrics.
            PGEM_accs.append(compute_acc(gt_df, r_est_np, device))
            PGEM_waccs.append(compute_weighted_acc(gt_df, r_est_np, device))
            # Recalculate Tau for the correctly-signed scores (it should now be positive)
            PGEM_taus.append(safe_kendalltau(r_est_np, gt_scores))
            del pg
        
        
        for seed in range(1):
            try:
                crowdbt_test = opt_fair.CrowdBT_3_0(data=PC_faceage, penalty=0, device=device, random_seed=seed)
                crowdbt_scores, _ = crowdbt_test.alternate_optim(size, K, lr_x=lr, lr_y=lr, tol=tol, iters=max_iter, verbose=verbose)
                crowdbt_scores_np = to_numpy(crowdbt_scores)
                tau = safe_kendalltau(crowdbt_scores_np, gt_scores)
                if tau < 0:
                    crowdbt_scores_np = -crowdbt_scores_np
                CrowdBT_accs.append(compute_acc(gt_df, crowdbt_scores_np, device))
                CrowdBT_waccs.append(compute_weighted_acc(gt_df, crowdbt_scores_np, device))
                CrowdBT_taus.append(safe_kendalltau(crowdbt_scores_np, gt_scores))
#                 print(CrowdBT_accs)
            except Exception as e:
                print(f"CrowdBT seed {seed} failed for N={N},K={K} with error {e}")
                continue

        
        
        # === BT (choix) ===
        try:
            bt_scores = opt_fair.opt_pairwise_gpu(size, all_pc_faceage, alpha=0, initial_params=None, max_iter=max_iter, tol=tol, device=device)
            bt_scores = to_numpy(bt_scores)
        
#             BT_tau = safe_kendalltau(bt_scores, gt_scores)
#             if BT_tau < 0:
#                 bt_scores = -bt_scores
            BT_acc = compute_acc(gt_df, bt_scores, device)
            BT_wacc = compute_weighted_acc(gt_df, bt_scores, device)
            BT_tau = safe_kendalltau(bt_scores, gt_scores)
            BT_accs.append(BT_acc)
            BT_waccs.append(BT_wacc)
            BT_taus.append(BT_tau)
        except Exception as e:
            # fallback: zeros
            print(f"choix.opt_pairwise failed for N={N},K={K} with error {e}")

        # === BARP (opt_fair) ===
        try:
            FaceAge = opt_fair.BARP(data=PC_faceage, penalty=0, classes=classes, device=device)
            annot_bt_temp, annot_bias = opt_fair._alternate_optim_torch(size, K, FaceAge, lr_x=lr, lr_y=lr, tol=tol, iters=max_iter, verbose=verbose)
            annot_bt_np = to_numpy(annot_bt_temp)
#             BARP_tau = safe_kendalltau(annot_bt_np, gt_scores)
#             if BARP_tau < 0:
#                 annot_bt_np = -annot_bt_np
            BARP_acc = compute_acc(gt_df, annot_bt_np, device)
            BARP_wacc = compute_weighted_acc(gt_df, annot_bt_np, device)
            BARP_tau = safe_kendalltau(annot_bt_np, gt_scores)
            BARP_accs.append(BARP_acc)
            BARP_waccs.append(BARP_wacc)
            BARP_taus.append(BARP_tau)
        except Exception as e:
            print(f"BARP failed for N={N},K={K} with error {e}")

        # === RankCentrality (opt_fair) ===
        try:
            rc_obj = opt_fair.RankCentrality(device)
            A = rc_obj.matrix_of_comparisons(size, all_pc_faceage)
            P = rc_obj.trans_prob(A)
            pi = rc_obj.stationary_dist(P)
            rc_scores = np.log(np.maximum(to_numpy(pi), 1e-12))
#             RC_tau = safe_kendalltau(rc_scores, gt_scores)
#             if RC_tau < 0:
#                 rc_scores = -rc_scores
            RC_acc = compute_acc(gt_df, rc_scores, device)
            RC_tau = safe_kendalltau(rc_scores, gt_scores)
            RC_wacc = compute_weighted_acc(gt_df, rc_scores, device)
            RC_accs.append(RC_acc)
            RC_waccs.append(RC_wacc)
            RC_taus.append(RC_tau)
        except Exception as e:
            print(f"RankCentrality failed for N={N},K={K} with error {e}")
        


        # === FactorBT (NoisyBradleyTerry from crowdkit) ===
        try:

            PC_faceage_noisybt, _performer_label_dict, _item_label_dict =  build_synthetic_pc_dict_for_noisybt(samples, N)
            noisybt_test = opt_fair.NoisyBT_3_0(data=PC_faceage_noisybt, penalty=0, device=device)
            noisybt_scores, noisybt_skills, noisybt_biases = noisybt_test.alternate_optim(size, K, lr_x=lr, lr_y=lr, tol=tol, iters=max_iter, verbose=verbose)
            factorbt_scores =  to_numpy(noisybt_scores)
            FactorBT_tau = safe_kendalltau(factorbt_scores, gt_scores)
            if FactorBT_tau < 0:
                factorbt_scores = -factorbt_scores
            FactorBT_acc = compute_acc(gt_df, factorbt_scores, device)
            FactorBT_wacc = compute_weighted_acc(gt_df, factorbt_scores, device)
            FactorBT_tau = safe_kendalltau(factorbt_scores, gt_scores)
            FactorBT_accs.append(FactorBT_acc)
            FactorBT_waccs.append(FactorBT_wacc)
            FactorBT_taus.append(FactorBT_tau)
        except Exception as e:
            traceback.print_exc()
            print(f"FactorBT (NoisyBradleyTerry) failed for N={N},K={K} with error {e}")

        
        print(f'Iteration: {iters}, N: {N}, K: {K}, m: {m}')
        pgem_mean, pgem_std = mean_std(PGEM_accs)
        bt_mean, bt_std = mean_std(BT_accs)
        barp_mean, barp_std = mean_std(BARP_accs)
        rc_mean, rc_std = mean_std(RC_accs)
        factor_mean, factor_std = mean_std(FactorBT_accs)
        crowd_mean, crowd_std = mean_std(CrowdBT_accs)
        htcv_mean, htcv_std = mean_std(HTCV_accs)
        hbtl_mean, hbtl_std = mean_std(HBTL_accs)

        print(
            f"PGEM={pgem_mean:.4f}±{pgem_std:.4f} | "
            f"BT={bt_mean:.4f}±{bt_std:.4f} | "
            f"BARP={barp_mean:.4f}±{barp_std:.4f} | "
            f"RC={rc_mean:.4f}±{rc_std:.4f} | "
            f"FactorBT={factor_mean:.4f}±{factor_std:.4f} | "
            f"CrowdBT={crowd_mean:.4f}±{crowd_std:.4f} | "
            f"HTCV={htcv_mean:.4f}±{htcv_std:.4f} | "
            f"HBTL={hbtl_mean:.4f}±{hbtl_std:.4f}"
        )
        
        del true_r, true_beta, samples, gt_df  # delete variables
        del sample_np
        del PC_passage
        del PC_faceage
        del all_pc_faceage
        del winners
        del losers
        del annotators
        del data_tensors
        gc.collect()
        torch.cuda.empty_cache()
    
    # --- Save row ---
    row = [
        N, K, m,
        *mean_std(PGEM_accs),
        *mean_std(PGEM_waccs),
        *mean_std(PGEM_taus),
        *mean_std(BT_accs),
        *mean_std(BT_waccs),
        *mean_std(BT_taus),
        *mean_std(BARP_accs),
        *mean_std(BARP_waccs),
        *mean_std(BARP_taus),
        *mean_std(RC_accs),
        *mean_std(RC_waccs),
        *mean_std(RC_taus),
        *mean_std(FactorBT_accs),
        *mean_std(FactorBT_waccs),
        *mean_std(FactorBT_taus),
        *mean_std(CrowdBT_accs),
        *mean_std(CrowdBT_waccs),
        *mean_std(CrowdBT_taus),
        *mean_std(HTCV_accs),
        *mean_std(HTCV_waccs),
        *mean_std(HTCV_taus),
        *mean_std(HBTL_accs),
        *mean_std(HBTL_waccs),
        *mean_std(HBTL_taus),
    ]

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    pgem_mean, pgem_std = mean_std(PGEM_accs)
    bt_mean, bt_std = mean_std(BT_accs)
    barp_mean, barp_std = mean_std(BARP_accs)
    rc_mean, rc_std = mean_std(RC_accs)
    factor_mean, factor_std = mean_std(FactorBT_accs)
    crowd_mean, crowd_std = mean_std(CrowdBT_accs)
    htcv_mean, htcv_std = mean_std(HTCV_accs)
    hbtl_mean, hbtl_std = mean_std(HBTL_accs)

    print(
        f"N={N}, K={K} | "
        f"PGEM={pgem_mean:.4f}±{pgem_std:.4f} | "
        f"BT={bt_mean:.4f}±{bt_std:.4f} | "
        f"BARP={barp_mean:.4f}±{barp_std:.4f} | "
        f"RC={rc_mean:.4f}±{rc_std:.4f} | "
        f"FactorBT={factor_mean:.4f}±{factor_std:.4f} | "
        f"CrowdBT={crowd_mean:.4f}±{crowd_std:.4f} | "
        f"HTCV={htcv_mean:.4f}±{htcv_std:.4f} | "
        f"HBTL={hbtl_mean:.4f}±{hbtl_std:.4f}"
    )

print(f"Metrics saved to {csv_file}")