# -*- coding: utf-8 -*-
"""ablation.py

Converted from ablation.ipynb (originally a Colab notebook):
    https://colab.research.google.com/drive/1KNprBsCcKpnQP0ZBEhiBOHkrvIs1kxiJ

Runs the HBTL, PG-EM, CrowdBT, and NoisyBT ablations over a grid of beta
values, and saves every results dict to a single JSON file at the end.
"""

import os
import sys
import time
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# choix / scipy / matplotlib / seaborn are imported in the original notebook
# but not directly used below; kept for parity with the source file in case
# downstream code or future edits rely on them being importable.
import choix  # noqa: F401
from scipy.optimize import minimize  # noqa: F401
import scipy.stats as stats  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
from matplotlib import colors  # noqa: F401
import seaborn as sns  # noqa: F401

sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("../../../"))

from metrics import compute_acc, compute_weighted_acc
from opt_fair import *
import opt_fair
from distribution_utils import safe_kendalltau, to_numpy

from grad_em import *
from pgem_initialize_beta_1 import EMWrapper


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _json_default(obj):
    """Fallback serializer for numpy / torch types that json can't handle."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (set,)):
        return list(obj)
    # Last resort: stringify so the dump never hard-fails.
    return str(obj)


def results_to_json_safe(d):
    """Recursively convert a results dict's values into JSON-safe types."""
    safe = {}
    for k, v in d.items():
        key = str(k)  # JSON object keys must be strings (betas are floats)
        if isinstance(v, list):
            safe[key] = [
                item.item() if isinstance(item, np.generic) else item
                for item in v
            ]
        else:
            safe[key] = v
    return safe


def setup_device():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(f"Current PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    return device


def load_data():
    with open("../data/FaceAgePC.pickle", "rb") as handle:
        PC_faceage = pickle.load(handle)
    with open("../data/FaceAgeDF.pickle", "rb") as handle:
        df_faceage = pickle.load(handle)
    return PC_faceage, df_faceage


def build_pc_dict_for_noisybt(
    df, df_faceage, worker_col="worker", left_col="left", right_col="right", label_col="label"
):
    """
    Builds the dict NoisyBT_3_0 expects: {worker_id: [(left, right, winner), ...]}

    Mirrors the exact item/worker id mapping used to build PC_faceage_spm:
      - worker ids: position of first appearance in df[worker_col].unique()
      - item ids: position of df_faceage['full_path'] (matched via Path(...).name,
        since df['left']/df['right']/df['label'] are full URLs but full_path
        entries are basenames)

    Unlike PC_faceage_spm (which collapses each row to (winner, loser)), this
    keeps the left/right slot assignment, since NoisyBT's bias parameter needs
    to know which item was shown in the "left" position.

    Returns the dict AND the two id mappings, so you can invert them later
    if needed (e.g. to map scores back to filenames).
    """
    unique_performers = list(df[worker_col].unique())
    performer_label_dict = {performer: i for i, performer in enumerate(unique_performers)}

    item_labels = list(df_faceage["full_path"])
    item_label_dict = {item: i for i, item in enumerate(item_labels)}

    pc_dict = defaultdict(list)
    for performer, group in df.groupby(worker_col):
        key = performer_label_dict[performer]
        for _, row in group.iterrows():
            left = Path(row[left_col]).name
            right = Path(row[right_col]).name
            winner = Path(row[label_col]).name

            left_id = item_label_dict[left]
            right_id = item_label_dict[right]
            winner_id = item_label_dict[winner]

            pc_dict[key].append((left_id, right_id, winner_id))

    return dict(pc_dict), performer_label_dict, item_label_dict


def sort_df(df, column_name):
    # Sort by a specific column (replace 'column_name' with your column)
    df_sorted = df.sort_values(by=column_name, ascending=True)  # or ascending=False
    return df_sorted


# ---------------------------------------------------------------------------
# Ablation runners
# ---------------------------------------------------------------------------
def run_hbtl(PC_faceage, df_faceage, device, betas_1, lr, max_iter):
    """### HBTL"""
    HBTL = defaultdict(list)
    gradem_time = []
    for sd in range(1):
        for beta in betas_1:
            start = time.time()
            grad_em = GradientEMWrapper(PC_faceage, lr, sd, device, max_iter=max_iter, init_beta=beta)
            r_est, beta_est = grad_em.run_algorithm()
            end = time.time()
            gradem_time.append(end - start)

            r_est_np = to_numpy(r_est)
            gt_scores = to_numpy(df_faceage["score"].tolist())
            current_tau = safe_kendalltau(r_est_np, gt_scores)
            if current_tau < 0:
                r_est_np = -r_est_np
            grad_acc = compute_acc(df_faceage, 1 * r_est_np, device)
            grad_wacc = compute_weighted_acc(df_faceage, 1 * r_est_np, device)
            grad_tau = safe_kendalltau(r_est_np, gt_scores)

            HBTL[beta] = [grad_acc, grad_wacc, grad_tau]
            print(HBTL[beta])

    print(HBTL)
    return HBTL, gradem_time


def run_pgem(PC_faceage, df_faceage, device, betas_1, max_iter):
    """### PG EM"""
    PGEM = defaultdict(list)
    pgem_time = []
    for sd in range(1):
        for beta in betas_1:
            start = time.time()
            pg = EMWrapper(PC_faceage, max_iter, device, sd, init_beta=beta)
            r_est, beta_est, ll = pg.run_algorithm()
            end = time.time()
            pgem_time.append(end - start)

            if np.isnan(r_est).any() or np.isnan(beta_est).any() or np.isnan(ll):
                print("Skipping nan")
                continue

            r_est_np = to_numpy(r_est)

            gt_scores = to_numpy(df_faceage["score"].tolist())
            current_tau = safe_kendalltau(r_est_np, gt_scores)
            if current_tau < 0:
                r_est_np = -r_est_np
            pgem_acc = compute_acc(df_faceage, 1 * r_est_np, device)
            pgem_wacc = compute_weighted_acc(df_faceage, 1 * r_est_np, device)
            pgem_tau = safe_kendalltau(r_est_np, gt_scores)

            PGEM[beta] = [pgem_acc, pgem_wacc, pgem_tau]
            print(PGEM[beta])

    print(PGEM)
    return PGEM, pgem_time


def run_crowdbt(PC_faceage, df_faceage, device, betas_2, size, num_reviewers, lr, tol, max_iter):
    """### CrowdBT"""
    gt_scores = to_numpy(df_faceage["score"].tolist())

    CrowdBT = defaultdict(list)
    K = num_reviewers
    gt_df = df_faceage
    crowdbt_time = []
    for seed in range(1):
        try:
            for beta in betas_2:
                start = time.time()
                crowdbt_test = opt_fair.CrowdBT_3_0(
                    data=PC_faceage, penalty=0, device=device, random_seed=seed, init_beta=beta,
                )
                crowdbt_scores, _ = crowdbt_test.alternate_optim(
                    size, K, lr_x=lr, lr_y=lr, tol=tol, iters=max_iter, verbose=False
                )
                end = time.time()
                crowdbt_time.append(end - start)
                r_est_np = to_numpy(crowdbt_scores)
                gt_scores = to_numpy(df_faceage["score"].tolist())
                current_tau = safe_kendalltau(r_est_np, gt_scores)
                if current_tau < 0:
                    r_est_np = -r_est_np
                crowdbt_acc = compute_acc(df_faceage, 1 * r_est_np, device)
                crowdbt_wacc = compute_weighted_acc(df_faceage, 1 * r_est_np, device)
                crowdbt_tau = safe_kendalltau(r_est_np, gt_scores)
                CrowdBT[beta] = [crowdbt_acc, crowdbt_wacc, crowdbt_tau]
                print(CrowdBT[beta])
        except Exception as e:
            print(e)

    print(CrowdBT)
    return CrowdBT, crowdbt_time


def run_noisybt(df_faceage, device, betas_2, size, num_reviewers, lr, tol, max_iter):
    """### FactorBT (NoisyBT)"""
    df = pd.read_csv("../data/crowd_labels.csv")
    df = df.rename(columns={"performer": "worker"})
    print(df.head())

    gt_df = df_faceage
    print(gt_df.head())

    NoisyBT = defaultdict(list)
    K = num_reviewers
    noisybt_time = []

    # Build the (left, right, winner) dict NoisyBT_3_0 needs, from the same
    # crowd_labels.csv-derived df used by the original NoisyBradleyTerry.
    # `df` must have integer-coded 'worker', 'left', 'right', 'label' columns
    # aligned with `size`/num_items and gt_df['score'] (see build_pc_dict.py).
    PC_faceage_noisybt, _performer_label_dict, _item_label_dict = build_pc_dict_for_noisybt(df, df_faceage)

    for seed in range(1):
        try:
            for beta in betas_2:
                start = time.time()
                noisybt_test = opt_fair.NoisyBT_3_0(
                    data=PC_faceage_noisybt, penalty=0, device=device, random_seed=seed, init_beta=beta,
                )
                noisybt_scores, noisybt_skills, noisybt_biases = noisybt_test.alternate_optim(
                    size, K, lr_x=lr, lr_y=lr, tol=tol, iters=max_iter, verbose=False, init_beta=beta
                )
                end = time.time()
                noisybt_time.append(end - start)
                r_est_np = to_numpy(noisybt_scores)
                gt_scores = to_numpy(gt_df["score"].tolist())
                current_tau = safe_kendalltau(r_est_np, gt_scores)
                if current_tau < 0:
                    r_est_np = -r_est_np
                noisybt_acc = compute_acc(gt_df, 1 * r_est_np, device)
                noisybt_wacc = compute_weighted_acc(gt_df, 1 * r_est_np, device)
                noisybt_tau = safe_kendalltau(r_est_np, gt_scores)
                NoisyBT[beta] = [noisybt_acc, noisybt_wacc, noisybt_tau]
        except Exception as e:
            print(e)

    print(NoisyBT)
    return NoisyBT, noisybt_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = setup_device()

    PC_faceage, df_faceage = load_data()
    print(df_faceage)

    all_pc_faceage = opt_fair._pc_without_reviewers(PC_faceage)

    size = len(df_faceage)
    print(size)
    print(len(all_pc_faceage))

    gt_scores = to_numpy(df_faceage["score"].tolist())  # noqa: F841 (kept for parity)

    max_iter = 30000
    lr = 0.001
    tol = 1e-6

    betas_1 = [
        -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    ]

    betas_2 = [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0,
    ]

#     # --- HBTL ---
#     HBTL, gradem_time = run_hbtl(PC_faceage, df_faceage, device, betas_1, lr, max_iter)

#     # --- PG EM ---
#     PGEM, pgem_time = run_pgem(PC_faceage, df_faceage, device, betas_1, max_iter)

#     # --- CrowdBT ---
#     crowd_labels = pd.read_csv("../data/crowd_labels.csv")
#     num_reviewers = crowd_labels["performer"].nunique()
#     print(device)

#     CrowdBT, crowdbt_time = run_crowdbt(
#         PC_faceage, df_faceage, device, betas_2, size, num_reviewers, lr, tol, max_iter
#     )

    # --- FactorBT / NoisyBT ---
    crowd_labels = pd.read_csv("../data/crowd_labels.csv")
    num_reviewers = crowd_labels["performer"].nunique()
    print(device)

    NoisyBT, noisybt_time = run_noisybt(
        df_faceage, device, betas_2, size, num_reviewers, lr, tol, max_iter
    )

    # --- Save everything to a single JSON file ---
    output = {
#         "HBTL": results_to_json_safe(HBTL),
#         "PGEM": results_to_json_safe(PGEM),
#         "CrowdBT": results_to_json_safe(CrowdBT),
        "NoisyBT": results_to_json_safe(NoisyBT),
        "timing": {
#             "gradem_time": gradem_time,
#             "pgem_time": pgem_time,
#             "crowdbt_time": crowdbt_time,
            "noisybt_time": noisybt_time,
        },
    }

    out_path = "ablation_results_noisybt.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)

    print(f"Saved all results to {out_path}")


if __name__ == "__main__":
    main()