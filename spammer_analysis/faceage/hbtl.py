# -------------------------
# Device setup
# -------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import gc
import csv
import pickle
import random
import numpy as np
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(f"Current PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

# -------------------------
# Imports from project
# -------------------------
import sys
sys.path.insert(0, "../")
sys.path.insert(1, "../../")

from spammer_types import *
from util import *
import opt_fair
from distribution_utils import (
    crowd_bt_dist,
    logistic_preference_dist,
    comparisons_to_df,
    safe_kendalltau,
    to_numpy,
)
from metrics import compute_acc, compute_weighted_acc
from grad_em import *

# -------------------------
# Config
# -------------------------
df_path = "../../real_data/faceage/data/crowd_labels.csv"
pickle_path = "../../real_data/faceage/data/FaceAgeDF.pickle"

percents = [10, 20, 40, 60, 80, 100, 150, 233, 400, 900]
seeds = range(20, 30)

lr = 0.001
max_iter = 100000
tol = 1e-6

# -------------------------
# Load data
# -------------------------
def sort_df(df, column_name):
    return df.sort_values(by=column_name, ascending=True)

df = pd.read_csv(df_path)
df = sort_df(df, "performer")
print(df[["left", "right", "label", "performer"]].head())

with open(pickle_path, "rb") as handle:
    df_passage = pickle.load(handle)

gt_df = df_passage
size = len(df_passage)
print(size)

classes = [0] * size  # kept as in original code

# -------------------------
# Helpers
# -------------------------
def ensure_results_dir(spammer_type):
    os.makedirs(f"results/{spammer_type}", exist_ok=True)

def init_csv(csv_file):
    header = [
        "percent",
        "GradEM_acc_mean", "GradEM_acc_std",
        "GradEM_wacc_mean", "GradEM_wacc_std",
        "GradEM_tau_mean", "GradEM_tau_std",
    ]
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

def cleanup(*objs):
    for obj_name in objs:
        if obj_name in globals():
            del globals()[obj_name]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_one_seed(
    random_df,
    sd,
    lr,
    device,
    max_iter,
    df_passage,
    gt_df,
):
    """
    Runs one seed and returns (acc, wacc, tau) or None if failed/invalid.
    Keeps the same evaluation logic as the original notebook.
    """
    PC_faceage = df_to_pickle_faceage(random_df, df_passage)

    try:
        grad_em = GradientEMWrapper(
            PC_faceage,
            lr,
            sd,
            device,
            max_iter=max_iter,
        )

        r_est, beta_est = grad_em.run_algorithm()
        r_est = to_numpy(r_est)

        if np.isnan(r_est).any() or np.isnan(beta_est).any():
            return None

        tau = safe_kendalltau(r_est, gt_df["score"].to_numpy())

        if tau < 0:
            r_est = -r_est

        acc = compute_acc(gt_df, r_est, device)
        wacc = compute_weighted_acc(gt_df, r_est, device)
        tau = safe_kendalltau(r_est, gt_df["score"].to_numpy())

        return acc, wacc, tau

    except Exception as e:
        print(f"GradEM failed due to {e}")
        return None

    finally:
        if "PC_faceage" in locals():
            del PC_faceage
        if "grad_em" in locals():
            del grad_em
        if "r_est" in locals():
            del r_est
        if "beta_est" in locals():
            del beta_est
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_spammer_experiment(
    spammer_type,
    csv_file,
    generator_fn,
    generator_kwargs_fn,
    percents,
    seeds,
    lr,
    device,
    max_iter,
    df,
    df_passage,
    gt_df,
):
    ensure_results_dir(spammer_type)
    init_csv(csv_file)

    for percent in percents:
        GradEM_accs, GradEM_waccs, GradEM_taus = [], [], []

        for sd in seeds:
            kwargs = generator_kwargs_fn(percent, sd)
            random_df, _ = generator_fn(df, **kwargs)

            result = run_one_seed(
                random_df=random_df,
                sd=sd,
                lr=lr,
                device=device,
                max_iter=max_iter,
                df_passage=df_passage,
                gt_df=gt_df,
            )

            if result is None:
                del random_df
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            acc, wacc, tau = result
            GradEM_accs.append(acc)
            GradEM_waccs.append(wacc)
            GradEM_taus.append(tau)

            del random_df
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        row = [
            percent,
            np.mean(GradEM_accs), np.std(GradEM_accs),
            np.mean(GradEM_waccs), np.std(GradEM_waccs),
            np.mean(GradEM_taus), np.std(GradEM_taus),
        ]

        with open(csv_file, mode="a", newline="") as f:
            csv.writer(f).writerow(row)

        print(
            f"GradEM | "
            f"Percent: {percent} | "
            f"Acc: {np.mean(GradEM_accs):.4f} ± {np.std(GradEM_accs):.4f} | "
            f"WAcc: {np.mean(GradEM_waccs):.4f} ± {np.std(GradEM_waccs):.4f} | "
            f"Tau: {np.mean(GradEM_taus):.4f} ± {np.std(GradEM_taus):.4f}"
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# -------------------------
# Experiment definitions
# -------------------------
experiments = [
    {
        "spammer_type": "random",
        "csv_file": "results/random/hbtl.csv",
        "generator_fn": add_random_spammer,
        "generator_kwargs_fn": lambda percent, sd: {
            "percent": percent,
            "seed": sd,
        },
    },
    {
        "spammer_type": "anti",
        "csv_file": "results/anti/hbtl.csv",
        "generator_fn": add_anti_personas,
        "generator_kwargs_fn": lambda percent, sd: {
            "percent": percent,
            "seed": sd,
        },
    },
    {
        "spammer_type": "left",
        "csv_file": "results/left/hbtl.csv",
        "generator_fn": add_position_biased_spammers,
        "generator_kwargs_fn": lambda percent, sd: {
            "percent": percent,
            "position_bias": "left",
            "seed": sd,
        },
    },
    {
        "spammer_type": "right",
        "csv_file": "results/right/hbtl.csv",
        "generator_fn": add_position_biased_spammers,
        "generator_kwargs_fn": lambda percent, sd: {
            "percent": percent,
            "position_bias": "right",
            "seed": sd,
        },
    },
    {
        "spammer_type": "equal",
        "csv_file": "results/equal/hbtl.csv",
        "generator_fn": add_equal_proportion_of_all_spammers,
        "generator_kwargs_fn": lambda percent, sd: {
            "percent": percent,
            "seed": sd,
        },
    },
]

# -------------------------
# Run all experiments
# -------------------------
for exp in experiments:
    run_spammer_experiment(
        spammer_type=exp["spammer_type"],
        csv_file=exp["csv_file"],
        generator_fn=exp["generator_fn"],
        generator_kwargs_fn=exp["generator_kwargs_fn"],
        percents=percents,
        seeds=seeds,
        lr=lr,
        device=device,
        max_iter=max_iter,
        df=df,
        df_passage=df_passage,
        gt_df=gt_df,
    )
