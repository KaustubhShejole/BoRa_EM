# =====================================================
# DEVICE SETUP
# =====================================================

import os
import gc
import csv
import pickle
import numpy as np
import pandas as pd
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
print(f"Current PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

# =====================================================
# IMPORTS
# =====================================================

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

from metrics import (
    compute_acc,
    compute_weighted_acc,
)

from htcv_m import *

# =====================================================
# CONFIGURATION
# =====================================================

SEEDS = range(20, 30)

PERCENTS = [
    10,
    20,
    40,
    60,
    80,
    100,
    150,
    233,
    400,
    900,
]

LR = 0.001
MAX_ITER = 100000

CSV_HEADER = [
    "percent",
    "htcv_acc_mean",
    "htcv_acc_std",
    "htcv_wacc_mean",
    "htcv_wacc_std",
    "htcv_tau_mean",
    "htcv_tau_std",
]

# =====================================================
# DATA LOADING
# =====================================================

df_path = "../../real_data/faceage/data/crowd_labels.csv"

df = pd.read_csv(df_path)
df = df.sort_values(by="performer", ascending=True)

with open(
    "../../real_data/faceage/data/FaceAgeDF.pickle",
    "rb",
) as handle:
    df_passage = pickle.load(handle)

gt_df = df_passage

print("Dataset size:", len(df_passage))

# =====================================================
# UTILITIES
# =====================================================

def cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def initialize_csv(csv_file):
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)


def append_row(csv_file, row):
    with open(csv_file, "a", newline="") as f:
        csv.writer(f).writerow(row)


def print_results(row):
    (
        percent,
        acc_mean,
        acc_std,
        wacc_mean,
        wacc_std,
        tau_mean,
        tau_std,
    ) = row

    print(
        f"HTCV | "
        f"Percent: {percent} | "
        f"Acc: {acc_mean:.4f} ± {acc_std:.4f} | "
        f"WAcc: {wacc_mean:.4f} ± {wacc_std:.4f} | "
        f"Tau: {tau_mean:.4f} ± {tau_std:.4f}"
    )


# =====================================================
# HTCV EVALUATION
# =====================================================

def run_htcv(
    PC_faceage,
    seed,
):
    htcv = HTCVWrapper(
        PC_faceage,
        LR,
        seed,
        device,
        max_iter=MAX_ITER,
    )

    r_est, beta_est = htcv.run_algorithm()

    r_est = to_numpy(r_est)

    if np.isnan(r_est).any():
        return None

    if np.isnan(beta_est).any():
        return None

    tau = safe_kendalltau(
        r_est,
        gt_df["score"].to_numpy(),
    )

    if tau < 0:
        r_est = -r_est

    acc = compute_acc(
        gt_df,
        r_est,
        device,
    )

    wacc = compute_weighted_acc(
        gt_df,
        r_est,
        device,
    )

    tau = safe_kendalltau(
        r_est,
        gt_df["score"].to_numpy(),
    )

    return acc, wacc, tau


# =====================================================
# SINGLE PERCENT EXPERIMENT
# =====================================================

def run_percent(
    percent,
    spammer_generator,
):
    htcv_accs = []
    htcv_waccs = []
    htcv_taus = []

    for sd in SEEDS:

        random_df, _ = spammer_generator(
            df,
            percent,
            sd,
        )

        PC_faceage = df_to_pickle_faceage(
            random_df,
            df_passage,
        )

        try:

            result = run_htcv(
                PC_faceage,
                sd,
            )

            if result is None:
                continue

            acc, wacc, tau = result

            htcv_accs.append(acc)
            htcv_waccs.append(wacc)
            htcv_taus.append(tau)

        except Exception as e:
            print(f"HTCV failed due to {e}")

        finally:

            del PC_faceage
            del random_df

            cleanup()

    return [
        percent,
        np.mean(htcv_accs),
        np.std(htcv_accs),
        np.mean(htcv_waccs),
        np.std(htcv_waccs),
        np.mean(htcv_taus),
        np.std(htcv_taus),
    ]


# =====================================================
# COMPLETE SPAMMER EXPERIMENT
# =====================================================

def run_spammer_experiment(
    spammer_type,
    spammer_generator,
):
    csv_file = f"results/{spammer_type}/htcv.csv"

    initialize_csv(csv_file)

    for percent in PERCENTS:

        row = run_percent(
            percent,
            spammer_generator,
        )

        append_row(
            csv_file,
            row,
        )

        print_results(row)

        cleanup()


# =====================================================
# SPAMMER GENERATORS
# =====================================================

SPAMMER_GENERATORS = {

    "random":
        lambda df_, p, sd:
            add_random_spammer(
                df_,
                p,
                seed=sd,
            ),

    "anti":
        lambda df_, p, sd:
            add_anti_personas(
                df_,
                p,
                seed=sd,
            ),

    "left":
        lambda df_, p, sd:
            add_position_biased_spammers(
                df_,
                p,
                position_bias="left",
                seed=sd,
            ),

    "right":
        lambda df_, p, sd:
            add_position_biased_spammers(
                df_,
                p,
                position_bias="right",
                seed=sd,
            ),

    "equal":
        lambda df_, p, sd:
            add_equal_proportion_of_all_spammers(
                df_,
                p,
                seed=sd,
            ),
}

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    for spammer_type, spammer_generator in SPAMMER_GENERATORS.items():

        print("\n" + "=" * 80)
        print(f"Running HTCV for {spammer_type}")
        print("=" * 80)

        run_spammer_experiment(
            spammer_type,
            spammer_generator,
        )

    print("\nAll experiments completed.")
