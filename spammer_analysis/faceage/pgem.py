# -------------------------
# Device setup
# -------------------------
import os
import sys
import csv
import gc
import pickle
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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

# -------------------------
# Imports from project
# -------------------------
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
from pgem_initialize_beta_1 import EMWrapper

# -------------------------
# Global experiment settings
# -------------------------
DF_PATH = "../../real_data/faceage/data/crowd_labels.csv"
PASSAGE_PICKLE_PATH = "../../real_data/faceage/data/FaceAgeDF.pickle"
PERCENTS = [10, 20, 40, 60, 80, 100, 150, 233, 400, 900]
SEEDS = range(20, 30)
MAX_ITER_PGEM = 30000

CSV_HEADER = [
    "percent",
    "PGEM_acc_mean", "PGEM_acc_std",
    "PGEM_wacc_mean", "PGEM_wacc_std",
    "PGEM_tau_mean", "PGEM_tau_std",
]


# -------------------------
# Data loading helpers
# -------------------------
def sort_df(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Sort a dataframe by one column in ascending order."""
    return df.sort_values(by=column_name, ascending=True)


def load_passage_data(df_path: str, passage_pickle_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the passage csv and the ground-truth PassageDF pickle."""
    df = pd.read_csv(df_path)
    df = sort_df(df, "performer")

    with open(passage_pickle_path, "rb") as handle:
        df_passage = pickle.load(handle)

    return df, df_passage


# -------------------------
# CSV helpers
# -------------------------
def prepare_csv(csv_file: str, header: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_csv_row(csv_file: str, row: Sequence[object]) -> None:
    with open(csv_file, mode="a", newline="") as f:
        csv.writer(f).writerow(row)


# -------------------------
# PGEM helpers
# -------------------------
def run_pgem_once(
    PC_faceage,
    max_iter_pgem: int,
    device: torch.device,
    random_seed: int = 0,
):
    """Run PGEM once and return (r_est, beta_est, ll)."""
    pg = EMWrapper(
        PC_faceage,
        max_iter_pgem,
        device,
        random_seed=random_seed,
    )
    r_est, beta_est, ll = pg.run_algorithm()
    return pg, r_est, beta_est, ll


def evaluate_ranking(
    r_est,
    gt_scores: np.ndarray,
    gt_df: pd.DataFrame,
    device: torch.device,
):
    """Compute tau, accuracy, and weighted accuracy, with sign correction if needed."""
    tau = safe_kendalltau(r_est, gt_scores)
    if tau < 0:
        r_est = -r_est

    acc = compute_acc(gt_df, r_est, device)
    wacc = compute_weighted_acc(gt_df, r_est, device)
    tau = safe_kendalltau(r_est, gt_scores)
    return r_est, acc, wacc, tau


def is_valid_pgem_output(r_est, beta_est, ll) -> bool:
    return not (
        np.isnan(r_est).any()
        or np.isnan(beta_est).any()
        or np.isnan(ll)
    )


# -------------------------
# Experiment core
# -------------------------
def run_spammer_experiment(
    *,
    spammer_type: str,
    csv_file: str,
    df: pd.DataFrame,
    df_passage: pd.DataFrame,
    gt_df: pd.DataFrame,
    percents: Sequence[int],
    seeds: Iterable[int],
    max_iter_pgem: int,
    device: torch.device,
    spammer_fn: Callable[..., Tuple[pd.DataFrame, object]],
    spammer_kwargs_fn: Callable[[int, int], Dict],
    print_progress: bool = False,
) -> None:
    """
    Generic experiment runner matching the original notebook logic.

    spammer_fn should be one of:
      - add_random_spammer
      - add_anti_personas
      - add_position_biased_spammers
      - add_equal_proportion_of_all_spammers

    spammer_kwargs_fn(percent, seed) returns the keyword arguments passed to spammer_fn.
    """
    prepare_csv(csv_file, CSV_HEADER)
    gt_scores = gt_df["score"].to_numpy()

    for percent in percents:
        PGEM_accs, PGEM_waccs, PGEM_taus = [], [], []

        for sd in seeds:
            # Create spammed data using the exact same helper pattern as the notebook.
            random_df, _ = spammer_fn(df, percent, **spammer_kwargs_fn(percent, sd))
            PC_faceage = df_to_pickle_faceage(random_df, df_passage)

            # Preserve the original anti/right/left/equal cleanup pattern where random_df is discarded early.
            del random_df

            try:
                pg, r_est, beta_est, ll = run_pgem_once(
                    PC_faceage,
                    max_iter_pgem,
                    device,
                    random_seed=0,
                )

                r_est = to_numpy(r_est)

                if not is_valid_pgem_output(r_est, beta_est, ll):
                    continue

                r_est, PGEM_acc, PGEM_wacc, PGEM_tau = evaluate_ranking(
                    r_est,
                    gt_scores,
                    gt_df,
                    device,
                )

                PGEM_accs.append(PGEM_acc)
                PGEM_waccs.append(PGEM_wacc)
                PGEM_taus.append(PGEM_tau)

            except Exception as e:
                print(f"PGEM failed due to {e}")

            finally:
                # Keep cleanup behavior close to the original notebook.
                try:
                    del PC_faceage
                except Exception:
                    pass

                for name in ["pg", "r_est", "beta_est", "ll"]:
                    if name in locals():
                        del locals()[name]

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        row = [
            percent,
            np.mean(PGEM_accs), np.std(PGEM_accs),
            np.mean(PGEM_waccs), np.std(PGEM_waccs),
            np.mean(PGEM_taus), np.std(PGEM_taus),
        ]
        append_csv_row(csv_file, row)

        if print_progress:
            print(
                f"PGEM | "
                f"Percent: {percent} | "
                f"Acc: {np.mean(PGEM_accs):.4f} ± {np.std(PGEM_accs):.4f} | "
                f"WAcc: {np.mean(PGEM_waccs):.4f} ± {np.std(PGEM_waccs):.4f} | "
                f"Tau: {np.mean(PGEM_taus):.4f} ± {np.std(PGEM_taus):.4f}"
            )


# -------------------------
# Spam configuration wrappers
# -------------------------
def run_random_spammers(df, df_passage, gt_df, percents, seeds, max_iter_pgem, device):
    spammer_type = "random"
    csv_file = f"results/{spammer_type}/pgem_initialize_beta_1.csv"

    run_spammer_experiment(
        spammer_type=spammer_type,
        csv_file=csv_file,
        df=df,
        df_passage=df_passage,
        gt_df=gt_df,
        percents=percents,
        seeds=seeds,
        max_iter_pgem=max_iter_pgem,
        device=device,
        spammer_fn=add_random_spammer,
        spammer_kwargs_fn=lambda percent, sd: {"seed": sd},
        print_progress=False,
    )


def run_anti_spammers(df, df_passage, gt_df, percents, seeds, max_iter_pgem, device):
    spammer_type = "anti"
    csv_file = f"results/{spammer_type}/pgem.csv"

    run_spammer_experiment(
        spammer_type=spammer_type,
        csv_file=csv_file,
        df=df,
        df_passage=df_passage,
        gt_df=gt_df,
        percents=percents,
        seeds=seeds,
        max_iter_pgem=max_iter_pgem,
        device=device,
        spammer_fn=add_anti_personas,
        spammer_kwargs_fn=lambda percent, sd: {"seed": sd},
        print_progress=True,
    )


def run_left_spammers(df, df_passage, gt_df, percents, seeds, max_iter_pgem, device):
    spammer_type = "left"
    csv_file = f"results/{spammer_type}/pgem.csv"

    run_spammer_experiment(
        spammer_type=spammer_type,
        csv_file=csv_file,
        df=df,
        df_passage=df_passage,
        gt_df=gt_df,
        percents=percents,
        seeds=seeds,
        max_iter_pgem=max_iter_pgem,
        device=device,
        spammer_fn=add_position_biased_spammers,
        spammer_kwargs_fn=lambda percent, sd: {"position_bias": "left", "seed": sd},
        print_progress=True,
    )


def run_right_spammers(df, df_passage, gt_df, percents, seeds, max_iter_pgem, device):
    spammer_type = "right"
    csv_file = f"results/{spammer_type}/pgem.csv"

    run_spammer_experiment(
        spammer_type=spammer_type,
        csv_file=csv_file,
        df=df,
        df_passage=df_passage,
        gt_df=gt_df,
        percents=percents,
        seeds=seeds,
        max_iter_pgem=max_iter_pgem,
        device=device,
        spammer_fn=add_position_biased_spammers,
        spammer_kwargs_fn=lambda percent, sd: {"position_bias": "right", "seed": sd},
        print_progress=True,
    )


def run_equal_spammers(df, df_passage, gt_df, percents, seeds, max_iter_pgem, device):
    spammer_type = "equal"
    csv_file = f"results/{spammer_type}/pgem.csv"

    run_spammer_experiment(
        spammer_type=spammer_type,
        csv_file=csv_file,
        df=df,
        df_passage=df_passage,
        gt_df=gt_df,
        percents=percents,
        seeds=seeds,
        max_iter_pgem=max_iter_pgem,
        device=device,
        spammer_fn=add_equal_proportion_of_all_spammers,
        spammer_kwargs_fn=lambda percent, sd: {"seed": sd},
        print_progress=True,
    )


# -------------------------
# Main
# -------------------------
def main():
    df, df_passage = load_passage_data(DF_PATH, PASSAGE_PICKLE_PATH)
    gt_df = df_passage

    size = len(df_passage)
    print(size)
    classes = [0] * size

    # Keep the original variables available if later cells/scripts depend on them.
    _ = classes

    run_random_spammers(df, df_passage, gt_df, PERCENTS, SEEDS, MAX_ITER_PGEM, device)
    run_anti_spammers(df, df_passage, gt_df, PERCENTS, SEEDS, MAX_ITER_PGEM, device)
#     run_left_spammers(df, df_passage, gt_df, PERCENTS, SEEDS, MAX_ITER_PGEM, device)
#     run_right_spammers(df, df_passage, gt_df, PERCENTS, SEEDS, MAX_ITER_PGEM, device)
#     run_equal_spammers(df, df_passage, gt_df, PERCENTS, SEEDS, MAX_ITER_PGEM, device)


if __name__ == "__main__":
    main()
