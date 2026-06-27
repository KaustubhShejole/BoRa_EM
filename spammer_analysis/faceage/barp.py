from __future__ import annotations

import csv
import gc
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# -------------------------
# Device setup
# -------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

from spammer_types import *  # noqa: F401,F403
from util import *  # noqa: F401,F403
import opt_fair
from distribution_utils import (
    comparisons_to_df,  # noqa: F401
    safe_kendalltau,
    to_numpy,
)
from metrics import compute_acc, compute_weighted_acc


# -------------------------
# Hyperparameters and constants
# -------------------------
LR = 0.001
MAX_ITER = 100000
TOL = 1e-6
SEEDS = range(20, 30)
PERCENTS = [10, 20, 40, 60, 80, 100, 150, 233, 400, 900]

DF_PATH = "../../real_data/faceage/data/crowd_labels.csv"
PASSAGE_PICKLE_PATH = "../../real_data/faceage/data/FaceAgeDF.pickle"

HEADER = [
    "percent",
    "BARP_acc_mean", "BARP_acc_std",
    "BARP_wacc_mean", "BARP_wacc_std",
    "BARP_tau_mean", "BARP_tau_std",
]


# -------------------------
# Data loading helpers
# -------------------------
def sort_df(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Sort a dataframe by a specific column in ascending order."""
    return df.sort_values(by=column_name, ascending=True)


def load_passage_data() -> Tuple[pd.DataFrame, pd.DataFrame, int, np.ndarray]:
    """Load the passage dataset and its ground-truth scores."""
    df = pd.read_csv(DF_PATH)
    df = sort_df(df, "performer")

    with open(PASSAGE_PICKLE_PATH, "rb") as handle:
        df_passage = pickle.load(handle)

    size = len(df_passage)
    gt_df = df_passage
    gt_scores = gt_df["score"].to_numpy()
    print(size)
    return df, df_passage, size, gt_scores


# -------------------------
# Experiment machinery
# -------------------------
@dataclass
class ExperimentConfig:
    spammer_type: str
    generator_fn: Callable
    generator_kwargs: Dict
    output_csv: str


def ensure_output_dir(spammer_type: str) -> str:
    out_dir = f"results/{spammer_type}"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def write_csv_header(csv_file: str) -> None:
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)


def run_single_trial(
    *,
    df: pd.DataFrame,
    df_passage,
    gt_scores: np.ndarray,
    size: int,
    device: torch.device,
    percent: int,
    seed: int,
    generator_fn: Callable,
    generator_kwargs: Dict,
) -> Tuple[float | None, float | None, float | None]:
    """Run one seed trial and return (acc, wacc, tau)."""
    random_df = None
    PC_faceage = None
    barp = None
    barp_scores = None
    r_est = None

    try:
        random_df, _ = generator_fn(df, percent, seed=seed, **generator_kwargs)
        PC_faceage = df_to_pickle_faceage(random_df, df_passage)
        K = len(PC_faceage.keys())

        classes = df_passage['gender']
        FaceAge = opt_fair.BARP(data = PC_faceage, 
                                penalty = 0, 
                                classes = classes, 
                                device=device)
        barp_scores, annot_bias =  opt_fair._alternate_optim_torch(size, K, FaceAge, lr_x=LR, lr_y=LR, iters = MAX_ITER, tol=TOL)
        
        r_est = to_numpy(barp_scores)

        if np.isnan(r_est).any():
            return None, None, None

        # no inversion check needed because BARP does not have the
        # problem of non-identifiability with negative rewards

        acc = compute_acc(df_passage, r_est, device)
        wacc = compute_weighted_acc(df_passage, r_est, device)
        tau = safe_kendalltau(r_est, gt_scores)
        return acc, wacc, tau

    except Exception as e:
        print(f"BARP failed due to {e}")
        return None, None, None

    finally:
        if PC_faceage is not None:
            del PC_faceage
        if random_df is not None:
            del random_df
        if barp is not None:
            del barp
        if barp_scores is not None:
            del barp_scores
        if r_est is not None:
            del r_est

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_experiment(
    *,
    df: pd.DataFrame,
    df_passage,
    gt_scores: np.ndarray,
    size: int,
    device: torch.device,
    config: ExperimentConfig,
    percents: List[int],
    seeds=SEEDS,
) -> None:
    """Run the full experiment for one spammer type."""
    ensure_output_dir(config.spammer_type)
    write_csv_header(config.output_csv)

    for percent in percents:
        BARP_accs: List[float] = []
        BARP_waccs: List[float] = []
        BARP_taus: List[float] = []

        for sd in seeds:
            acc, wacc, tau = run_single_trial(
                df=df,
                df_passage=df_passage,
                gt_scores=gt_scores,
                size=size,
                device=device,
                percent=percent,
                seed=sd,
                generator_fn=config.generator_fn,
                generator_kwargs=config.generator_kwargs,
            )

            if acc is None or wacc is None or tau is None:
                continue

            BARP_accs.append(acc)
            BARP_waccs.append(wacc)
            BARP_taus.append(tau)

        row = [
            percent,
            np.mean(BARP_accs), np.std(BARP_accs),
            np.mean(BARP_waccs), np.std(BARP_waccs),
            np.mean(BARP_taus), np.std(BARP_taus),
        ]

        with open(config.output_csv, mode="a", newline="") as f:
            csv.writer(f).writerow(row)

        print(
            f"BARP | "
            f"Percent: {percent} | "
            f"Acc: {np.mean(BARP_accs):.4f} ± {np.std(BARP_accs):.4f} | "
            f"WAcc: {np.mean(BARP_waccs):.4f} ± {np.std(BARP_waccs):.4f} | "
            f"Tau: {np.mean(BARP_taus):.4f} ± {np.std(BARP_taus):.4f}"
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -------------------------
# Main entry point
# -------------------------
def main() -> None:
    df, df_passage, size, gt_scores = load_passage_data()

    configs = [
        ExperimentConfig(
            spammer_type="random",
            generator_fn=add_random_spammer,
            generator_kwargs={},
            output_csv="results/random/barp.csv",
        ),
        ExperimentConfig(
            spammer_type="anti",
            generator_fn=add_anti_personas,
            generator_kwargs={},
            output_csv="results/anti/barp.csv",
        ),
        ExperimentConfig(
            spammer_type="left",
            generator_fn=add_position_biased_spammers,
            generator_kwargs={"position_bias": "left"},
            output_csv="results/left/barp.csv",
        ),
        ExperimentConfig(
            spammer_type="right",
            generator_fn=add_position_biased_spammers,
            generator_kwargs={"position_bias": "right"},
            output_csv="results/right/barp.csv",
        ),
        ExperimentConfig(
            spammer_type="equal",
            generator_fn=add_equal_proportion_of_all_spammers,
            generator_kwargs={},
            output_csv="results/equal/barp.csv",
        ),
    ]

    for config in configs:
        print(f"\n### Addition of {config.spammer_type.title()} Spammers ###")
        run_experiment(
            df=df,
            df_passage=df_passage,
            gt_scores=gt_scores,
            size=size,
            device=device,
            config=config,
            percents=PERCENTS,
            seeds=SEEDS,
        )


if __name__ == "__main__":
    main()
