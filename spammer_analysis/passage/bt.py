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

DF_PATH = "../../real_data/passage/data/passage_cleaned.csv"
PASSAGE_PICKLE_PATH = "../../real_data/passage/data/PassageDF.pickle"

HEADER = [
    "percent",
    "BT_acc_mean", "BT_acc_std",
    "BT_wacc_mean", "BT_wacc_std",
    "BT_tau_mean", "BT_tau_std",
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
    bt = None
    bt_scores = None
    r_est = None

    try:
        random_df, _ = generator_fn(df, percent, seed=seed, **generator_kwargs)
        PC_faceage = df_to_pickle(random_df, df_passage)
        K = len(PC_faceage.keys())
        all_pc_faceage  = opt_fair._pc_without_reviewers(PC_faceage)

        bt_scores = opt_fair.opt_pairwise_gpu(
            size, 
            all_pc_faceage, 
            alpha=0, 
            initial_params=None, 
            max_iter=MAX_ITER, 
            tol=TOL,
        )

        r_est = to_numpy(bt_scores)

        if np.isnan(r_est).any():
            return None, None, None

        # no inversion check needed because BT does not have the
        # problem of non-identifiability with negative rewards

        acc = compute_acc(df_passage, r_est, device)
        wacc = compute_weighted_acc(df_passage, r_est, device)
        tau = safe_kendalltau(r_est, gt_scores)
        return acc, wacc, tau

    except Exception as e:
        print(f"BT failed due to {e}")
        return None, None, None

    finally:
        if PC_faceage is not None:
            del PC_faceage
        if random_df is not None:
            del random_df
        if bt is not None:
            del bt
        if bt_scores is not None:
            del bt_scores
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
        BT_accs: List[float] = []
        BT_waccs: List[float] = []
        BT_taus: List[float] = []

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

            BT_accs.append(acc)
            BT_waccs.append(wacc)
            BT_taus.append(tau)

        row = [
            percent,
            np.mean(BT_accs), np.std(BT_accs),
            np.mean(BT_waccs), np.std(BT_waccs),
            np.mean(BT_taus), np.std(BT_taus),
        ]

        with open(config.output_csv, mode="a", newline="") as f:
            csv.writer(f).writerow(row)

        print(
            f"BT | "
            f"Percent: {percent} | "
            f"Acc: {np.mean(BT_accs):.4f} ± {np.std(BT_accs):.4f} | "
            f"WAcc: {np.mean(BT_waccs):.4f} ± {np.std(BT_waccs):.4f} | "
            f"Tau: {np.mean(BT_taus):.4f} ± {np.std(BT_taus):.4f}"
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
            output_csv="results/random/bt.csv",
        ),
        ExperimentConfig(
            spammer_type="anti",
            generator_fn=add_anti_personas,
            generator_kwargs={},
            output_csv="results/anti/bt.csv",
        ),
        ExperimentConfig(
            spammer_type="left",
            generator_fn=add_position_biased_spammers,
            generator_kwargs={"position_bias": "left"},
            output_csv="results/left/bt.csv",
        ),
        ExperimentConfig(
            spammer_type="right",
            generator_fn=add_position_biased_spammers,
            generator_kwargs={"position_bias": "right"},
            output_csv="results/right/bt.csv",
        ),
        ExperimentConfig(
            spammer_type="equal",
            generator_fn=add_equal_proportion_of_all_spammers,
            generator_kwargs={},
            output_csv="results/equal/bt.csv",
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
