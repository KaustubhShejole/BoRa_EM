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
    crowd_bt_dist,  # noqa: F401
    logistic_preference_dist,  # noqa: F401
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
    "NoisyBT_acc_mean", "NoisyBT_acc_std",
    "NoisyBT_wacc_mean", "NoisyBT_wacc_std",
    "NoisyBT_tau_mean", "NoisyBT_tau_std",
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


from collections import defaultdict
from pathlib import Path

def build_pc_dict_for_noisybt(df, df_faceage, worker_col="worker", left_col="left", right_col="right", label_col="label"):
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
    noisybt = None
    noisybt_scores = None
    r_est = None

    try:
        random_df, _ = generator_fn(df, percent, seed=seed, **generator_kwargs)
        random_df = random_df.rename(columns={'performer': 'worker'})
        PC_faceage, _performer_label_dict, _item_label_dict = build_pc_dict_for_noisybt(random_df, df_passage)
#         PC_faceage = df_to_pickle(random_df, df_passage)
        K = len(PC_faceage.keys())

        noisybt_test = opt_fair.NoisyBT_3_0(data=PC_faceage, penalty=0, device=device, random_seed=seed)
        noisybt_scores, noisybt_skills, noisybt_biases = noisybt_test.alternate_optim(size, K, lr_x=LR, lr_y=LR, tol=TOL, iters=MAX_ITER)
        

        r_est = to_numpy(noisybt_scores)

        if np.isnan(r_est).any():
            return None, None, None

        tau = safe_kendalltau(r_est, gt_scores)

        if tau < 0:
            r_est = -r_est

        acc = compute_acc(df_passage, r_est, device)
        wacc = compute_weighted_acc(df_passage, r_est, device)
        tau = safe_kendalltau(r_est, gt_scores)
        return acc, wacc, tau

    except Exception as e:
        print(f"NoisyBT failed due to {e}")
        return None, None, None

    finally:
        if PC_faceage is not None:
            del PC_faceage
        if random_df is not None:
            del random_df
        if noisybt is not None:
            del noisybt
        if noisybt_scores is not None:
            del noisybt_scores
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
        NoisyBT_accs: List[float] = []
        NoisyBT_waccs: List[float] = []
        NoisyBT_taus: List[float] = []

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

            NoisyBT_accs.append(acc)
            NoisyBT_waccs.append(wacc)
            NoisyBT_taus.append(tau)

        row = [
            percent,
            np.mean(NoisyBT_accs), np.std(NoisyBT_accs),
            np.mean(NoisyBT_waccs), np.std(NoisyBT_waccs),
            np.mean(NoisyBT_taus), np.std(NoisyBT_taus),
        ]

        with open(config.output_csv, mode="a", newline="") as f:
            csv.writer(f).writerow(row)

        print(
            f"NoisyBT | "
            f"Percent: {percent} | "
            f"Acc: {np.mean(NoisyBT_accs):.4f} ± {np.std(NoisyBT_accs):.4f} | "
            f"WAcc: {np.mean(NoisyBT_waccs):.4f} ± {np.std(NoisyBT_waccs):.4f} | "
            f"Tau: {np.mean(NoisyBT_taus):.4f} ± {np.std(NoisyBT_taus):.4f}"
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
            output_csv="results/random/noisybt.csv",
        ),
        ExperimentConfig(
            spammer_type="anti",
            generator_fn=add_anti_personas,
            generator_kwargs={},
            output_csv="results/anti/noisybt.csv",
        ),
        ExperimentConfig(
            spammer_type="left",
            generator_fn=add_position_biased_spammers,
            generator_kwargs={"position_bias": "left"},
            output_csv="results/left/noisybt.csv",
        ),
        ExperimentConfig(
            spammer_type="right",
            generator_fn=add_position_biased_spammers,
            generator_kwargs={"position_bias": "right"},
            output_csv="results/right/noisybt.csv",
        ),
        ExperimentConfig(
            spammer_type="equal",
            generator_fn=add_equal_proportion_of_all_spammers,
            generator_kwargs={},
            output_csv="results/equal/noisybt.csv",
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
