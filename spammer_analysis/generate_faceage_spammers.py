#!/usr/bin/env python3
"""
Generate FaceAge dataset with injected spammers.

This script modifies the FaceAge dataset by injecting different types of spammers
(random, anti-personas, or position-biased) and saves the resulting pairwise
comparisons for later use.
"""

import argparse
import os
import pickle
from collections import defaultdict
import torch
import pandas as pd

# Local imports
from spammer_types import (
    add_random_spammer,
    add_anti_personas,
    add_position_biased_spammers,
    add_equal_proportion_of_all_spammers
)


def build_spammer_dataset(spammer_type: str, spammer_percent: float) -> None:
    """Inject spammers into FaceAge dataset and save the result."""

    # Load datasets
    with open("../data/FaceAgeDF1.pickle", "rb") as handle:
        df_faceage = pickle.load(handle)

    df = pd.read_csv("../age_dataset/crowd_labels.csv")
    print("Crowd labels sample:")
    print(df.tail())

    # Add spammers
    if spammer_type == "random":
        df, spammer_ids = add_random_spammer(df, spammer_percent / 100)
    elif spammer_type == "anti":
        df, spammer_ids = add_anti_personas(df, spammer_percent / 100)
    elif spammer_type == "left":
        df, spammer_ids = add_position_biased_spammers(df, spammer_percent / 100, "left")
    elif spammer_type == 'right':
        df, spammer_ids = add_position_biased_spammers(df, spammer_percent / 100, "right")
    else:
        df, spammer_ids = add_equal_proportion_of_all_spammers(df, spammer_percent / 100)
    
    import os

    # Ensure the output directory exists
    output_dir = "spammer_data"
    os.makedirs(output_dir, exist_ok=True)

    # Construct the filename
    filename = f"df_FaceAge_{spammer_type}_{int(spammer_percent)}.csv"
    filepath = os.path.join(output_dir, filename)

    df.rename(columns={"performer": "worker"}, inplace=True)

    # Save the DataFrame
    df.to_csv(filepath, index=False)

    print(f"Saved spammer dataset to: {filepath}")
    
    # Map performers to integer IDs
    unique_performers = list(df["worker"].unique())
    print(f"Unique performers: {len(unique_performers)}")

    performer_label_dict = {performer: i for i, performer in enumerate(unique_performers)}

    # Map items to integer IDs
    item_labels = list(df_faceage["full_path"])
    item_label_dict = {item: i for i, item in enumerate(item_labels)}

    # Build pairwise comparisons per performer
    PC_faceage_spm = defaultdict(list)
    total_pairs = 0

    for performer, group in df.groupby("worker"):
        key = performer_label_dict[performer]

        for _, row in group.iterrows():
            total_pairs += 1

            left = row["left"].split("/")[-1]
            right = row["right"].split("/")[-1]
            winner = row["label"].split("/")[-1]  # whichever side was chosen

            # Map to indices
            left_label = item_label_dict[left]
            right_label = item_label_dict[right]
            winner_label = item_label_dict[winner]

            # Ensure pair stored as (winner, loser)
            if winner_label == left_label:
                PC_faceage_spm[key].append((left_label, right_label))
            else:
                PC_faceage_spm[key].append((right_label, left_label))

    print(f"Total pairwise comparisons: {total_pairs}")

    # Save output
    os.makedirs("spammer_data", exist_ok=True)
    out_file = f"spammer_data/FaceAge_{spammer_type}_{int(spammer_percent)}.pickle"

    with open(out_file, "wb") as handle:
        pickle.dump(PC_faceage_spm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved modified dataset to {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Inject spammers into the FaceAge dataset."
    )
    parser.add_argument(
        "--spammer_type",
        type=str,
        default="random",
        choices=["random", "anti", "left", "right", "combine"],
        help="Type of spammers to add.",
    )
    parser.add_argument(
        "--spammer_percent",
        type=float,
        default=10,
        help="Percentage of spammers to add (0-100).",
    )

    args = parser.parse_args()
    build_spammer_dataset(args.spammer_type, args.spammer_percent)


if __name__ == "__main__":
    main()
