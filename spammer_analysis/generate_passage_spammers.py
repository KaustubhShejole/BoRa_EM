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
    add_equal_proportion_of_all_spammers,
    add_spammers_sample_through_competence
)


def build_spammer_dataset(spammer_type: str, spammer_percent: float, data_name=None) -> None:
    """Inject spammers into Passage dataset and save the result."""

    # Load datasets
    with open("../data/PassageDF1.pickle", "rb") as handle:
        df_faceage = pickle.load(handle)

    import pandas as pd
    df = pd.read_csv('../data/passage/passage_cleaned.csv')
    
    
    def sort_df(df, column_name):
        # Sort by a specific column (replace 'column_name' with your column)
        df_sorted = df.sort_values(by=column_name, ascending=True)  # or ascending=False

        return df_sorted
    df = sort_df(df, 'performer')

    # Add spammers
    if spammer_type == "random":
        df, spammer_ids = add_random_spammer(df, spammer_percent / 100)
    elif spammer_type == "anti":
        df, spammer_ids = add_anti_personas(df, spammer_percent / 100)
    elif spammer_type == "left":
        df, spammer_ids = add_position_biased_spammers(df, spammer_percent / 100, "left")
    elif spammer_type == 'right':
        df, spammer_ids = add_position_biased_spammers(df, spammer_percent / 100, "right")
    elif spammer_type == 'combine':
        df, spammer_ids = add_equal_proportion_of_all_spammers(df, spammer_percent / 100)
    else:
        df, spammer_ids = add_spammers_sample_through_competence(df, spammer_percent / 100, data_name)
    
    import os

    # Ensure the output directory exists
    output_dir = "spammer_data"
    os.makedirs(output_dir, exist_ok=True)

    # Construct the filename
    filename = f"df_Passage_{spammer_type}_{int(spammer_percent)}.csv"
    filepath = os.path.join(output_dir, filename)

    df.rename(columns={"performer": "worker"}, inplace=True)
#     print(df.head())
    # Save the DataFrame
    df.to_csv(filepath, index=False)

    print(f"Saved spammer dataset to: {filepath}")
    
    # Map performers to integer IDs
    unique_performers = list(df["worker"].unique())
    print(f"Unique performers: {len(unique_performers)}")

    performer_label_dict = {performer: i for i, performer in enumerate(unique_performers)}

    # Map items to integer IDs
    item_labels = list(df_faceage["label"])
    item_label_dict = {item: i for i, item in enumerate(item_labels)}

    # Build pairwise comparisons per performer
    PC_faceage_spm = defaultdict(list)
    total_pairs = 0

    for performer, group in df.groupby("worker"):
        key = performer_label_dict[performer]

        for _, row in group.iterrows():
            total_pairs += 1

            left = row["left"]
            right = row["right"]
            winner = row["label"]  # whichever side was chosen

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
    out_file = f"spammer_data/Passage_{spammer_type}_{int(spammer_percent)}.pickle"

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
        choices=["random", "anti", "left", "right", "combine", 'compe'],
        help="Type of spammers to add.",
    )
    parser.add_argument(
        "--spammer_percent",
        type=float,
        default=10,
        help="Percentage of spammers to add (0-100).",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default='age',
        help="dataset name",
    )

    args = parser.parse_args()
    build_spammer_dataset(args.spammer_type, args.spammer_percent, args.data_name)


if __name__ == "__main__":
    main()
