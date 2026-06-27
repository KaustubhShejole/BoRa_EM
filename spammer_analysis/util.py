from collections import *
from pathlib import Path

def df_to_pickle(df, df_faceage):
    df.rename(columns={"performer": "worker"}, inplace=True)
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
    return PC_faceage_spm

def df_to_pickle_faceage(df, df_faceage):
    df.rename(columns={"performer": "worker"}, inplace=True)
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

            left = Path(row["left"]).name
            right = Path(row["right"]).name
            winner = Path(row["label"]).name  # whichever side was chosen

            # Map to indices
            left_label = item_label_dict[left]
            right_label = item_label_dict[right]
            winner_label = item_label_dict[winner]

            # Ensure pair stored as (winner, loser)
            if winner_label == left_label:
                PC_faceage_spm[key].append((left_label, right_label))
            else:
                PC_faceage_spm[key].append((right_label, left_label))
    return PC_faceage_spm