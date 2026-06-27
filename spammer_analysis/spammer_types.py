'''
Four main types of spammers:
    1. 'random': random guessors
    2. 'anti': opposite personas
    3. 'left': left position biased spammers
    4. 'right': right position biased spammers

Fifth types:
    5. 'equal': having equal proportion of these four spammers
'''

'''
1. 'random'
        function: add_random_spammer(df, percent, seed=42)
        returns: pd.DataFrame: Original + spammer-augmented DataFrame and spammer ids

2. 'anti'
        function: add_anti_personas(df, percent, seed=42)
        returns: pd.DataFrame: Original + spammer-augmented DataFrame and spammer ids
3. 'left'
        function: add_position_biased_spammers(df, percent, position_bias, seed=42)
        position_bias = "left"
        returns: pd.DataFrame: Original + spammer-augmented DataFrame and spammer ids
4. 'right'
        function: add_position_biased_spammers(df, percent, position_bias, seed=42)
        position_bias = "right"
        returns: pd.DataFrame: Original + spammer-augmented DataFrame and spammer ids
5. 'equal'
        function: add_equal_proportion_of_all_spammers(df, percent, seed, proportions=None)
        returns: pd.DataFrame: Original + spammer-augmented DataFrame and spammer ids
'''






import pandas as pd
import numpy as np
import random

def add_random_spammer(df, percent, seed=42):
    """
    Adds spammer annotators who randomly prefer left or right item.
    The 'label' is the actual item ID, and behavior is reproducible using a seed.
    
    Args:
        df (pd.DataFrame): Must contain ['performer', 'left', 'right', 'label']
        percent (float): Fraction of spammer annotators to add
        seed (int): Seed for full reproducibility
    
    Returns:
        pd.DataFrame: Original + spammer-augmented DataFrame
    """
    df = df.copy()

    # Fix seeds
    random.seed(seed)
    np.random.seed(seed)  # just in case NumPy is used elsewhere
    percent = percent/100
    original_annotators = df['performer'].unique()
    num_annotators = len(original_annotators)
    num_spammers = int(percent * num_annotators)
    max_performer_id = df['performer'].max()
    comparisons_per_annotator = len(df) // num_annotators

    new_rows = []
    spammer_ids = []
    for i in range(num_spammers):
        spammer_id = max_performer_id + i + 1
        spammer_ids.append(spammer_id)

        # Sample with fixed but varying seed for diversity + reproducibility
        sampled = df.sample(
            n=comparisons_per_annotator,
            replace=True,
            random_state=seed + i  # ensures different samples per spammer
        )

        for _, row in sampled.iterrows():
            preferred_item = random.choice([row['left'], row['right']])
            new_rows.append({
                'performer': spammer_id,
                'left': row['left'],
                'right': row['right'],
                'label': preferred_item
            })

    spammer_df = pd.DataFrame(new_rows)
    return pd.concat([df, spammer_df], ignore_index=True), spammer_ids

def add_anti_personas(df, percent, seed=42):
    """
    Adds spammer annotators who randomly prefer left or right item.
    The 'label' is the actual item ID, and behavior is reproducible using a seed.
    
    Args:
        df (pd.DataFrame): Must contain ['performer', 'left', 'right', 'label']
        percent (float): Fraction of spammer annotators to add
        seed (int): Seed for full reproducibility
    
    Returns:
        pd.DataFrame: Original + spammer-augmented DataFrame
    """
    df = df.copy()

    # Fix seeds
    random.seed(seed)
    np.random.seed(seed)  # just in case NumPy is used elsewhere
    percent = percent/100
    original_annotators = df['performer'].unique()
    num_annotators = len(original_annotators)
    num_spammers = int(percent * num_annotators)
    max_performer_id = df['performer'].max()
    comparisons_per_annotator = len(df) // num_annotators

    new_rows = []
    spammer_ids = []
    for i in range(num_spammers):
        spammer_id = max_performer_id + i + 1
        spammer_ids.append(spammer_id)

        # Sample with fixed but varying seed for diversity + reproducibility
        sampled = df.sample(
            n=comparisons_per_annotator,
            replace=True,
            random_state=seed + i  # ensures different samples per spammer
        )

        for _, row in sampled.iterrows():
            if row['label'] == row['right']:
                preferred_item = row['left']
            else:
                preferred_item = row['right']
            new_rows.append({
                'performer': spammer_id,
                'left': row['left'],
                'right': row['right'],
                'label': preferred_item
            })

    spammer_df = pd.DataFrame(new_rows)
    return (
        pd.concat([df, spammer_df], ignore_index=True)
          .sample(frac=1, random_state=seed)
          .reset_index(drop=True),
        spammer_ids
    )
#     return pd.concat([df, spammer_df], ignore_index=True), spammer_ids

def add_position_biased_spammers(df, percent, position_bias, seed=42):
    df = df.copy()

    # Fix seeds
    random.seed(seed)
    np.random.seed(seed)
    percent = percent/100
    original_annotators = df['performer'].unique()
    num_annotators = len(original_annotators)
    num_spammers = int(percent * num_annotators)
    max_performer_id = df['performer'].max()
    comparisons_per_annotator = len(df) // num_annotators

    new_rows = []
    spammer_ids = []

    for i in range(num_spammers):
        spammer_id = max_performer_id + i + 1
        spammer_ids.append(spammer_id)

        sampled = df.sample(
            n=comparisons_per_annotator,
            replace=True,
            random_state=seed + i
        )

        for _, row in sampled.iterrows():
            if position_bias == 'left':
                new_rows.append({
                    'performer': spammer_id,
                    'left': row['left'],
                    'right': row['right'],
                    'label': row['left']
                })
            else:
                new_rows.append({
                    'performer': spammer_id,
                    'left': row['left'],
                    'right': row['right'],
                    'label': row['right']
                })

    spammer_df = pd.DataFrame(new_rows)
    return pd.concat([df, spammer_df], ignore_index=True), spammer_ids


import pandas as pd
import numpy as np
import random
from scipy.special import expit as sigmoid  # logistic function

# def add_spammers_sample_through_competence(df, percent, data_name, seed=42):
#     """
#     Adds spammers to the dataset by sampling existing comparisons but 
#     flipping labels according to a competence parameter (beta).
    
#     Args:
#         df (pd.DataFrame): Original dataframe with columns ['performer', 'left', 'right', 'label'].
#         percent (float): Fraction of annotators to replace with spammers (0–1).
#         data_name (str): Which dataset, e.g., 'passage'.
#         seed (int): Random seed for reproducibility.

#     Returns:
#         pd.DataFrame: Original + spammer-augmented dataframe.
#         list: IDs of the new spammer annotators.
#     """
#     df = df.copy()

#     # Fix seeds
#     random.seed(seed)
#     np.random.seed(seed)

#     # Load ground truth scores for the dataset
#     if data_name == 'passage':
#         gt_df = pd.read_csv('../data/gt_df_passage.csv')
#         r = dict(zip(gt_df['label'], gt_df['score']))
#     else:
#         raise ValueError(f"Unknown dataset {data_name}")

#     # Get annotator information
#     original_annotators = df['performer'].unique()
#     num_annotators = len(original_annotators)
#     num_spammers = int(percent * num_annotators)
#     max_performer_id = df['performer'].max()
#     comparisons_per_annotator = len(df) // num_annotators

#     new_rows = []
#     spammer_ids = []

#     # Generate competence levels (betas) between -1 and 0
#     betas = np.random.uniform(-1, 0, size=num_spammers)

#     for i in range(num_spammers):
#         spammer_id = max_performer_id + i + 1
#         spammer_ids.append(spammer_id)

#         # Sample tasks from dataset for this spammer
#         sampled = df.sample(
#             n=comparisons_per_annotator,
#             replace=True,
#             random_state=seed + i
#         )

#         for _, row in sampled.iterrows():
#             left_id = row['left']
#             right_id = row['right']
#             r_left = r[left_id]
#             r_right = r[right_id]

#             # Probability that spammer says "left > right"
#             p_left = sigmoid(betas[i] * (r_left - r_right))

#             label = left_id if np.random.rand() < p_left else right_id  # 1: left wins, 0: right wins

#             new_rows.append({
#                 'performer': spammer_id,
#                 'left': left_id,
#                 'right': right_id,
#                 'label': label
#             })

#     spammer_df = pd.DataFrame(new_rows)
#     return pd.concat([df, spammer_df], ignore_index=True), spammer_ids


import pandas as pd
import numpy as np
import random
from typing import Tuple, List, Dict, Optional

def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s)

def _next_performer_ids(start: int, count: int) -> List[int]:
    return list(range(start + 1, start + 1 + count))

def add_equal_proportion_of_all_spammers(
    df: pd.DataFrame,
    percent: float,
    seed: int = 42,
    proportions: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, List]:
    """
    Add four types of spammers to the dataset, allocating the total number of
    spammers based on the ORIGINAL annotator count (not progressively).
    
    Types: 'random', 'anti', 'left', 'right'
    
    Args:
        df: DataFrame with ['performer', 'left', 'right', 'label']
        percent: fraction of ORIGINAL annotators to create as spammers (e.g. 0.2)
        seed: random seed
        proportions: optional dict with fractions for each type summing to 1.0
                     keys: 'random', 'anti', 'left', 'right'
                     if None, they are equally split.
    Returns:
        (augmented_df, spammer_ids_list)
    """
    base_df = df.copy()
    random.seed(seed)
    np.random.seed(seed)

    original_annotators = base_df['performer'].unique()
    num_annotators = len(original_annotators)
    if num_annotators == 0:
        raise ValueError("DataFrame has no performers")
    percent = percent/100
    # Determine total number of spammers (based on original annotators)
    total_spammers = int(round(percent * num_annotators))
    if total_spammers == 0:
        return df.copy(), []  # nothing to add

    # Default equal proportions if not provided
    types = ['random', 'anti', 'left', 'right']
    if proportions is None:
        proportions = {t: 1.0 / 4.0 for t in types}
    else:
        # validate proportions
        if not all(t in proportions for t in types):
            raise ValueError(f"proportions must contain keys: {types}")
        s = sum(proportions.values())
        if abs(s - 1.0) > 1e-6:
            # normalize if they don't sum exactly to 1
            proportions = {k: v / s for k, v in proportions.items()}

    # allocate integer counts to each type (ensure sum == total_spammers)
    raw_counts = {t: proportions[t] * total_spammers for t in types}
    int_counts = {t: int(np.floor(raw_counts[t])) for t in types}
    remainder = total_spammers - sum(int_counts.values())
    # distribute remainder by largest fractional parts
    frac_parts = sorted(types, key=lambda t: raw_counts[t] - int_counts[t], reverse=True)
    for i in range(remainder):
        int_counts[frac_parts[i]] += 1

    comparisons_per_annotator = len(base_df) // num_annotators
    # If comparisons_per_annotator == 0 (very small dataset), fallback to 1
    if comparisons_per_annotator == 0:
        comparisons_per_annotator = 1

    # Prepare unique performer ids for new spammers.
    performer_series = base_df['performer']
    if _is_numeric_series(performer_series):
        max_id = int(performer_series.max())
        # ensure max_id is finite
        if pd.isna(max_id):
            start_id = 0
        else:
            start_id = max_id
        # create a pool of new numeric IDs
        new_id_pool = _next_performer_ids(start_id, total_spammers)
        use_numeric_ids = True
    else:
        # non-numeric performer ids -> generate string IDs
        new_id_pool = [f"spammer_{i+1}" for i in range(total_spammers)]
        use_numeric_ids = False

    # helper to sample comparisons and create rows for a spammer
    def _create_spammer_rows(spammer_id, sampled_rows, behavior: str, rng_seed: int):
        rng = random.Random(rng_seed)
        rows = []
        for _, r in sampled_rows.iterrows():
            left = r['left']
            right = r['right']
            if behavior == 'random':
                preferred = rng.choice([left, right])
            elif behavior == 'anti':
                # choose the opposite of the original label
                if r['label'] == right:
                    preferred = left
                else:
                    preferred = right
            elif behavior == 'left':
                preferred = left
            elif behavior == 'right':
                preferred = right
            else:
                raise ValueError("unknown behavior")
            rows.append({'performer': spammer_id, 'left': left, 'right': right, 'label': preferred})
        return rows

    new_rows = []
    spammer_ids = []
    pool_idx = 0
    # For reproducibility and to avoid overlapping samples across types,
    # always sample from base_df (not the progressively augmented df).
    for t in types:
        count = int_counts[t]
        for i in range(count):
            spammer_id = new_id_pool[pool_idx]
            pool_idx += 1
            spammer_ids.append(spammer_id)

            # Use a different random_state for each spammer to diversify samples
            sampled = base_df.sample(
                n=comparisons_per_annotator,
                replace=True,
                random_state=seed + hash((t, i)) % (2**31 - 1)
            )
            # consistent behavior RNG seed
            behavior_seed = seed + pool_idx
            new_rows.extend(_create_spammer_rows(spammer_id, sampled, behavior=t, rng_seed=behavior_seed))

    spammer_df = pd.DataFrame(new_rows)
    augmented = pd.concat([df, spammer_df], ignore_index=True)
    return augmented, spammer_ids


import random
import numpy as np
import pandas as pd

import random
import numpy as np
import pandas as pd

import random
import numpy as np
import pandas as pd
from pathlib import Path

def add_anti_personas2(df, percent, gt_df, seed=42):
    """
    Anti-rational spammers:
    - Sample any pair from gt_df
    - Always pick WORSE item using true scores
    """

    df = df.copy()

    random.seed(seed)
    np.random.seed(seed)

    percent /= 100

    # ✅ Normalize item IDs to filenames (IMPORTANT: must match df)
    gt_df = gt_df.copy()
    gt_df["item_id"] = gt_df["full_path"]

    # Mapping: item_id -> score
    item_to_score = dict(zip(gt_df["item_id"], gt_df["score"]))

    item_list = gt_df["item_id"].tolist()

    original_annotators = df['performer'].unique()
    num_annotators = len(original_annotators)
    num_spammers = int(percent * num_annotators)

    max_performer_id = df['performer'].max()
    comparisons_per_annotator = len(df) // num_annotators

    new_rows = []
    spammer_ids = []

    for i in range(num_spammers):
        spammer_id = max_performer_id + i + 1
        spammer_ids.append(spammer_id)

        rng = np.random.RandomState(seed + i)

        for _ in range(comparisons_per_annotator):
            left, right = rng.choice(item_list, size=2, replace=False)

            # Compare using true scores
            if item_to_score[left] > item_to_score[right]:
                wrong_winner = right
            else:
                wrong_winner = left

            new_rows.append({
                'performer': spammer_id,
                'left': left,
                'right': right,
                'label': wrong_winner
            })

    spammer_df = pd.DataFrame(new_rows)

    return (
        pd.concat([df, spammer_df], ignore_index=True)
          .sample(frac=1, random_state=seed)
          .reset_index(drop=True),
        spammer_ids
    )