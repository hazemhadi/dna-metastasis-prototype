
#!/usr/bin/env python3
"""
Synthetic cfDNA fragment generator for metastasis prototype.

Generates:
- Healthy vs metastatic-like fragments
- Fragment length differences
- GC-content shifts
- Motif frequency shifts
- Simple hotspot mutations

Outputs a CSV with:
id, split, label, fragment_length, gc_content, sequence
"""

import argparse
import csv
import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np


HEALTHY_LABEL = 0
METASTATIC_LABEL = 1


def clamp(value: int, min_val: int, max_val: int) -> int:
    return max(min_val, min(max_val, value))


def sample_fragment_length(label: int) -> int:
    """
    Healthy: slightly longer, narrower distribution
    Metastatic: slightly shorter, a bit more variable
    """
    if label == HEALTHY_LABEL:
        length = int(np.random.normal(loc=170, scale=20))
        return clamp(length, 130, 260)
    else:
        length = int(np.random.normal(loc=150, scale=25))
        return clamp(length, 120, 240)


def base_probs_for_label(label: int) -> Dict[str, float]:
    """
    Define label-specific base distributions to encode GC-content differences.
    Healthy: GC ~ 45%
    Metastatic: GC ~ 55% (or vice versa, main point is a shift)
    """
    if label == HEALTHY_LABEL:
        return {"A": 0.275, "C": 0.225, "G": 0.225, "T": 0.275}
    else:
        return {"A": 0.20, "C": 0.30, "G": 0.25, "T": 0.25}


def multinomial_sample(bases: List[str], probs: List[float]) -> str:
    return np.random.choice(bases, p=probs)


def insert_motif(sequence: List[str], motif: str, num_insertions: int) -> None:
    """
    In-place insertion of motif occurrences at random positions.
    Will overwrite existing bases (not true insertion to keep length fixed).
    """
    if len(sequence) < len(motif):
        return

    for _ in range(num_insertions):
        start_idx = random.randint(0, len(sequence) - len(motif))
        for i, base in enumerate(motif):
            sequence[start_idx + i] = base


def apply_hotspot_mutations(sequence: List[str], label: int) -> None:
    """
    Simulate hotspot mutations: specific relative positions are more likely mutated
    in metastatic fragments.

    This is deliberately simple:
    - Choose a few positions near the middle and ends.
    - In metastatic fragments, mutate them to G/C with higher probability.
    """
    if len(sequence) < 20:
        return

    length = len(sequence)
    hotspot_positions = [
        length // 2,              # center
        length // 2 - 5,
        length // 2 + 5,
        5,                        # near 5' end
        length - 6                # near 3' end
    ]

    for pos in hotspot_positions:
        if label == METASTATIC_LABEL:
            # higher chance of mutation
            if random.random() < 0.5:
                sequence[pos] = random.choice(["G", "C"])
        else:
            # lower chance, more subtle
            if random.random() < 0.1:
                sequence[pos] = random.choice(["A", "T"])


def generate_fragment(label: int) -> Tuple[str, int, float]:
    """
    Generate a single fragment sequence for a given label.

    Returns:
        sequence (str), length (int), gc_content (float)
    """
    length = sample_fragment_length(label)
    probs = base_probs_for_label(label)
    bases = list(probs.keys())
    probs_list = [probs[b] for b in bases]

    # Initial base-level sampling
    seq_list = [multinomial_sample(bases, probs_list) for _ in range(length)]

    # Motif frequency differences:
    # - Healthy: more AT-rich motifs
    # - Metastatic: more CG/CCG motifs
    if label == HEALTHY_LABEL:
        insert_motif(seq_list, "ATA", num_insertions=max(1, length // 80))
    else:
        insert_motif(seq_list, "CCG", num_insertions=max(1, length // 60))
        insert_motif(seq_list, "CG", num_insertions=max(1, length // 80))

    # Hotspot mutations
    apply_hotspot_mutations(seq_list, label)

    # Compute GC-content
    seq_str = "".join(seq_list)
    gc_count = seq_str.count("G") + seq_str.count("C")
    gc_content = gc_count / float(length)

    return seq_str, length, gc_content


def generate_dataset(
    n_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    metastatic_fraction: float,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Generate a full dataset as a list of dicts.

    Each entry:
        id, split, label, fragment_length, gc_content, sequence
    """
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6), \
        "Train/val/test ratios must sum to 1.0"

    np.random.seed(seed)
    random.seed(seed)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val

    splits = (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * n_test
    )
    random.shuffle(splits)

    dataset: List[Dict[str, str]] = []

    for idx, split in enumerate(splits):
        # Decide label
        if random.random() < metastatic_fraction:
            label = METASTATIC_LABEL
        else:
            label = HEALTHY_LABEL

        seq, length, gc_content = generate_fragment(label)
        row = {
            "id": f"frag_{idx:06d}",
            "split": split,
            "label": str(label),
            "fragment_length": str(length),
            "gc_content": f"{gc_content:.4f}",
            "sequence": seq,
        }
        dataset.append(row)

    return dataset


def write_csv(dataset: List[Dict[str, str]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not dataset:
        raise ValueError("Dataset is empty, nothing to write.")

    fieldnames = list(dataset[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic cfDNA fragments for metastasis classification."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Total number of fragments to generate (train+val+test).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of samples in the training split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of samples in the validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of samples in the test split.",
    )
    parser.add_argument(
        "--metastatic-fraction",
        type=float,
        default=0.5,
        help="Approximate fraction of metastatic-like fragments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/synthetic_fragments.csv",
        help="Path to output CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[*] Generating synthetic dataset...")
    dataset = generate_dataset(
        n_samples=args.n_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        metastatic_fraction=args.metastatic_fraction,
        seed=args.seed,
    )
    print(f"[*] Generated {len(dataset)} fragments.")

    print(f"[*] Writing CSV to {args.output_path} ...")
    write_csv(dataset, args.output_path)
    print("[*] Done.")


if __name__ == "__main__":
    main()
