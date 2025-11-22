
# model/dataset.py

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


DNA_BASES = ["A", "C", "G", "T"]
AMBIGUOUS_BASE = "N"


def random_base_mask(seq: str, mask_prob: float = 0.05) -> str:
    """
    Randomly replace some bases with 'N' to simulate sequencing noise or uncertainty.
    mask_prob: probability per base to be masked.
    """
    seq_chars = list(seq)
    for i, base in enumerate(seq_chars):
        if base in DNA_BASES and random.random() < mask_prob:
            seq_chars[i] = AMBIGUOUS_BASE
    return "".join(seq_chars)


def fragment_length_jitter(
    seq: str,
    min_delta: int = -20,
    max_delta: int = 20,
    min_length: int = 50,
) -> str:
    """
    Randomly shorten or slightly extend the fragment length.

    Negative delta -> crop sequence
    Positive delta -> small random duplication padding at ends (very naive)

    This is a simple simulation just to introduce length variability.
    """
    if len(seq) <= min_length:
        return seq  # don't touch very short fragments

    delta = random.randint(min_delta, max_delta)
    new_len = len(seq) + delta

    # enforce minimum and maximum
    new_len = max(min_length, new_len)
    new_len = min(len(seq), new_len)  # we won't actually extend beyond original in this simple version

    # simple strategy: crop from both ends
    if new_len < len(seq):
        # amount to remove
        total_crop = len(seq) - new_len
        left_crop = total_crop // 2
        right_crop = total_crop - left_crop
        return seq[left_crop: len(seq) - right_crop]

    return seq


@dataclass
class DNADatasetConfig:
    csv_path: str
    max_length: int = 256
    augment: bool = False
    mask_prob: float = 0.05
    jitter_min_delta: int = -20
    jitter_max_delta: int = 20
    jitter_min_length: int = 50
    sequence_column: str = "sequence"
    label_column: str = "label"


class DNADataset(Dataset):
    """
    PyTorch Dataset for DNA fragments and class labels.
    Loads sequences from a CSV file and encodes them using the provided tokenizer.
    """

    def __init__(
        self,
            config: DNADatasetConfig,
            tokenizer: Any,
        ) -> None:
            """
            tokenizer: object with .encode(text, max_length) -> List[int] and .pad_token_id
            """
            self.config = config
            self.tokenizer = tokenizer

            df = pd.read_csv(config.csv_path)
            # Store sequences and labels in memory for simplicity
            self.sequences: List[str] = df[config.sequence_column].astype(str).tolist()
            self.labels: List[int] = df[config.label_column].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.sequences)

    def _apply_augmentations(self, seq: str) -> str:
        """
        Apply training-time augmentations to DNA sequence.
        """
        # fragment length jitter
        seq = fragment_length_jitter(
            seq,
            min_delta=self.config.jitter_min_delta,
            max_delta=self.config.jitter_max_delta,
            min_length=self.config.jitter_min_length,
        )
        # random base mask
        seq = random_base_mask(seq, mask_prob=self.config.mask_prob)
        return seq

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        label = self.labels[idx]

        if self.config.augment:
            seq = self._apply_augmentations(seq)

        # tokenization
        input_ids_list: List[int] = self.tokenizer.encode(
            seq,
            max_length=self.config.max_length,
        )

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        # attention_mask is created in collate function based on padding

        return {
            "input_ids": input_ids,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def dna_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Collate function to:
    - pad input_ids to same length in batch
    - create attention_mask
    - stack labels
    """
    input_ids_list = [item["input_ids"] for item in batch]
    labels_list = [item["labels"] for item in batch]

    # pad_sequence expects shape [seq_len] tensors, returns [batch, max_len]
    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)

    attention_mask = (padded_input_ids != pad_token_id).long()
    labels = torch.stack(labels_list, dim=0)

    return {
        "input_ids": padded_input_ids,      # [batch_size, max_seq_len]
        "attention_mask": attention_mask,   # [batch_size, max_seq_len]
        "labels": labels,                   # [batch_size]
    }
