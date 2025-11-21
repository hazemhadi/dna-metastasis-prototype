
# model/dna_tokenizer.py
from typing import List, Dict, Tuple
import torch


class DNATokenizer:
    """
    Minimal and simple character-level tokenizer for DNA sequences.
    
    Vocab:
        0: [PAD]
        1: [BOS]
        2: [EOS]
        3: [MASK]
        4: A
        5: C
        6: G
        7: T
        8: N  (unknown/any base)
    """

    def __init__(self) -> None:
        # Define vocab
        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.mask_token = "[MASK]"

        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.mask_token]
        self.base_tokens = ["A", "C", "G", "T", "N"]

        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}

        # Build vocab
        all_tokens = self.special_tokens + self.base_tokens
        for idx, tok in enumerate(all_tokens):
            self.id_to_token[idx] = tok
            self.token_to_id[tok] = idx

        self.pad_token_id = self.token_to_id[self.pad_token]
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.mask_token_id = self.token_to_id[self.mask_token]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def _normalize_seq(self, seq: str) -> str:
        """Uppercase and keep only A/C/G/T/N (others → N)."""
        seq = seq.strip().upper()
        valid = {"A", "C", "G", "T", "N"}
        return "".join(ch if ch in valid else "N" for ch in seq)

    def encode(
        self,
        seq: str,
        max_length: int,
        add_special_tokens: bool = True,
    ) -> Tuple[List[int], List[int]]:
        """
        Encode one DNA sequence into token ids and attention mask.

        Returns:
            input_ids: List[int] of length max_length
            attention_mask: List[int] of length max_length
        """
        seq = self._normalize_seq(seq)

        token_ids: List[int] = []

        if add_special_tokens:
            token_ids.append(self.bos_token_id)

        # Map each base to id
        for ch in seq:
            tok_id = self.token_to_id.get(ch, self.token_to_id["N"])
            token_ids.append(tok_id)

        if add_special_tokens:
            token_ids.append(self.eos_token_id)

        # Truncate if too long
        if len(token_ids) > max_length:
            # Keep BOS + first (max_length-2) bases + EOS
            if add_special_tokens:
                token_ids = (
                    [self.bos_token_id]
                    + token_ids[1 : max_length - 1]
                    + [self.eos_token_id]
                )
            else:
                token_ids = token_ids[:max_length]

        # Build attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)

        # Pad if too short
        while len(token_ids) < max_length:
            token_ids.append(self.pad_token_id)
            attention_mask.append(0)

        return token_ids, attention_mask

    def batch_encode(
        self,
        seqs: List[str],
        max_length: int,
        add_special_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a list of sequences into a batch of tensors.

        Returns:
            {
                "input_ids": LongTensor [batch_size, max_length],
                "attention_mask": LongTensor [batch_size, max_length],
            }
        """
        all_ids: List[List[int]] = []
        all_masks: List[List[int]] = []

        for s in seqs:
            ids, mask = self.encode(s, max_length=max_length, add_special_tokens=add_special_tokens)
            all_ids.append(ids)
            all_masks.append(mask)

        input_ids = torch.tensor(all_ids, dtype=torch.long)
        attention_mask = torch.tensor(all_masks, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token ids back into a DNA string.
        """
        tokens = []
        for idx in ids:
            tok = self.id_to_token.get(int(idx), "N")
            if skip_special_tokens and tok in self.special_tokens:
                continue
            tokens.append(tok)

        # Join but remove possible 'N' coming from non-base tokens (if any)
        seq = "".join(t for t in tokens if t in self.base_tokens)
        return seq
