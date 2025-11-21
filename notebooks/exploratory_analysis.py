
# notebooks/exploratory_analysis.ipynb

import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

# Make repo root importable
repo_root = Path.cwd().parent  # adjust if needed
sys.path.append(str(repo_root))

from model.dna_tokenizer import DNATokenizer

tokenizer = DNATokenizer()
print("Vocab size:", tokenizer.vocab_size)
print("Token map:", tokenizer.token_to_id)

seq = "ACGTNACGTACGT"
max_len = 20

input_ids, attention_mask = tokenizer.encode(seq, max_length=max_len)
print("Input IDs:", input_ids)
print("Attention mask:", attention_mask)

print("Decoded:", tokenizer.decode(input_ids))

# Batch encode example
seqs = [
    "ACGTACGTACGT",
    "NNNNNNNNNN",
    "TTTTACGT",
]

batch = tokenizer.batch_encode(seqs, max_length=20)
print("Batch input_ids shape:", batch["input_ids"].shape)
print(batch["input_ids"])
print(batch["attention_mask"])

# Simple visualization: number of non-padding tokens
non_pad_counts = batch["attention_mask"].sum(dim=1).tolist()

plt.bar(range(len(seqs)), non_pad_counts)
plt.xlabel("Sequence index")
plt.ylabel("Non-PAD token count")
plt.title("Tokenized sequence lengths (with BOS/EOS)")
plt.show()
