
"""
Custom BART-based model for cfDNA / metastasis fragment classification.

This module:
- Loads a pretrained BART encoder (facebook/bart-base)
- Optionally freezes the base encoder (common in scientific fine-tuning)
- Adds a custom "FragmentPatternLayer" (MultiheadAttention)
- Adds a classification head on top for binary classification
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import BartModel


class FragmentPatternLayer(nn.Module):
    """
    A custom attention layer that re-processes the BART encoder outputs
    to focus on fragment-level patterns.

    Conceptual story for interview:
    - BART encoder learns contextual embeddings for each base/k-mer.
    - This layer re-attends over the sequence to highlight patterns
      like fragment ends, motif clusters, GC shifts, etc.
    """

    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        # batch_first=True -> (batch, seq, hidden) which matches BartModel output
        self.pattern_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Encoder hidden states, shape (batch, seq_len, hidden_size)
            attention_mask: 1 for tokens to attend to, 0 for padding,
                            shape (batch, seq_len) or None.

        Returns:
            x_out: Attention-refined representations, same shape as input.
        """
        # MultiheadAttention expects key_padding_mask with True for padding positions.
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: 1 = keep, 0 = pad
            # key_padding_mask: True = ignore (pad), False = keep
            key_padding_mask = (attention_mask == 0)

        # Self-attention over the same sequence
        x_out, _ = self.pattern_attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        return x_out


class CustomBartFragmentClassifier(nn.Module):
    """
    BART-based fragment classifier with an extra FragmentPatternLayer on top.

    - Uses the BART encoder as a base model.
    - Adds a FragmentPatternLayer that re-attends over encoder outputs.
    - Applies masked mean pooling to get a sequence-level representation.
    - Feeds pooled representation into a classification head.

    Default: binary classification (e.g. 0 = non-metastatic-like, 1 = metastatic-like).
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        num_labels: int = 2,
        freeze_encoder: bool = True,
        hidden_dropout_prob: float = 0.1,
    ):
        super().__init__()

        # Load pretrained BART (encoder + decoder, but we'll use encoder outputs)
        self.bart = BartModel.from_pretrained(model_name)

        hidden_size = self.bart.config.hidden_size
        self.num_labels = num_labels

        # Optionally freeze the encoder (common when data is limited)
        if freeze_encoder:
            for param in self.bart.encoder.parameters():
                param.requires_grad = False

        # Custom fragment-pattern layer
        self.fragment_layer = FragmentPatternLayer(hidden_size=hidden_size, num_heads=4)

        # Classification head on top of pooled representation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(hidden_size, num_labels),
        )

        # Standard cross-entropy for classification
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through:
        BART encoder -> FragmentPatternLayer -> masked pooling -> classifier.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) with 1 for tokens, 0 for padding
            labels: (batch,) with class indices [0 .. num_labels-1]

        Returns:
            A dict with:
                logits: (batch, num_labels)
                loss: scalar tensor or None if labels is None
        """
        # BART returns:
        # last_hidden_state: (batch, seq_len, hidden_size)
        encoder_outputs = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = encoder_outputs.last_hidden_state  # (B, L, H)

        # Apply custom fragment-level attention layer
        fragment_repr = self.fragment_layer(hidden_states, attention_mask=attention_mask)

        # Masked mean pooling over sequence
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
            fragment_repr = fragment_repr * mask  # zero-out padded positions
            sum_repr = fragment_repr.sum(dim=1)  # (B, H)
            lengths = mask.sum(dim=1).clamp(min=1e-9)  # avoid division by zero
            pooled = sum_repr / lengths
        else:
            # Fallback: simple mean over sequence
            pooled = fragment_repr.mean(dim=1)  # (B, H)

        # Classification head
        logits = self.classifier(pooled)  # (B, num_labels)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
        }


if __name__ == "__main__":
    # Quick sanity check so you can run:
    #   python -m model.custom_bart_layer
    # from the project root.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomBartFragmentClassifier(
        model_name="facebook/bart-base",
        num_labels=2,
        freeze_encoder=True,
    ).to(device)

    batch_size = 4
    seq_len = 128

    # Dummy batch (as if already tokenized DNA sequences)
    input_ids = torch.randint(low=0, high=1000, size=(batch_size, seq_len)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    labels = torch.randint(low=0, high=2, size=(batch_size,)).to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print("Logits shape:", out["logits"].shape)
    print("Loss:", out["loss"].item())
