# network.py

from typing import Optional, List

import torch
import torch.nn as nn
from torch import Tensor

from modules import (
    TokenEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerBlock,
)


def build_causal_mask(seq_length: int, device: torch.device) -> Tensor:
    """
    Create a causal attention mask where positions [i, j] with j > i are masked.
    Returns a boolean mask of shape [seq_length, seq_length]: True for illegal (masked) positions.
    """
    # mask[i, j] = True if j > i
    mask = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool, device=device), diagonal=1)
    return mask


class TransformerLM(nn.Module):
    """
    Decoder-only Transformer for MLM/fill-in-the-blank tasks.
    """
    def __init__(
        self,
        vocab_size: int,
        seq_length: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        ffn_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_length: int = seq_length
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # embeddings
        self.token_embedding: TokenEmbedding = TokenEmbedding(vocab_size, hidden_size)
        self.pos_embedding: SinusoidalPositionalEmbedding = SinusoidalPositionalEmbedding(
            seq_length, hidden_size
        )
        # transformer layers
        self.layers: nn.ModuleList = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ffn_size, dropout)
            for _ in range(num_layers)
        ])
        # final norm + head
        self.norm: nn.LayerNorm = nn.LayerNorm(hidden_size)
        self.lm_head: nn.Linear = nn.Linear(hidden_size, vocab_size, bias=False)
        # tie weights
        self.lm_head.weight = self.token_embedding.token_emb.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: LongTensor of shape [batch_size, seq_length]
        Returns:
            logits: Tensor of shape [batch_size, seq_length, vocab_size]
        """
        batch_size, seq_len = input_ids.size()
        B, seq_len = input_ids.shape
        # allow any seq_len up to model max; error if it’s longer
        if seq_len > self.seq_length:
            raise ValueError(f"Input sequence length {seq_len} exceeds model max of {self.seq_length}")

        # # build or reuse mask
        # causal_mask = build_causal_mask(self.seq_length, self.device)
        # build a causal mask exactly the size of our current sequence
        causal_mask = build_causal_mask(seq_len, input_ids.device)
        
        # embeddings
        # x: Tensor = self.token_embedding(input_ids)  # [B, S, H]
        # x = x + self.pos_embedding(x)  # [B, S, H]
        x = self.token_embedding(input_ids)       # [B, S, H]
        x = x + self.pos_embedding(x)             # [B, S, H]

        # transformer blocks
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask)
            
        # final layers
        x = self.norm(x)
        logits: Tensor = self.lm_head(x)
        return logits

    def generate(
        self,
        input_ids: Tensor,
        max_length: int = 64,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Autoregressive text generation using greedy decoding.
        input_ids: [batch_size, cur_len]
        Returns: [batch_size, generated_len]
        """
        device = input_ids.device
        generated = input_ids
        for _ in range(max_length - generated.size(1)):
            seq_len = generated.size(1)
            # assert seq_len <= self.seq_length, "Generation length exceeds model context"
            # # pad to seq_length
            # pad_len = self.seq_length - seq_len
            # input_pad = nn.functional.pad(generated, (pad_len, 0), value=0)
            # logits = self.forward(input_pad)  # [B, S, V]
            # next_logits = logits[:, seq_len - 1, :]  # last position


            if seq_len > self.seq_length:
                raise ValueError(f"Generation length {seq_len} > model context {self.seq_length}")
            # Directly forward the growing sequence (now allowed ≤ seq_length)
            logits = self.forward(generated)    # [B, seq_len, V]
            next_logits = logits[:, -1, :]      # last position
            
            next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)  # [B,1]
            generated = torch.cat([generated, next_tokens], dim=1)
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
        return generated
