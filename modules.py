# modules.py
# Core building blocks for a small-scale decoder-only Transformer (MLM-capable)

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init


class TokenEmbedding(nn.Module):
    """
    Embedding layer mapping token IDs to vectors.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.token_emb: nn.Embedding = nn.Embedding(vocab_size, hidden_size)
        self.initialize(hidden_size)

    def initialize(self, hidden_size: int) -> None:
        init.normal_(self.token_emb.weight, mean=0.0, std=hidden_size ** -0.5)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: LongTensor of shape [batch_size, seq_length]
        Returns:
            Tensor of shape [batch_size, seq_length, hidden_size]
        """
        return self.token_emb(input_ids)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal positional embedding (no learned parameters).
    """
    # declare the buffer’s type so static checkers treat it as a Tensor
    pos_emb: Tensor
    
    def __init__(
        self,
        seq_length: int,
        hidden_size: int,
        max_period: int = 10000,
    ) -> None:
        super().__init__()
        self.seq_length: int = seq_length
        self.hidden_size: int = hidden_size
        self.max_period: int = max_period
        # precompute [seq_len, hidden]
        self.register_buffer("pos_emb", self._build_embedding())  # Tensor [seq_length, hidden_size]

    def _build_embedding(self) -> Tensor:
        position: Tensor = torch.arange(0, self.seq_length, dtype=torch.float).unsqueeze(1)
        div_term: Tensor = torch.exp(
            -math.log(self.max_period) * torch.arange(0, self.hidden_size, 2, dtype=torch.float) / self.hidden_size
        )
        emb: Tensor = torch.zeros(self.seq_length, self.hidden_size)
        emb[:, 0::2] = torch.sin(position * div_term)
        emb[:, 1::2] = torch.cos(position * div_term)
        return emb

    # def forward(self, x: Tensor) -> Tensor:
    #     """
    #     Args:
    #         x: Tensor of shape [batch_size, seq_length, hidden_size]
    #     Returns:
    #         pos_emb expanded to [batch_size, seq_length, hidden_size]
    #     """
    #     batch_size: int = x.size(0)
    #     return self.pos_emb.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, hidden_size]
        Returns:
            pos_emb: Tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        # slice down to the actual sequence length
        # self.pos_emb is [max_seq_length, hidden_size]
        pos = self.pos_emb[:seq_len, :]           # [seq_len, H]
        # broadcast to [batch_size, seq_len, H]
        return pos.unsqueeze(0).expand(batch_size, seq_len, -1)

class MultiHeadSelfAttention(nn.Module):
    """
    Batch-first multi-head self-attention with causal mask support.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias = False
        )
        self.initialize()

    def initialize(self) -> None:
        # initialize in_proj and out_proj
        init.xavier_uniform_(self.attn.in_proj_weight)
        init.xavier_uniform_(self.attn.out_proj.weight)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: [batch_size, seq_length, hidden_size]
            attn_mask: [seq_length, seq_length] causal mask where True indicates prohibited
        Returns:
            Tensor same shape as x
        """
        out: Tensor
        out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        return out


class FeedForward(nn.Module):
    """
    Two-layer feed-forward network with activation and dropout.
    """
    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
        )
        self.initialize()

    def initialize(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch_size, seq_length, hidden_size]
        Returns:
            Tensor same shape as x
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single Transformer block with pre-norm, self-attention, and feed-forward.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1: nn.LayerNorm = nn.LayerNorm(hidden_size)
        self.attn: MultiHeadSelfAttention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2: nn.LayerNorm = nn.LayerNorm(hidden_size)
        self.ff: FeedForward = FeedForward(
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            dropout=dropout,
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: [batch_size, seq_length, hidden_size]
            attn_mask: [seq_length, seq_length]
        Returns:
            Tensor same shape as x
        """
        h1: Tensor = self.norm1(x)
        h1 = self.attn(h1, attn_mask=attn_mask)
        x = x + h1
        h2: Tensor = self.norm2(x)
        h2 = self.ff(h2)
        return x + h2
