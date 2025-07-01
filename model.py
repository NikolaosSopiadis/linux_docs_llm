# model.py
# Helper class wrapping the Transformer LM with training and utility methods

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

data = torch.utils.data

class LMModule(nn.Module):
    """
    Wrapper for a decoder-only Transformer language model that provides
    loss computation, generation, and checkpointing utilities.
    """
    def __init__(
        self,
        network: nn.Module,
        pad_token_id: int,
        eos_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.network: nn.Module = network
        self.pad_token_id: int = pad_token_id
        self.eos_token_id: Optional[int] = eos_token_id

    @property
    def device(self) -> torch.device:
        """Device on which the network parameters reside."""
        return next(self.network.parameters()).device

    def get_loss(
        self,
        input_ids: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Compute MLM loss between logits and labels.

        Args:
            input_ids: [batch_size, seq_length]
            labels:    [batch_size, seq_length] with -100 for positions to ignore
        Returns:
            scalar loss tensor
        """
        logits: Tensor = self.network(input_ids)  # [B, S, V]
        B, S, V = logits.shape
        # flatten
        logits_flat: Tensor = logits.view(-1, V)
        labels_flat: Tensor = labels.view(-1)
        loss: Tensor = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
        return loss

    @torch.no_grad()
    def sample(
        self,
        input_ids: Tensor,
        max_length: int = 64,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Generate tokens autoregressively using the inner network's generate method.

        Args:
            input_ids: [batch_size, cur_len]
            max_length: total length including initial
            eos_token_id: optional token id to stop on

        Returns:
            Tensor [batch_size, generated_len]
        """
        # delegate to network if it has generate
        if hasattr(self.network, 'generate'):
            return self.network.generate(input_ids=input_ids, max_length=max_length, eos_token_id=eos_token_id)  # type: ignore
        # fallback greedy
        generated = input_ids
        for _ in range(max_length - generated.size(1)):
            logits = self.network(generated)
            next_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tokens], dim=1)
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
        return generated

    def save(self, file_path: str) -> None:
        """Save model and optimizer state to file."""
        payload: Dict[str, Any] = {
            'model_state': self.network.state_dict(),
            'pad_token_id': self.pad_token_id,
            'eos_token_id': self.eos_token_id,
        }
        torch.save(payload, file_path)

    def load(self, file_path: str) -> None:
        """Load model state from file."""
        payload = torch.load(file_path, map_location='cpu')
        state_dict = payload.get('model_state', payload)
        self.network.load_state_dict(state_dict)
        # restore tokenizer specials if present
        if 'pad_token_id' in payload:
            self.pad_token_id = payload['pad_token_id']  # type: ignore
        if 'eos_token_id' in payload:
            self.eos_token_id = payload['eos_token_id']  # type: ignore
