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
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Autoregressive sampling from the model.

        Args:
          input_ids: [B, S] starting tokens
          max_new_tokens: how many tokens to generate
          temperature: >0. Higher = more random. 1.0 = no scaling.
          top_k: if set, only sample from the top_k logits each step
          top_p: if set, use nucleus sampling: smallest set with cum-prob ≥ top_p
          eos_token_id: if provided, stop when all batches have generated it

        Returns:
          Tensor of shape [B, S+T] with the generated continuation appended.
        """
        B, S = input_ids.shape
        device = input_ids.device
        eos = eos_token_id if eos_token_id is not None else self.eos_token_id

        generated = input_ids
        for _ in range(max_new_tokens):
            # 1) forward pass
            logits = self.network(generated)           # [B, S+t, V]
            next_logits = logits[:, -1, :]            # [B, V]

            # 2) apply temperature
            next_logits = next_logits / temperature

            # 3) top-k
            if top_k is not None and top_k > 0:
                values, indices = torch.topk(next_logits, top_k)
                # mask out everything not in top_k
                mask = next_logits < values[..., -1, None]
                next_logits = next_logits.masked_fill(mask, -float('Inf'))

            # 4) top-p (nucleus) sampling
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # mask tokens beyond cumulative top_p
                sorted_mask = cumulative_probs > top_p
                # shift mask right so we always keep at least one
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False

                # scatter the mask back to original logits
                mask = sorted_mask.scatter(-1, sorted_indices, sorted_mask)
                next_logits = next_logits.masked_fill(mask, -float('Inf'))

            # 5) sample
            probs = F.softmax(next_logits, dim=-1)       # [B, V]
            next_tokens = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # 6) append
            generated = torch.cat([generated, next_tokens], dim=1)  # [B, S+t+1]

            # 7) optional early stop
            if eos is not None:
                # if every batch has emitted eos at last position, stop.
                if (next_tokens == eos).all():
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
        # pick out whichever key holds the raw weights
        if 'model_state_dict' in payload:
            raw_sd = payload['model_state_dict']
        elif 'model_state' in payload:
            raw_sd = payload['model_state']
        elif 'state_dict' in payload:
            raw_sd = payload['state_dict']
        else:
            raw_sd = payload

        # strip any "_orig_mod." prefixes from keys
        new_sd = {}
        prefix = '_orig_mod.'
        for k, v in raw_sd.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
            else:
                new_key = k
            new_sd[new_key] = v

        # load into network
        self.network.load_state_dict(new_sd)
        # restore tokenizer specials if present
        if 'pad_token_id' in payload:
            self.pad_token_id = payload['pad_token_id']  # type: ignore
        if 'eos_token_id' in payload:
            self.eos_token_id = payload['eos_token_id']  # type: ignore
