# dataset.py
# Data utilities: MLMDataset for masked language modeling on text corpora

import random
from pathlib import Path
from typing import Optional, List

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


class MLMDataset(Dataset):
    """
    Dataset for Masked Language Modeling (MLM) on a directory of text files.

    Each file in `data_dir` is expected to contain an individual document or article.
    The text is tokenized, packed into fixed-length sequences with padding,
    and dynamic masking is applied on the fly.

    Args:
        data_dir: Path to directory containing text files (e.g., ".txt").
        tokenizer: Hugging Face tokenizer with `encode` and `mask_token_id` attributes.
        seq_length: Sequence length to pack tokens into (with padding/truncation).
        mask_rate: Fraction of tokens to mask in each sequence (e.g., 0.15).
    """
    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizerFast,
        seq_length: int,
        mask_rate: float = 0.15,
    ) -> None:
        self.data_dir: Path = Path(data_dir)
        self.tokenizer: PreTrainedTokenizerFast = tokenizer
        self.seq_length: int = seq_length
        self.mask_rate: float = mask_rate
        self.samples: List[Tensor] = []

        # Load and tokenize all files
        for file_path in self.data_dir.glob("*.txt"):
            text = file_path.read_text(encoding="utf-8")
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            # Pack into seq_length chunks
            for i in range(0, len(token_ids), seq_length):
                chunk = token_ids[i : i + seq_length]
                # pad if necessary
                if len(chunk) < seq_length:
                    chunk += [tokenizer.pad_token_id] * (seq_length - len(chunk))
                self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> (Tensor, Tensor):
        """
        Returns a pair (input_ids, labels) where:
        - input_ids has masked tokens according to `mask_rate` and
        - labels has the original token IDs or -100 for unmasked positions.
        """
        # Clone raw tokens
        tokens: Tensor = self.samples[index].clone()
        labels: Tensor = tokens.clone()

        # Determine mask indices dynamically
        n_to_mask: int = max(1, int(self.mask_rate * self.seq_length))
        mask_indices = random.sample(range(self.seq_length), n_to_mask)

        for idx in mask_indices:
            prob = random.random()
            if prob < 0.8:
                tokens[idx] = self.tokenizer.mask_token_id
            elif prob < 0.9:
                tokens[idx] = random.randrange(self.tokenizer.vocab_size)
            # else: 10% keep original

        # For unlabeled positions, set label to -100 to ignore in loss
        labels_mask = torch.full((self.seq_length,), -100, dtype=torch.long)
        labels_mask[mask_indices] = labels[mask_indices]

        return tokens, labels_mask
