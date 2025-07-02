# dataset.py

import random
import torch
from typing import Iterator, Tuple
from torch.utils.data import IterableDataset
from torch.utils.data.datapipes.iter import (
    FileLister,
    FileOpener,
    Shuffler,
    Batcher
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class StreamingMLMDataset(IterableDataset):
    """
    Streams text files from a directory, tokenizes and chunks them into fixed-length sequences,
    shuffles locally, and applies dynamic MLM masking in batches.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizerFast,
        seq_length: int,
        mask_rate: float,
        batch_size: int,
        num_workers: int
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.mask_rate = mask_rate

        # Build the DataPipe pipeline
        files_dp = FileLister(self.data_dir, masks=["*.txt"])
        # Shuffle file order per epoch
        files_dp = Shuffler(files_dp, buffer_size=len(list(files_dp)))
        files_dp = FileOpener(files_dp, mode="r")         # yields (filepath, fileobj)
        lines_dp = files_dp.flatmap(lambda path_obj: path_obj[1])

        # Tokenize each line
        tok_dp = lines_dp.map(lambda line: self.tokenizer.encode(
            line.strip(), add_special_tokens=False
        ))

        # Chunk into fixed-length windows
        flat_dp = tok_dp.flatmap(
            lambda ids: [
                ids[i : i + self.seq_length]
                for i in range(0, len(ids) - self.seq_length + 1, self.seq_length)
            ]
        )

        # Shuffle windows locally
        flat_dp = Shuffler(flat_dp, buffer_size=10_000)

        # Batch into micro-batches
        batch_dp = Batcher(flat_dp, batch_size=batch_size, drop_last=True)

        # Apply dynamic masking per batch
        self.datapipe = batch_dp.map(self._mask_batch)

    def _mask_batch(self, batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens_list, labels_list = [], []
        for seq in batch:
            labels = seq.copy()
            tokens = seq.copy()
            for i in range(len(tokens)):
                if random.random() < self.mask_rate:
                    p = random.random()
                    if p < 0.8:
                        tokens[i] = self.tokenizer.mask_token_id
                    elif p < 0.9:
                        tokens[i] = random.randrange(self.tokenizer.vocab_size)
                    # else: leave unchanged
                else:
                    labels[i] = -100 # ignore index
            tokens_list.append(tokens)
            labels_list.append(labels)

        tokens_tensor = torch.tensor(tokens_list, dtype=torch.long)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        return tokens_tensor, labels_tensor

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        return iter(self.datapipe)
