# dataset.py

import random, os
from pathlib import Path
import torch
from typing import Iterator, Tuple
from torch.utils.data import IterableDataset
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
        mask_rate: float = 0.15,
    ):
        super().__init__()
        self.files = list(Path(data_dir).glob("*.txt"))
        if not self.files:
            raise ValueError(f"No .txt files found in {data_dir}")
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.mask_rate = mask_rate

        # # Build the DataPipe pipeline
        # files_dp = FileLister(self.data_dir, masks=["*.txt"])
        # # Shuffle file order per epoch
        # files_dp = Shuffler(files_dp, buffer_size=len(list(files_dp)))
        # files_dp = FileOpener(files_dp, mode="r")         # yields (filepath, fileobj)
        # lines_dp = files_dp.flatmap(
        #     lambda path_file: path_file[1].read().splitlines()
        # )

        # # Tokenize each line
        # tok_dp = lines_dp.map(lambda line: self.tokenizer.encode(
        #     line.strip(), add_special_tokens=False
        # ))

        # # Chunk into fixed-length windows
        # flat_dp = tok_dp.flatmap(
        #     lambda ids: [
        #         ids[i : i + self.seq_length]
        #         for i in range(0, len(ids) - self.seq_length + 1, self.seq_length)
        #     ]
        # )

        # # Shuffle windows locally
        # flat_dp = Shuffler(flat_dp, buffer_size=10_000)

        # # Batch into micro-batches
        # batch_dp = Batcher(flat_dp, batch_size=batch_size, drop_last=True)

        # # Apply dynamic masking per batch
        # self.datapipe = batch_dp.map(self._mask_batch)
    def set_mask_rate(self, rate: float):
        self.mask_rate = rate
        
    # def _mask_batch(self, batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    #     tokens_list, labels_list = [], []
    #     for seq in batch:
    #         labels = seq.copy()
    #         tokens = seq.copy()
    #         for i in range(len(tokens)):
    #             if random.random() < self.mask_rate:
    #                 p = random.random()
    #                 if p < 0.8:
    #                     tokens[i] = self.tokenizer.mask_token_id
    #                 elif p < 0.9:
    #                     tokens[i] = random.randrange(self.tokenizer.vocab_size)
    #                 # else: leave unchanged
    #             else:
    #                 labels[i] = -100 # ignore index
    #         tokens_list.append(tokens)
    #         labels_list.append(labels)

    #     tokens_tensor = torch.tensor(tokens_list, dtype=torch.long)
    #     labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    #     return tokens_tensor, labels_tensor

    # def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    #     return iter(self.datapipe)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # If multiple workers, split file list evenly
        worker_info = torch.utils.data.get_worker_info()
        files = self.files
        if worker_info:
            per_worker = files[worker_info.id :: worker_info.num_workers]
        else:
            per_worker = files

        # Shuffle file order per epoch
        random.shuffle(per_worker)

        for file_path in per_worker:
            text = file_path.read_text(encoding="utf-8", errors="ignore")

            # Tokenize entire file once
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )

            # Make chunk start indices: 0, seq_length, 2*seq_length, …
            n_tokens = len(token_ids)
            if n_tokens < self.seq_length:
                continue
            starts = list(range(0, n_tokens - self.seq_length + 1, self.seq_length))
            random.shuffle(starts)

            for st in starts:
                chunk = token_ids[st : st + self.seq_length]
                tokens = torch.tensor(chunk, dtype=torch.long)
                labels = tokens.clone()

                # Dynamic masking 80/10/10:
                mask_indices = torch.rand(self.seq_length) < self.mask_rate
                mask_token = self.tokenizer.mask_token_id
                for i in torch.nonzero(mask_indices, as_tuple=False).view(-1).tolist():
                    p = random.random()
                    if p < 0.8:
                        # mask_token is actually an int at runtime so we can ignore typechecker
                        tokens[i] = mask_token # type: ignore
                    elif p < 0.9:
                        tokens[i] = random.randrange(self.tokenizer.vocab_size)
                    # else 10% leave unchanged

                yield tokens, labels