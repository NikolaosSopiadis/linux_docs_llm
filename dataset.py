# dataset.py

import random, os
from typing import List
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
        mask_rate: float = 0.25,
    ):
        super().__init__()
        self.files = list(Path(data_dir).glob("*.txt"))
        if not self.files:
            raise ValueError(f"No .txt files found in {data_dir}")
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.mask_rate = mask_rate

    def set_mask_rate(self, rate: float):
        self.mask_rate = rate
        
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
            buffer_ids: List[int] = []
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                while True:
                    chunk = f.read(1 << 20)  # 1 MiB at a time
                    if not chunk:
                        break
                    ids = self.tokenizer.encode(chunk, add_special_tokens=False)
                    buffer_ids.extend(ids)

                    # emit all full seq_length chunks
                    n = len(buffer_ids) // self.seq_length
                    if not n:
                        continue
                    for i in range(n):
                        seq = buffer_ids[i*self.seq_length : (i+1)*self.seq_length]
                        tokens = torch.tensor(seq, dtype=torch.long)
                        labels = tokens.clone()
                        
                        # 80/10/10 dynamic mJonas Geipingasking:
                        #  - mask_rate of positions selected for masking
                        #  - 80% replaced with [MASK], 10% random, 10% unchanged
                        mask_indices = torch.rand(self.seq_length) < self.mask_rate
                        mask_token = self.tokenizer.mask_token_id

                        # For tokens *not* masked, set label = -100 so they're ignored in loss
                        labels[~mask_indices] = -100

                        # Apply masking to selected positions
                        for idx in torch.nonzero(mask_indices, as_tuple=False).view(-1).tolist():
                            prob = random.random()
                            if prob < 0.8:
                                tokens[idx] = mask_token  # type: ignore
                            elif prob < 0.9:
                                tokens[idx] = random.randrange(self.tokenizer.vocab_size)
                            # else: leave the token unchanged 
                        
                        yield tokens, labels

                    # drop the emitted tokens, keep the remainder
                    buffer_ids = buffer_ids[n*self.seq_length :]
https://github.com/NikolaosSopiadis/linux_docs_llm