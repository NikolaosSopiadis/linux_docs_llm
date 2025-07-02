#!/usr/bin/env python3
import argparse
from pathlib import Path
from transformers import BertTokenizerFast
from tqdm import tqdm

def count_file_chunks(path: Path, tokenizer: BertTokenizerFast, seq_length: int) -> int:
    """
    Read the file in 1 MiB chunks, tokenize each chunk, and
    count how many non-overlapping seq_length sequences it yields,
    using a small rolling buffer to catch leftover tokens.
    """
    total = 0
    buffer = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            text = f.read(1 << 20)  # 1 MiB
            if not text:
                break
            toks = tokenizer.encode(text, add_special_tokens=False)
            if not toks:
                continue
            buffer.extend(toks)
            # count how many full seq_length chunks we can take
            n = len(buffer) // seq_length
            if n:
                total += n
                # drop those tokens from the buffer
                buffer = buffer[n * seq_length :]
    return total

def main():
    parser = argparse.ArgumentParser(
        description="Compute exact steps_per_epoch for 128-token MLM sequences"
    )
    parser.add_argument("--data_dir",    type=str, default="data/train")
    parser.add_argument("--seq_length",  type=int, default=128)
    parser.add_argument("--batch_size",  type=int, default=96)
    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-uncased",
        model_max_length= args.seq_length * 1_000_000,  # avoid “sequence length” warnings
        pad_to_max_length=False
    )


    files = list(Path(args.data_dir).glob("*.txt"))
    if not files:
        raise RuntimeError(f"No .txt files found under {args.data_dir}")

    total_seqs = 0
    for path in tqdm(files, desc="Counting files"):
        total_seqs += count_file_chunks(path, tokenizer, args.seq_length)

    steps_per_epoch = (total_seqs + args.batch_size - 1) // args.batch_size
    print(f"{total_seqs:,} total {args.seq_length}-token sequences")
    print(f"{steps_per_epoch:,} steps per epoch (batch_size={args.batch_size})")

if __name__ == "__main__":
    main()

# output:
# Counting files: 100%|████████████████████████████████████████████████████████████████| 2594/2594 [15:31<00:00,  2.79it/s]
# 4,905,930 total 128-token sequences
# 51,104 steps per epoch (batch_size=96)