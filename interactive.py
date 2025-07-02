#!/usr/bin/env python3
# interactive.py

import argparse
import json
import os
import torch
from transformers import BertTokenizerFast
from model import LMModule
from network import TransformerLM
from sampler import TextSampler


def main():
    parser = argparse.ArgumentParser(
        description="Interactive prompt/response with your trained LLM"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to a .pt checkpoint (epoch or final model)"
    )
    parser.add_argument(
        "--vocab_model",
        type=str,
        default="bert-base-uncased",
        help="HF model name for the tokenizer"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="How many tokens to generate per prompt"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (>0, higher = more random)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="If set, restrict sampling to top_k tokens"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="If set, use nucleus (top-p) sampling"
    )
    args = parser.parse_args()

    # infer model dir & load its config.json
    model_dir = os.path.dirname(args.checkpoint)
    cfg_path  = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Couldn’t find {cfg_path}")
    cfg = json.load(open(cfg_path, "r"))

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build network with the same hyperparams you used in training
    network = TransformerLM(
        vocab_size     = cfg["vocab_size"],
        seq_length     = cfg["seq_length"],
        hidden_size    = cfg["hidden_size"],
        num_layers     = cfg["num_layers"],
        num_heads      = cfg["num_heads"],
        ffn_size       = cfg["ffn_size"],
        dropout        = cfg["dropout"],
    ).to(device)

    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(
        args.vocab_model,
        model_max_length=network.seq_length * 1_000_000,  # avoid “sequence length” warnings
        pad_to_max_length=False,
    )

    # wrap & load weights
    lm = LMModule(
        network,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id,
    ).to(device)
    lm.load(args.checkpoint)

    # sampler helper
    sampler = TextSampler(
        lm,
        tokenizer,
        device,
        max_new_tokens = args.max_new_tokens,
        temperature    = args.temperature,
        top_k          = args.top_k,
        top_p          = args.top_p,
    )

    print("Loaded checkpoint:", args.checkpoint)
    print("Enter a prompt (or blank to quit).")
    while True:
        prompt = input(">>> ").strip()
        if not prompt:
            break
        out = sampler.generate(
            prompt,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
            top_p          = args.top_p,
        )
        print(out)
        print("-" * 80)

if __name__ == "__main__":
        main()
