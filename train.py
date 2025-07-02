# train.py
# End-to-end training script for small-scale decoder-only Transformer MLM with mask-rate scheduling and eval metrics

import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import List
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from transformers import BertTokenizerFast #, get_cosine_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import TokenEmbedding, SinusoidalPositionalEmbedding, TransformerBlock
from network import TransformerLM
from model import LMModule
from dataset import StreamingMLMDataset

# silence the “tokenizers parallelism” warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def parse_args():
    parser = argparse.ArgumentParser(description='Train a decoder-only Transformer MLM')
    # Data
    parser.add_argument('--train_dir', type=str, required=True, help='Directory of training text files')
    parser.add_argument('--eval_dir', type=str, required=True, help='Directory of eval text files')
    parser.add_argument('--seq_length', type=int, default=128)
    parser.add_argument('--vocab_model', type=str, default='bert-base-uncased')
    # Mask scheduling
    parser.add_argument('--initial_mask_rate', type=float, default=0.4, help='Starting mask rate')
    parser.add_argument('--final_mask_rate', type=float, default=0.15, help='Ending mask rate')
    # Model
    parser.add_argument('--vocab_size', type=int, default=32128)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ffn_size', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    # Training
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--accumulate_steps', type=int, default=24)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--peak_lr', type=float, default=3e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--clip_grad_norm', type=float, default=0.5)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup output directory and config
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(args.save_dir) / now
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(
        args.vocab_model,
        model_max_length=args.seq_length * 1_000_000,  # avoid “sequence length” warnings
        pad_to_max_length=False,
    )
    # sync our model size with the tokenizer’s vocab
    args.vocab_size = tokenizer.vocab_size

    # Datasets (streaming) with initial mask rate
    train_ds = StreamingMLMDataset(
        args.train_dir, tokenizer,
        args.seq_length, args.initial_mask_rate,
        args.batch_size, args.num_workers
    )
    eval_ds = StreamingMLMDataset(
        args.eval_dir, tokenizer,
        args.seq_length, args.initial_mask_rate,
        args.batch_size, args.num_workers
    )

    # DataLoaders: batch_size=None because datapipes already batch
    train_loader = DataLoader(
        train_ds,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Model
    network = TransformerLM(
        vocab_size=args.vocab_size,
        seq_length=args.seq_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_size=args.ffn_size,
        dropout=args.dropout,
    ).to(device)
    lm_module = LMModule(network, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        lm_module.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay
    )
    total_steps = (len(train_loader) * args.max_epochs) // args.accumulate_steps
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Training for {args.max_epochs} epochs ({total_steps} steps), warmup={warmup_steps} steps.")
    
    # Training Loop
    global_step = 0
    val_losses: List[float] = []
    val_ppls:   List[float] = []
    scaler = torch.GradScaler("cuda")

    print("Beginning training loop...")
    for epoch in range(1, args.max_epochs + 1):
        # Update mask rate per epoch
        if args.max_epochs == 1:
            mask_rate = args.final_mask_rate
        else:
            mask_rate = args.initial_mask_rate + (args.final_mask_rate - args.initial_mask_rate) * ((epoch-1)/(args.max_epochs-1))
        train_ds.mask_rate = mask_rate
        eval_ds.mask_rate  = mask_rate

        train_ds.mask_rate = mask_rate
        eval_ds.mask_rate = mask_rate
        print(f"Epoch {epoch} mask rate: {mask_rate:.3f}")

        lm_module.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for step, (tokens, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            tokens, labels = tokens.to(device), labels.to(device)
            # use mixed precision
            with torch.autocast("cuda"):
                loss = lm_module.get_loss(tokens, labels) / args.accumulate_steps
            scaler.scale(loss).backward()
            running_loss += loss.item()

            if (step + 1) % args.accumulate_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(lm_module.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # if global_step % args.log_interval == 0:
                #     avg_loss = running_loss / args.log_interval
                #     current_lr = scheduler.get_last_lr()[0]
                #     print(f"Step {global_step}, LR: {current_lr:.3e}, Loss: {avg_loss:.4f}")
                #     running_loss = 0.0
                #     # Save checkpoint
                #     ckpt = {
                #         'epoch': epoch,
                #         'global_step': global_step,
                #         'model_state': lm_module.network.state_dict(),
                #         'optimizer_state': optimizer.state_dict(),
                #         'scheduler_state': scheduler.state_dict(),
                #     }
                #     torch.save(ckpt, save_path / f'checkpoint_step{global_step}.pt')

        # Validation
        lm_module.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tokens, labels in eval_loader:
                tokens, labels = tokens.to(device), labels.to(device)
                val_loss += lm_module.get_loss(tokens, labels).item()
        val_loss /= len(eval_loader)
        val_ppl = float(torch.exp(torch.tensor(val_loss)))
        val_losses.append(val_loss)
        val_ppls.append(val_ppl)
        epochs = list(range(1, epoch + 1))    

        # Plot validation loss curve
        plt.figure()
        plt.plot(epochs, val_losses, marker='o')
        plt.title('Validation Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(save_path / 'val_loss_per_epoch.png')
        plt.close()

        # Plot perplexity curve
        plt.figure()
        plt.plot(epochs, val_ppls, marker='o')
        plt.title('Validation Perplexity per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.grid(True)
        plt.savefig(save_path / 'val_ppl_per_epoch.png')
        plt.close()
        
        print(f"Finished epoch {epoch}/{args.max_epochs}")
        print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")

    # Final model save
    torch.save(lm_module.network.state_dict(), save_path / 'final_model.pt')
    print("Training complete.")


if __name__ == '__main__':
    main()
