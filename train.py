# train.py

import argparse
import json
import os
import random
import itertools
import math
import numpy as np
import torch
import torch.nn as nn
from typing import List
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from transformers import BertTokenizerFast #, get_cosine_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import TokenEmbedding, SinusoidalPositionalEmbedding, TransformerBlock
from network import TransformerLM
from model import LMModule
from dataset import StreamingMLMDataset

# silence the “tokenizers parallelism” warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# kernel tuning for incrased performance 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description='Train a decoder-only Transformer MLM')
    # Data
    parser.add_argument('--train_dir', type=str, required=True, help='Directory of training text files')
    parser.add_argument('--eval_dir', type=str, required=True, help='Directory of eval text files')
    parser.add_argument('--seq_length', type=int, default=128)
    parser.add_argument('--vocab_model', type=str, default='bert-base-uncased')
    # Mask scheduling
    parser.add_argument('--mask_rate', type=float, default=0.25)
    # Model
    parser.add_argument('--vocab_size', type=int, default=32128)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--ffn_size', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--onecycle', type=bool, default=False)
    # Training
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--accumulate_steps', type=int, default=24)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--peak_lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm_start', type=float, default=1.5)
    parser.add_argument('--clip_grad_norm_end', type=float, default=0.5)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--steps_per_epoch', type=int, default=3951)
    parser.add_argument("--eval_steps", type=int, default=438)
    parser.add_argument("--acc_steps_start",type=int, default=10)
    parser.add_argument("--acc_steps_end", type=int, default=40)

    return parser.parse_args()

def topk_accuracy(
    logits: torch.Tensor,    # [N, V]
    labels: torch.Tensor,    # [N]
    k: int
) -> float:
    """
    Fraction of positions where the true label is in the top-k logits.
    """
    topk = torch.topk(logits, k, dim=-1).indices   # [N, k]
    hits = (topk == labels.unsqueeze(-1)).any(dim=-1)
    return hits.float().mean().item()


def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(
        args.vocab_model,
        model_max_length=args.seq_length * 1_000_000,  # avoid “sequence length” warnings
        pad_to_max_length=False,
    )
    # sync our model size with the tokenizer’s vocab
    args.vocab_size = tokenizer.vocab_size

    # Setup output directory and config
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(args.save_dir) / now
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Datasets (streaming) with constant mask rate
    train_ds = StreamingMLMDataset(args.train_dir, tokenizer, args.seq_length, args.mask_rate)
    eval_ds  = StreamingMLMDataset(args.eval_dir,  tokenizer, args.seq_length, args.mask_rate)

    loader_args = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        # possible performance incrase
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )

    train_loader = DataLoader(
        train_ds,
        **loader_args
    )
    eval_loader = DataLoader(
        eval_ds,
        **loader_args
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
    # use jit to increase performance
    if torch.__version__ >= "2.0":
        network = torch.compile(network)

    lm_module = LMModule(network, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id).to(device)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        lm_module.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps
    )
    # total_steps = (len(train_loader) * args.max_epochs) // args.accumulate_steps
    # warmup_steps = int(args.warmup_ratio * total_steps)
    
    if args.steps_per_epoch is not None:
        steps_per_epoch = args.steps_per_epoch
    else:
        # If train_loader has __len__, use it; otherwise error out
        try:
            steps_per_epoch = len(train_loader)
        except TypeError:
            raise RuntimeError(
                "train_loader has no length. Please pass --steps_per_epoch when "
                "using an IterableDataset."
            )

    # Total optimizer steps across all epochs (after accumulation)
    total_steps = (steps_per_epoch * args.max_epochs) // args.accumulate_steps
    warmup_steps = int(args.warmup_ratio * total_steps)
    
    if args.onecycle:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.peak_lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
            cycle_momentum=False
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Training for {args.max_epochs} epochs ({total_steps} steps), warmup={warmup_steps} steps.")
    
    # Training Loop
    global_step = 0
    val_losses: List[float] = []
    val_ppls:   List[float] = []
    val_accs:   List[float] = []   
    val_top5:   List[float] = []   
    lr_history: List[float] = []   
    scaler = torch.GradScaler("cuda")

    print("Beginning training loop...")
    for epoch in range(1, args.max_epochs + 1):
        print(f"Epoch {epoch}")
        
        # linearly ramp acc_steps over epochs
        if args.max_epochs == 1:
            acc_steps = args.acc_steps_end
        else:
            frac = (epoch - 1) / (args.max_epochs - 1)
            acc_steps = int(args.acc_steps_start + (args.acc_steps_end - args.acc_steps_start) * frac)
        print(f"Accumulating {acc_steps} micro-batches per update (total batch ~{acc_steps*args.batch_size})")

        lm_module.train()
        running_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch}")
        for step, (tokens, labels) in enumerate(pbar, start=1):
            # tokens, labels = tokens.to(device), labels.to(device)
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # use mixed precision
            with torch.autocast("cuda"):
                loss = lm_module.get_loss(tokens, labels) / acc_steps
            scaler.scale(loss).backward()
            running_loss += loss.item()

            if step % acc_steps == 0:
                # pbar.set_postfix_str(f"global_step={global_step}")                
                scaler.unscale_(optimizer)
                # linearly decrease gradient norm threshold
                start_clip = args.clip_grad_norm_start
                end_clip   = args.clip_grad_norm_end
                frac       = (epoch - 1)/(args.max_epochs - 1)
                clip_val   = start_clip + frac*(end_clip - start_clip)
                nn.utils.clip_grad_norm_(lm_module.parameters(), clip_val)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
        last_lr = scheduler.get_last_lr()[0]
        lr_history.append(last_lr)  

        # Validation
        lm_module.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        
        total_masked = 0
        correct1 = 0
        correct5 = 0
        
        val_iter = eval_loader
        if args.eval_steps:
            val_iter = itertools.islice(eval_loader, args.eval_steps)

        with torch.no_grad():
            for tokens, labels in tqdm(val_iter, total=args.eval_steps or None, desc="Validation"):
                # tokens, labels = tokens.to(device), labels.to(device)
                tokens = tokens.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True) # [B, S]
                
                # forward once
                logits = lm_module.network(tokens)              # [B, S, V]
                B, S, V = logits.shape
                
                # compute loss
                loss  = torch.nn.functional.cross_entropy(
                    logits.view(-1, V),
                    labels.view(-1),
                    ignore_index=-100
                )
                val_loss_sum += loss.item()
                # val_loss += lm_module.get_loss(tokens, labels).item()
                n_val_batches += 1
                        
                # top-1 predictions
                preds = logits.argmax(-1)                    # [B, S]

                # top-5 predictions
                top5 = logits.topk(5, dim=-1).indices        # [B, S, 5]

                # mask and count
                mask = labels != -100                        # [B, S]
                n_masked = mask.sum().item()
                if n_masked == 0:
                    continue

                correct1 += (preds[mask] == labels[mask]).sum().item()

                # for each masked position, did true label appear in top-5?
                correct5 += (
                    (top5[mask] == labels[mask].unsqueeze(-1))
                    .any(dim=-1)
                    .sum()
                    .item()
                )

                total_masked += n_masked
                        
        # aggregate
        # val_loss /= n_val_batches
        # val_ppl = float(torch.exp(torch.tensor(val_loss)))
        val_loss = val_loss_sum / n_val_batches if n_val_batches else float("nan")
        val_ppl  = math.exp(val_loss)
        acc1     = correct1 / total_masked
        acc5     = correct5 / total_masked

        # save metrics
        val_losses.append(val_loss)
        val_ppls.append(val_ppl)
        val_accs.append(acc1)                         
        val_top5.append(acc5)                        

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

        # Plot validation accuracy per epoch
        plt.figure()
        plt.plot(range(1, epoch+1), val_accs, marker='o')
        plt.title('Validation Masked-Token Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(save_path / 'val_acc_per_epoch.png')
        plt.close()

        # Plot top-5 accuracy per epoch
        plt.figure()
        plt.plot(range(1, epoch+1), val_top5, marker='o')
        plt.title('Validation Top-5 Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Top-5 Accuracy')
        plt.grid(True)
        plt.savefig(save_path / 'val_top5_per_epoch.png')
        plt.close()

        ckpt = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': lm_module.network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(ckpt, save_path / f'checkpoint_epoch{epoch}.pt')
        print(f"Saved checkpoint_epoch{epoch}.pt")
        
        print(f"Finished epoch {epoch}/{args.max_epochs}")
        # print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")
        print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}, "
              f"Acc: {acc1:.3f}, Top-5: {acc5:.3f}, LR: {last_lr:.2e}")


    # Final model save
    # torch.save(lm_module.network.state_dict(), save_path / 'final_model.pt')
    print("Training complete.")


if __name__ == '__main__':
    main()

# Using mixed precision produced no performance gains
# Using torch optimizations like kernel parameters and jit compiling the network: From ~3.3 it/s To ~3.7 it/s
# Dropping q/k/v biases: From ~3.7 it/s to ~4.1 it/s
