"""
train_il.py
===========
Phase 1 training: Imitation Learning (IL) from the Kaggle replay dataset.

WHAT THIS DOES (Plain English)
-------------------------------
We load the (state, action) pairs from parse_replays.py and train our neural
network to COPY what the good bots did. This is just supervised learning —
like training an image classifier, but instead of "cat vs dog" we're learning
"given this game state, which actions did the winning bot take?"

Loss function: Cross-entropy per planet slot.
  For each of the MAX_MY_PLANETS planet slots, the network predicts a
  probability distribution over NUM_ACTIONS_PER_PLANET actions.
  We penalise it for giving low probability to the action the expert took.

After this phase the bot won't be brilliant, but it will:
  - Know to send fleets to neutral planets (not random directions)
  - Know to attack weak enemies
  - Have a head start for RL fine-tuning

TRAINING TIPS (already baked in)
---------------------------------
- Mixed precision (fp16) — ~2× faster on MI300X with no accuracy loss.
- Cosine LR schedule — learning rate warms up then decays smoothly.
- Early stopping — saves the best checkpoint, stops if val loss stagnates.
- Gradient clipping — prevents exploding gradients.

USAGE
-----
  python train_il.py --data_dir ./il_data --out_dir ./checkpoints

  Or with options:
  python train_il.py \
      --data_dir    ./il_data        \
      --out_dir     ./checkpoints    \
      --epochs      30               \
      --batch_size  512              \
      --lr          3e-4             \
      --hidden_dim  512
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

# Allow imports from agent/ whether run from repo root or training/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AGENT_DIR = os.path.join(_REPO_ROOT, "agent")
for _p in [_AGENT_DIR, _REPO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model        import OrbitWarsNet, save_model
from action_utils import MAX_MY_PLANETS, NUM_ACTIONS_PER_PLANET
from feature_utils import FEATURE_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class ReplayDataset(Dataset):
    """
    Loads pre-parsed IL data from parse_replays.py.

    states  : (N, FEATURE_DIM)  float32
    actions : (N, MAX_MY_PLANETS) int32
    """
    def __init__(self, data_dir):
        states_path  = os.path.join(data_dir, "states.npy")
        actions_path = os.path.join(data_dir, "actions.npy")

        if not os.path.exists(states_path):
            raise FileNotFoundError(
                f"states.npy not found in {data_dir}\n"
                f"Run parse_replays.py first:\n"
                f"  python parse_replays.py --replay_dir ./replays --out_dir {data_dir}"
            )

        self.states  = torch.from_numpy(np.load(states_path)).float()
        self.actions = torch.from_numpy(np.load(actions_path)).long()

        print(f"[dataset] Loaded {len(self.states):,} samples from {data_dir}")
        print(f"[dataset] states  : {self.states.shape}   {self.states.element_size() * self.states.nelement() / 1e6:.1f} MB")
        print(f"[dataset] actions : {self.actions.shape}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_il(
    data_dir   = os.path.join(_REPO_ROOT, "il_data"),
    out_dir    = os.path.join(_REPO_ROOT, "checkpoints"),
    epochs     = 30,
    batch_size = 512,
    lr         = 3e-4,
    hidden_dim = 512,
    num_layers = 3,
    val_frac   = 0.05,         # 5% of data held out for validation
    patience   = 5,            # early stopping patience (epochs)
    grad_clip  = 0.5,
    num_workers= 4,
    log_every  = 100,          # log every N batches
):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[IL] Device          : {device}")
    if device == "cuda":
        print(f"[IL] GPU             : {torch.cuda.get_device_name(0)}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset  = ReplayDataset(data_dir)
    val_size = max(1, int(len(dataset) * val_frac))
    trn_size = len(dataset) - val_size
    trn_ds, val_ds = random_split(dataset, [trn_size, val_size])

    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=(device=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=num_workers, pin_memory=(device=="cuda"))

    print(f"[IL] Train samples   : {trn_size:,}")
    print(f"[IL] Val   samples   : {val_size:,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = OrbitWarsNet(
        hidden_dim    = hidden_dim,
        num_layers    = num_layers,
        dropout       = 0.1,
        max_my_planets= MAX_MY_PLANETS,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[IL] Model params    : {total_params:,}")

    # ── Optimizer & schedule ──────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_steps = epochs * len(trn_loader)
    warmup_steps= min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    import math
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Loss: cross-entropy averaged over planet slots ────────────────────────
    # For each sample we have MAX_MY_PLANETS slots, each with a target action.
    # We compute CE for each slot and average.
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)   # -1 = masked/invalid slot

    # ── Mixed precision ───────────────────────────────────────────────────────
    use_amp = (device == "cuda")
    scaler  = GradScaler(enabled=use_amp)

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_ctr  = 0
    global_step   = 0
    best_ckpt     = os.path.join(out_dir, "il_best.pt")

    print(f"\n[IL] Starting training  ({epochs} epochs)\n")

    for epoch in range(1, epochs + 1):
        model.train()
        trn_loss_sum = 0.0
        trn_batches  = 0
        t0 = time.time()

        for batch_states, batch_actions in trn_loader:
            batch_states  = batch_states.to(device)
            batch_actions = batch_actions.to(device)   # (B, MAX_MY_PLANETS)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                logits, _ = model(batch_states)
                # logits: (B, MAX_MY_PLANETS, NUM_ACTIONS)
                # CE expects (B, C, ...) format
                # Reshape: (B * MAX_MY_PLANETS, NUM_ACTIONS) vs (B * MAX_MY_PLANETS,)
                B = batch_states.shape[0]
                logits_flat  = logits.view(B * MAX_MY_PLANETS, NUM_ACTIONS_PER_PLANET)
                targets_flat = batch_actions.view(B * MAX_MY_PLANETS)
                loss = ce_loss(logits_flat, targets_flat)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            trn_loss_sum += loss.item()
            trn_batches  += 1
            global_step  += 1

            if global_step % log_every == 0:
                avg = trn_loss_sum / trn_batches
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step {global_step:6d} | loss {avg:.4f} | lr {lr_now:.2e}")

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_batches  = 0
        correct_total= 0
        total_slots  = 0

        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                batch_states  = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                B = batch_states.shape[0]

                with autocast(enabled=use_amp):
                    logits, _ = model(batch_states)
                    logits_flat  = logits.view(B * MAX_MY_PLANETS, NUM_ACTIONS_PER_PLANET)
                    targets_flat = batch_actions.view(B * MAX_MY_PLANETS)
                    val_loss = ce_loss(logits_flat, targets_flat)

                val_loss_sum += val_loss.item()
                val_batches  += 1

                # Top-1 accuracy
                preds = logits_flat.argmax(dim=-1)
                # Don't count hold (0) slots — they're trivially easy and inflate accuracy
                non_hold = (targets_flat > 0)
                correct_total += (preds[non_hold] == targets_flat[non_hold]).sum().item()
                total_slots   += non_hold.sum().item()

        avg_trn = trn_loss_sum / max(1, trn_batches)
        avg_val = val_loss_sum  / max(1, val_batches)
        acc     = correct_total / max(1, total_slots) * 100
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{epochs}  |  "
              f"trn_loss {avg_trn:.4f}  |  "
              f"val_loss {avg_val:.4f}  |  "
              f"acc {acc:.1f}%  |  "
              f"{elapsed:.0f}s")

        # ── Checkpoint / early stop ───────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_ctr  = 0
            save_model(model, best_ckpt, metadata={
                "epoch"        : epoch,
                "val_loss"     : avg_val,
                "hidden_dim"   : hidden_dim,
                "num_layers"   : num_layers,
            })
            print(f"  ✔ New best checkpoint saved (val_loss={avg_val:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n[IL] Early stopping triggered after {patience} epochs without improvement.")
                break

    # Save final checkpoint regardless
    final_ckpt = os.path.join(out_dir, "il_final.pt")
    save_model(model, final_ckpt, metadata={"epoch": epoch, "final": True})

    print(f"\n[IL] Training complete.")
    print(f"[IL] Best val loss   : {best_val_loss:.4f}")
    print(f"[IL] Best checkpoint : {best_ckpt}")
    print(f"[IL] Final checkpoint: {final_ckpt}")
    print(f"\nNext step: run train_ppo.py --init_checkpoint {best_ckpt}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imitation Learning from Orbit Wars replays")
    parser.add_argument("--data_dir",    default="./il_data",      help="IL dataset directory")
    parser.add_argument("--out_dir",     default="./checkpoints",  help="Checkpoint output dir")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--hidden_dim",  type=int,   default=512)
    parser.add_argument("--num_layers",  type=int,   default=3)
    parser.add_argument("--patience",    type=int,   default=5)
    parser.add_argument("--num_workers", type=int,   default=4)
    args = parser.parse_args()

    train_il(
        data_dir   = args.data_dir,
        out_dir    = args.out_dir,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        hidden_dim = args.hidden_dim,
        num_layers = args.num_layers,
        patience   = args.patience,
        num_workers= args.num_workers,
    )
