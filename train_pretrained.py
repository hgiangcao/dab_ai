"""
train_pretrained.py — Standalone Supervised Pretraining from Bot Game Logs
===========================================================================

Trains the AlphaZero network on game_logs_bot.jsonl ONLY (no self-play data).
Mirrors the supervised training loop in distributed/pretrained.py but adds:
  • Per-epoch TensorBoard logging (same scalar tags as distributed/trainner.py)
  • Post-epoch evaluation: 100 games of AlphaZero(0 sims) vs UCLABot_v3 / Greedy
  • Early stopping: stops when evaluation score has not improved for 10 epochs

Usage:
    python train_pretrained.py
    python train_pretrained.py --epochs 30 --lr 1e-3 --batch_size 4096
    python train_pretrained.py --load best.pth.tar --epochs 20
    python train_pretrained.py --log_dir logs/pretrain_run1
"""

import os
import sys
import random
import argparse
import time
import multiprocessing as mp

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "distributed"))

import config
from game import DotsAndBoxesGame
from model import NNetWrapper, dotdict
from dataset import DotsAndBoxesDataset
from distributed.pretrained import load_examples_from_jsonl


# ── Hyperparameters ──────────────────────────────────────────────────────────
# Lite network: 128 ch × 5 blocks vs full 256 ch × 10 blocks.
# FLOPs ∝ C² × blocks → (128²×5)/(256²×10) ≈ 1/8 FLOPs → 4–6× wall-time
# speedup (3–5× target).  NOTE: checkpoints from this script are NOT
# weight-compatible with full-size best.pth.tar (different architecture).
DEFAULT_ARGS = dotdict({
    "lr":                 1e-3,
    "epochs":             1,
    "batch_size":         4096,
    "num_channels":       128,   # ↓ from 256  (4–6× faster inference)
    "num_res_blocks":     5,     # ↓ from 10
    "l2_reg":             1e-4,
    "lr_scheduler_steps": 336,   # kept for NNetWrapper scheduler compat
    "device":             "cuda" if torch.cuda.is_available() else "cpu",
})

# Evaluation: games per opponent per evaluation round
EVAL_GAMES_PER_OPPONENT = 200

# Early-stopping patience: stop after this many evaluations with no improvement
EARLY_STOP_PATIENCE = 10

# Opponents used for evaluation (names must match create_eval_agent() below)
EVAL_OPPONENTS = ["UCLABot_v3", "Greedy"]


# ── Evaluation agent factory ─────────────────────────────────────────────────
def create_eval_agent(name: str):
    """Return an evaluation bot by name (non-AlphaZero bots only)."""
    if name == "Greedy":
        from bots.greedy import GreedyPlayer
        return GreedyPlayer(name=name)
    elif name == "UCLABot_v3":
        from bots.ucla_bot import UCLABot_v3
        return UCLABot_v3(name=name)
    elif name == "SimpleBot":
        from bots.simple_bot import SimpleBot
        return SimpleBot(name=name)
    elif name == "UCLABot_v6":
        from bots.ucla_bot_v6 import UCLABot_v6
        return UCLABot_v6(name=name)
    else:
        raise ValueError(f"Unknown eval opponent: {name}")


# ── AlphaZero pure-NN agent (0 MCTS simulations = direct policy) ─────────────
class AlphaZeroPolicyAgent:
    """Wraps NNetWrapper: selects argmax of the raw NN policy (no MCTS)."""

    def __init__(self, nnet: NNetWrapper, name: str = "AZ-Policy"):
        self.nnet = nnet
        self.name = name

    def reset(self):
        pass  # stateless

    def get_move(self, game: DotsAndBoxesGame) -> int:
        # Encode current state into 4-channel board tensor
        size = game.SIZE
        sp1 = size + 1
        buf = np.zeros((4, sp1, sp1), dtype=np.float32)
        canonical_lines = game.get_canonical_lines()
        h, v = game.l_to_h_v(canonical_lines)
        buf[0, :sp1, :size] = h
        buf[1, :size, :sp1] = v
        canonical_boxes = game.get_canonical_boxes()
        buf[2, :size, :size] = np.where(canonical_boxes == 1,  1.0, 0.0)
        buf[3, :size, :size] = np.where(canonical_boxes == -1, 1.0, 0.0)

        policy, _ = self.nnet.predict(buf)

        # Mask to valid moves and pick argmax
        valid_moves = game.get_valid_moves()
        masked = np.zeros(game.N_LINES, dtype=np.float64)
        masked[valid_moves] = policy[valid_moves]
        total = masked.sum()
        if total > 0:
            masked /= total
        else:
            masked[valid_moves] = 1.0 / len(valid_moves)
        return int(np.argmax(masked))


# ── Single evaluation game (called in a process pool) ────────────────────────
def _run_eval_game(args):
    """
    Run one evaluation game.
    Returns (result, examples) where:
      result   : 1 (AZ win) | 0 (draw) | -1 (AZ loss)
      examples : list of (canonical_lines, canonical_boxes, one_hot_pi, value)
                 collected at every AZ move — ready to merge into training data.
    """
    model_path, opponent_name, game_index, size = args
    import torch
    torch.set_num_threads(1)

    from game import DotsAndBoxesGame
    from model import NNetWrapper, dotdict
    import config as _cfg

    eval_args = dotdict({
        "lr": _cfg.LEARNING_RATE,
        "epochs": _cfg.EPOCHS,
        "batch_size": _cfg.BATCH_SIZE,
        "num_channels": 128,   # lite: matches DEFAULT_ARGS for this script
        "num_res_blocks": 5,   # lite: 4–6× faster inference vs 256ch/10-block
        "l2_reg": 1e-4,
        "lr_scheduler_steps": 336,
        "device": "cpu",
    })

    dummy = DotsAndBoxesGame(size=size)
    nnet = NNetWrapper(dummy, eval_args)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    weights = state.get("state_dict", state)
    nnet.nnet.load_state_dict(weights)
    nnet.nnet.eval()

    az_agent = AlphaZeroPolicyAgent(nnet, name="AZ-Policy")
    opp_agent = create_eval_agent(opponent_name)

    # Alternate who goes first: even index → AZ is P1, odd → AZ is P2
    az_is_p1 = (game_index % 2 == 0)
    az_player = 1 if az_is_p1 else -1
    game = DotsAndBoxesGame(size=size, starting_player=1)

    # Collect (lines_before_move, boxes_before_move, move_played) for AZ turns
    pending = []  # (canonical_lines, canonical_boxes, move_index)

    while game.is_running():
        is_az_turn = (game.current_player == 1) == az_is_p1
        agent = az_agent if is_az_turn else opp_agent

        if is_az_turn:
            # Snapshot board state in canonical form BEFORE the move
            cl = game.get_canonical_lines().copy()
            cb = game.get_canonical_boxes().copy()

        move = agent.get_move(game)

        if is_az_turn:
            pending.append((cl, cb, move, game.N_LINES))

        game.execute_move(move)

    # Outcome from AZ perspective
    if game.result == az_player:
        outcome = 1
    elif game.result == 0:
        outcome = 0
    else:
        outcome = -1

    # Build training examples: one-hot policy on the move played, outcome as value
    examples = []
    for cl, cb, move_idx, n_lines in pending:
        pi = np.zeros(n_lines, dtype=np.float32)
        pi[move_idx] = 1.0
        value = float(outcome)   # from the canonical player's perspective
        examples.append((cl, cb, pi, value))

    return outcome, examples


# ── Evaluation round ─────────────────────────────────────────────────────────
def evaluate_model(
    model_path: str,
    nnet: NNetWrapper,
    n_games: int,
    size: int,
    n_workers: int,
    writer: SummaryWriter,
    epoch: int,
) -> tuple:
    """
    Play n_games against each EVAL_OPPONENTS using multiprocessing.

    For every AZ move the game data is captured and returned as training
    examples.  Policy prediction accuracy (top-1) is also computed: for each
    AZ position we check whether argmax(NN policy) == move actually played.

    Returns:
        (mean_win_rate: float, new_examples: list)
    """
    all_scores   = []
    all_examples = []  # training examples harvested from eval games

    # --- Policy-accuracy accumulators (run on the trainer process, no grad) ---
    nnet.nnet.eval()
    total_positions = 0
    correct_top1    = 0

    for opp_name in EVAL_OPPONENTS:
        tasks = [(model_path, opp_name, i, size) for i in range(n_games)]
        game_results  = []
        game_examples = []

        with mp.Pool(processes=n_workers) as pool:
            for outcome, exs in tqdm(
                pool.imap_unordered(_run_eval_game, tasks),
                total=n_games,
                desc=f"  Eval vs {opp_name}",
            ):
                game_results.append(outcome)
                game_examples.extend(exs)

        wins   = sum(1 for r in game_results if r == 1)
        draws  = sum(1 for r in game_results if r == 0)
        losses = sum(1 for r in game_results if r == -1)
        wr = (wins + 0.5 * draws) / n_games

        print(
            f"  Eval vs {opp_name}: W={wins} D={draws} L={losses} "
            f"WR={wr:.3f}  |  {len(game_examples)} positions collected"
        )
        if writer:
            writer.add_scalar(f"Eval/WinRate_vs_{opp_name}", wr, epoch)

        all_scores.append(wr)
        all_examples.extend(game_examples)

        # -- Policy prediction accuracy (top-1) --------------------------------
        # For each captured AZ position: does argmax(NN policy) == move played?
        device = next(nnet.nnet.parameters()).device
        from dataset import DotsAndBoxesDataset
        from torch.utils.data import DataLoader as _DL
        if game_examples:
            tmp_ds = DotsAndBoxesDataset(game_examples)
            tmp_dl = _DL(tmp_ds, batch_size=512, shuffle=False,
                         num_workers=0, drop_last=False)
            with torch.no_grad():
                for boards_t, pis_t, _ in tmp_dl:
                    boards_t = boards_t.to(device)
                    pis_t    = pis_t.to(device)          # one-hot targets
                    out_pi, _ = nnet.nnet(boards_t)       # log-softmax
                    pred_move   = out_pi.argmax(dim=1)    # NN's top choice
                    actual_move = pis_t.argmax(dim=1)     # move actually played
                    correct_top1    += (pred_move == actual_move).sum().item()
                    total_positions += boards_t.size(0)

    mean_wr  = float(np.mean(all_scores))
    top1_acc = correct_top1 / total_positions if total_positions > 0 else 0.0

    print(
        f"  [Eval] Policy top-1 accuracy over {total_positions} AZ positions: "
        f"{top1_acc:.4f} ({correct_top1}/{total_positions})"
    )

    if writer:
        writer.add_scalar("Eval/Mean_WinRate",            mean_wr,        epoch)
        writer.add_scalar("Eval/Policy_TopK_Accuracy",    top1_acc,       epoch)
        writer.add_scalar("Eval/Positions_Evaluated",     total_positions, epoch)
        writer.flush()

    return mean_wr, all_examples


# ── Per-epoch supervised training ─────────────────────────────────────────────
def train_one_epoch(
    nnet: NNetWrapper,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple:
    """Train for one epoch. Returns (avg_pi_loss, avg_v_loss, avg_total, avg_entropy)."""
    nnet.nnet.train()
    pi_losses, v_losses, total_losses, entropies = [], [], [], []

    bar = tqdm(dataloader, desc="  Training")
    for boards_t, pis_t, vs_t in bar:
        boards_t = boards_t.to(device, non_blocking=True)
        pis_t    = pis_t.to(device, non_blocking=True)
        vs_t     = vs_t.to(device, non_blocking=True)

        out_pi, out_v = nnet.nnet(boards_t)

        # Cross-entropy policy loss (model outputs log-softmax)
        l_pi = -torch.sum(pis_t * out_pi) / pis_t.size(0)
        # MSE value loss
        l_v  = torch.sum((vs_t - out_v.view(-1)) ** 2) / vs_t.size(0)
        loss = l_pi + l_v

        # Entropy of predicted policy distribution
        probs   = torch.exp(out_pi)
        entropy = -torch.sum(probs * out_pi) / probs.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pi_losses.append(l_pi.item())
        v_losses.append(l_v.item())
        total_losses.append(loss.item())
        entropies.append(entropy.item())

        bar.set_postfix(pi=f"{l_pi.item():.4f}", v=f"{l_v.item():.4f}", ent=f"{entropy.item():.4f}")

    n = len(pi_losses)
    return (
        sum(pi_losses)   / n,
        sum(v_losses)    / n,
        sum(total_losses)/ n,
        sum(entropies)   / n,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Supervised pretraining on bot game logs.")
    parser.add_argument("--log_file",   type=str,   default="game_logs_bot.jsonl",
                        help="Path to .jsonl bot log file (default: game_logs_bot.jsonl).")
    parser.add_argument("--load",       type=str,   default=None,
                        help="Checkpoint path to start from (default: random init).")
    parser.add_argument("--save_dir",   type=str,   default=None,
                        help="Directory to save epoch checkpoints (default: logs/pretrain_<timestamp>/).")
    parser.add_argument("--log_dir",    type=str,   default=None,
                        help="TensorBoard log directory (defaults to save_dir).")
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_ARGS.epochs)
    parser.add_argument("--lr",         type=float, default=DEFAULT_ARGS.lr)
    parser.add_argument("--batch_size", type=int,   default=DEFAULT_ARGS.batch_size)
    parser.add_argument("--size",       type=int,   default=5,
                        help="Board size (default: 5).")
    parser.add_argument("--eval_games", type=int,   default=EVAL_GAMES_PER_OPPONENT,
                        help="Games per opponent per evaluation round (default: 50).")
    parser.add_argument("--patience",   type=int,   default=EARLY_STOP_PATIENCE,
                        help="Early-stop after N epochs with no eval improvement (default: 10).")
    parser.add_argument("--workers",    type=int,   default=None,
                        help="Parallel workers for evaluation (default: cpu_count - 1).")
    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_dir or os.path.join(PROJECT_ROOT, "logs", f"pretrain_{timestamp}")
    log_dir  = args.log_dir  or save_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    log_file = args.log_file if os.path.isabs(args.log_file) else \
               os.path.join(PROJECT_ROOT, args.log_file)

    n_workers = args.workers if args.workers is not None else max(1, mp.cpu_count() - 1)

    print("=" * 60)
    print("Supervised Pretraining from Bot Logs")
    print("=" * 60)
    print(f"  Log file   : {log_file}")
    print(f"  Save dir   : {save_dir}")
    print(f"  TensorBoard: {log_dir}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  LR         : {args.lr}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Eval games : {args.eval_games} per opponent ({EVAL_OPPONENTS})")
    print(f"  Patience   : {args.patience} epochs")
    print(f"  Workers    : {n_workers}")
    print()

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=log_dir)

    # ── Load dataset ──────────────────────────────────────────────────────────
    if not os.path.exists(log_file):
        print(f"ERROR: Log file not found: {log_file}")
        sys.exit(1)

    print(f"Loading examples from {os.path.basename(log_file)} ...")
    raw_examples = load_examples_from_jsonl(log_file, game_size=args.size)
    if not raw_examples:
        print("ERROR: No examples loaded. Aborting.")
        sys.exit(1)

    print(f"Loaded {len(raw_examples):,} raw examples.")
    dataset = DotsAndBoxesDataset(raw_examples)
    print(f"Dataset size after 8-fold augmentation: {len(dataset):,} samples.\n")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,          # 0 avoids /dev/shm overflow; dataset is cheap
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    train_args = dotdict({
        "lr":                 args.lr,
        "epochs":             args.epochs,
        "batch_size":         args.batch_size,
        "num_channels":       DEFAULT_ARGS.num_channels,
        "num_res_blocks":     DEFAULT_ARGS.num_res_blocks,
        "l2_reg":             DEFAULT_ARGS.l2_reg,
        "lr_scheduler_steps": DEFAULT_ARGS.lr_scheduler_steps,
        "device":             DEFAULT_ARGS.device,
    })

    dummy_game = DotsAndBoxesGame(size=args.size)
    nnet = NNetWrapper(dummy_game, train_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nnet.nnet.to(device)

    if args.load:
        load_path = args.load if os.path.isabs(args.load) else \
                    os.path.join(PROJECT_ROOT, args.load)
        if os.path.exists(load_path):
            state = torch.load(load_path, map_location="cpu", weights_only=False)
            weights = state.get("state_dict", state)
            nnet.nnet.load_state_dict(weights)
            print(f"Loaded weights from {load_path}\n")
        else:
            print(f"WARNING: --load path not found ({load_path}). Starting from random weights.\n")
    else:
        print("Starting from random weight initialization.\n")

    # ── Optimiser + LR scheduler (independent of the AZ cosine scheduler) ────
    optimizer = optim.AdamW(
        nnet.nnet.parameters(),
        lr=args.lr,
        weight_decay=DEFAULT_ARGS.l2_reg,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-4
    )

    # ── Training + evaluation loop ────────────────────────────────────────────
    best_score      = -1.0
    no_improve_cnt  = 0
    best_ckpt_path  = os.path.join(save_dir, "best.pth.tar")

    # Temp weights file passed to worker processes for evaluation
    _eval_tmp_path = os.path.join(save_dir, "_eval_tmp.pth.tar")

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        lr_now = optimizer.param_groups[0]["lr"]

        # ── Supervised training pass ──────────────────────────────────────────
        n_train_raw = len(raw_examples)
        n_train_aug = len(dataset)     # after 8-fold symmetry
        print(f"  [Data] {n_train_raw:,} raw examples → {n_train_aug:,} augmented samples")

        avg_pi, avg_v, avg_total, avg_ent = train_one_epoch(
            nnet, dataloader, optimizer, device
        )
        scheduler.step()

        print(
            f"  [Train] pi={avg_pi:.4f} | v={avg_v:.4f} | "
            f"total={avg_total:.4f} | ent={avg_ent:.4f} | lr={lr_now:.2e}"
        )

        # TensorBoard — same tag names as distributed/trainner.py
        writer.add_scalar("Pretrain/Policy_Loss",    avg_pi,    epoch)
        writer.add_scalar("Pretrain/Value_Loss",     avg_v,     epoch)
        writer.add_scalar("Pretrain/Total_Loss",     avg_total, epoch)
        writer.add_scalar("Pretrain/Policy_Entropy", avg_ent,   epoch)
        writer.add_scalar("Pretrain/LR",             lr_now,    epoch)
        writer.add_scalar("Data/Total_Training_Samples_Raw", n_train_raw, epoch)
        writer.add_scalar("Data/Total_Training_Samples_Aug", n_train_aug, epoch)

        # Save per-epoch checkpoint
        epoch_path = os.path.join(save_dir, f"pretrain_epoch_{epoch:03d}.pth.tar")
        torch.save({"state_dict": nnet.nnet.state_dict()}, epoch_path)
        print(f"  [Save] {os.path.basename(epoch_path)}")

        # ── Evaluation ───────────────────────────────────────────────────────
        print(f"\n  [Eval] {args.eval_games} games per opponent: {EVAL_OPPONENTS}")
        nnet.nnet.eval()

        # Dump weights to temp file so worker processes can load them
        torch.save({"state_dict": nnet.nnet.state_dict()}, _eval_tmp_path)

        score, eval_examples = evaluate_model(
            model_path=_eval_tmp_path,
            nnet=nnet,
            n_games=args.eval_games,
            size=args.size,
            n_workers=n_workers,
            writer=writer,
            epoch=epoch,
        )
        print(f"  [Eval] Mean WinRate = {score:.4f}  (best = {max(best_score, 0.0):.4f})")

        # ── Merge eval examples into training data ────────────────────────────
        if eval_examples:
            raw_examples.extend(eval_examples)
            dataset   = DotsAndBoxesDataset(raw_examples)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
            )
            print(
                f"  [Data] Added {len(eval_examples):,} eval examples → "
                f"total raw: {len(raw_examples):,} | augmented: {len(dataset):,}"
            )

        # ── Early stopping ────────────────────────────────────────────────────
        if score > best_score:
            best_score     = score
            no_improve_cnt = 0
            torch.save({"state_dict": nnet.nnet.state_dict()}, best_ckpt_path)
            print(f"  [Best] Score improved to {best_score:.4f} → saved {os.path.basename(best_ckpt_path)}")
        else:
            no_improve_cnt += 1
            print(
                f"  [Early Stop] No improvement for {no_improve_cnt}/{args.patience} evaluations."
            )
            if no_improve_cnt >= args.patience:
                print(
                    f"\nEarly stopping triggered after epoch {epoch} "
                    f"({args.patience} consecutive evaluations without improvement)."
                )
                break

    # ── Final summary ─────────────────────────────────────────────────────────
    writer.close()
    if os.path.exists(_eval_tmp_path):
        os.remove(_eval_tmp_path)

    print("\n" + "=" * 60)
    print("Pretraining complete.")
    print(f"  Best eval score : {best_score:.4f}")
    print(f"  Best checkpoint : {best_ckpt_path}")
    print(f"  All checkpoints : {save_dir}")
    print(f"  TensorBoard     : tensorboard --logdir {log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    mp.freeze_support()
    main()
