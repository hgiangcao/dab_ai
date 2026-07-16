"""
pretrained.py — Supervised Pretraining from Bot Game Logs

Reads game_logs_bot.jsonl (tournament games) and/or game_logs.jsonl (generate_logs.py)
and runs supervised learning on the network BEFORE AlphaZero self-play begins.

Data format per line (JSON):
    {
        "winner": int (+1, -1, or 0),
        "moves":  [int, ...],       # sequence of line indices played
        "policies": [[float, ...],  # one-hot or soft policy per move
                     ...]
    }

Pipeline:
  1. Load all game logs.
  2. Replay each game to extract (board_state, policy, value) at every move.
  3. Augment with 8-fold symmetry (same as AlphaZero training).
  4. Run AdamW training for pretrain_epochs epochs.
  5. Save to <run_dir>/pretrained.pth.tar and overwrite checkpoint_0 / best.pth.tar.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import config
import model_manager
from game import DotsAndBoxesGame
from model import NNetWrapper, dotdict


# ─────────────────────────────────────────────────────────────────────────────
# Pretrain hyperparameters (as recommended in the instruction doc)
# ─────────────────────────────────────────────────────────────────────────────
PRETRAIN_ARGS = dotdict({
    'lr':                 1e-3,        # AdamW LR (higher than AZ training LR)
    'epochs':             15,          # 10-20 epochs recommended
    'batch_size':         512,         # 256-512 recommended
    'num_channels':       256,
    'num_res_blocks':     10,
    'l2_reg':             1e-4,        # Weight decay
    'lr_scheduler_steps': 336,
    'device':             'cuda' if torch.cuda.is_available() else 'cpu'
})


# ─────────────────────────────────────────────────────────────────────────────
# Board encoding (identical to trainner.py / coach.py augment_data)
# ─────────────────────────────────────────────────────────────────────────────
def encode_board(game: DotsAndBoxesGame) -> np.ndarray:
    """
    Encode the current game state into 4-channel (C, H, W) tensor:
      ch0: horizontal line occupancy
      ch1: vertical line occupancy
      ch2: boxes owned by player 1
      ch3: boxes owned by player -1
    """
    lines = game.lines           # 1-D array of line states
    boxes = game.boxes           # 2-D (SIZE x SIZE) array of box owners

    h, v_mat = DotsAndBoxesGame.l_to_h_v(lines)
    size = game.SIZE

    c1 = np.zeros((size + 1, size + 1))
    c1[:size + 1, :size] = h

    c2 = np.zeros((size + 1, size + 1))
    c2[:size, :size + 1] = v_mat

    c3 = np.zeros((size + 1, size + 1))
    c3[:size, :size] = (boxes == 1).astype(float)

    c4 = np.zeros((size + 1, size + 1))
    c4[:size, :size] = (boxes == -1).astype(float)

    return np.stack([c1, c2, c3, c4])


# ─────────────────────────────────────────────────────────────────────────────
# Game log → training examples
# ─────────────────────────────────────────────────────────────────────────────
def game_record_to_examples(record: dict, game_size: int = 5):
    """
    Replay a single game record and extract (board_state, policy, value)
    for every move in the game.

    Args:
        record: dict with keys 'winner', 'moves', 'policies'
        game_size: board size (default 5)

    Returns:
        list of (board_state [np.ndarray], policy [np.ndarray], value [float])
    """
    moves    = record.get("moves", [])
    policies = record.get("policies", [])
    winner   = record.get("winner", 0)

    if not moves or len(moves) != len(policies):
        return []

    game = DotsAndBoxesGame(size=game_size)
    examples = []

    for move, policy in zip(moves, policies):
        # Encode current board state BEFORE executing the move
        board_state = encode_board(game)
        pi = np.array(policy, dtype=np.float32)

        # Value: final result from the perspective of the CURRENT player
        # game.current_player is +1 or -1
        value = float(winner) * game.current_player

        examples.append((board_state, pi, value))

        # Execute the move to advance game state
        if move in game.get_valid_moves():
            game.execute_move(move)
        else:
            break  # Corrupted record — skip remaining moves

    return examples


def augment_examples(examples):
    """
    Apply 8-fold symmetry augmentation (same as AlphaZero training).
    Input:  list of (board_state, policy, value)
    Output: augmented list of same format
    """
    augmented = []
    for board_state, pi, value in examples:
        # board_state shape: (4, size+1, size+1)
        # Recover lines representation from board channels
        size = board_state.shape[1] - 1
        n_lines = 2 * size * (size + 1)

        # Reconstruct lines array from h/v channels
        h_ch  = board_state[0, :size + 1, :size]    # (size+1, size)
        v_ch  = board_state[1, :size, :size + 1]    # (size, size+1)
        lines = DotsAndBoxesGame.h_v_to_l(h_ch, v_ch)

        boxes_p1 = board_state[2, :size, :size]
        boxes_m1 = board_state[3, :size, :size]
        boxes = boxes_p1.astype(float) - boxes_m1.astype(float)

        pi_lines = pi  # already in line-index space

        for aug_lines, aug_boxes, aug_pi in zip(
            DotsAndBoxesGame.get_rotations_and_reflections_lines(lines),
            DotsAndBoxesGame.get_rotations_and_reflections_boxes(boxes),
            DotsAndBoxesGame.get_rotations_and_reflections_lines(pi_lines)
        ):
            h, v_mat = DotsAndBoxesGame.l_to_h_v(aug_lines)

            c1 = np.zeros((size + 1, size + 1))
            c1[:size + 1, :size] = h

            c2 = np.zeros((size + 1, size + 1))
            c2[:size, :size + 1] = v_mat

            c3 = np.zeros((size + 1, size + 1))
            c3[:size, :size] = (aug_boxes == 1).astype(float)

            c4 = np.zeros((size + 1, size + 1))
            c4[:size, :size] = (aug_boxes == -1).astype(float)

            aug_board = np.stack([c1, c2, c3, c4])
            augmented.append((aug_board, aug_pi.astype(np.float32), value))

    return augmented


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────
def load_examples_from_jsonl(filepath: str, game_size: int = 5):
    """Load and convert all game records from a .jsonl file into training examples."""
    if not os.path.exists(filepath):
        return []

    examples = []
    with open(filepath, "r") as f:
        for line in tqdm(f, desc=f"Loading {os.path.basename(filepath)}"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                examples.extend(game_record_to_examples(record, game_size))
            except Exception:
                continue

    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Supervised training loop
# ─────────────────────────────────────────────────────────────────────────────
def pretrain(nnet: NNetWrapper, examples, args=PRETRAIN_ARGS, writer=None, start_step=0):
    """
    Run supervised pretraining on the provided examples.

    Args:
        nnet:       NNetWrapper instance (will be trained in-place)
        examples:   list of (board_state, policy, value)
        args:       pretrain hyperparameters
        writer:     optional TensorBoard SummaryWriter
        start_step: global step offset for TensorBoard

    Returns:
        Final (pi_loss, v_loss, total_loss) averages over the last epoch
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nnet.nnet.to(device)

    # Use a separate AdamW optimizer for pretraining (not tied to the AZ scheduler)
    optimizer = optim.AdamW(
        nnet.nnet.parameters(),
        lr=args.lr,
        weight_decay=args.l2_reg
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    batch_size  = args.batch_size
    n_examples  = len(examples)
    global_step = start_step

    last_pi, last_v, last_total = 0.0, 0.0, 0.0

    for epoch in range(args.epochs):
        nnet.nnet.train()
        np.random.shuffle(examples)

        n_batches = max(1, n_examples // batch_size)
        pi_losses, v_losses, total_losses = [], [], []

        bar = tqdm(range(n_batches), desc=f"Pretrain Epoch {epoch + 1}/{args.epochs}")
        for _ in bar:
            idx = np.random.randint(n_examples, size=batch_size)
            boards, pis, vs = zip(*[examples[i] for i in idx])

            boards_t = torch.FloatTensor(np.array(boards).astype(np.float32)).to(device)
            pis_t    = torch.FloatTensor(np.array(pis)).to(device)
            vs_t     = torch.FloatTensor(np.array(vs).astype(np.float32)).to(device)

            out_pi, out_v = nnet.nnet(boards_t)

            # Policy: cross-entropy (model outputs log-softmax)
            l_pi = -torch.sum(pis_t * out_pi) / pis_t.size(0)
            # Value: MSE
            l_v  = torch.sum((vs_t - out_v.view(-1)) ** 2) / vs_t.size(0)
            loss = l_pi + l_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pi_losses.append(l_pi.item())
            v_losses.append(l_v.item())
            total_losses.append(loss.item())
            global_step += 1

            bar.set_postfix(Loss_pi=f"{l_pi.item():.4f}", Loss_v=f"{l_v.item():.4f}")

        avg_pi    = sum(pi_losses)    / len(pi_losses)
        avg_v     = sum(v_losses)     / len(v_losses)
        avg_total = sum(total_losses) / len(total_losses)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  [Pretrain] Epoch {epoch + 1}/{args.epochs} | "
              f"pi={avg_pi:.4f} | v={avg_v:.4f} | total={avg_total:.4f} | lr={lr_now:.2e}")

        if writer:
            writer.add_scalar('Pretrain/Policy_Loss', avg_pi,    epoch)
            writer.add_scalar('Pretrain/Value_Loss',  avg_v,     epoch)
            writer.add_scalar('Pretrain/Total_Loss',  avg_total, epoch)
            writer.add_scalar('Pretrain/LR',          lr_now,    epoch)

        scheduler.step()

        last_pi, last_v, last_total = avg_pi, avg_v, avg_total

    return last_pi, last_v, last_total


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point — called from trainner.py
# ─────────────────────────────────────────────────────────────────────────────
def run_pretraining(nnet: NNetWrapper, run_dir: str, writer=None, game_size: int = 5):
    """
    Full pretraining pipeline:
      1. Check if pretrained.pth.tar already exists — skip if so.
      2. Load game_logs_bot.jsonl and game_logs.jsonl.
      3. Augment with 8-fold symmetry.
      4. Train supervised.
      5. Save to <run_dir>/pretrained.pth.tar AND overwrite checkpoint_0 + best.pth.tar.

    Args:
        nnet:      NNetWrapper to pretrain (modified in-place)
        run_dir:   path to the current experiment log directory
        writer:    optional TensorBoard SummaryWriter
        game_size: board size (default 5)

    Returns:
        True if pretraining was performed, False if skipped.
    """
    pretrained_path = os.path.join(run_dir, "pretrained.pth.tar")

    if os.path.exists(pretrained_path):
        # Pretraining already done in a previous run.
        # Do NOT reload pretrained weights — the caller already loaded the latest
        # AZ checkpoint (which is more up-to-date than the pretrained snapshot).
        print(f"[Pretrain] pretrained.pth.tar found. Pretraining already completed — skipping.")
        return False

    # ── Collect data from all available log files ──────────────────────────
    all_examples = []

    bot_log  = os.path.join(PROJECT_ROOT, "game_logs_bot.jsonl")
    rand_log = os.path.join(PROJECT_ROOT, "game_logs.jsonl")

    for logfile in [bot_log, rand_log]:
        exs = load_examples_from_jsonl(logfile, game_size)
        if exs:
            print(f"[Pretrain] Loaded {len(exs):,} raw examples from {os.path.basename(logfile)}")
            all_examples.extend(exs)

    if not all_examples:
        print("[Pretrain] No game log files found (game_logs_bot.jsonl / game_logs.jsonl). Skipping pretraining.")
        return False

    print(f"[Pretrain] Total raw examples: {len(all_examples):,}")

    # ── Augment ────────────────────────────────────────────────────────────
    try:
        augmented = augment_examples(all_examples)
        print(f"[Pretrain] After 8-fold augmentation: {len(augmented):,} examples")
    except Exception as e:
        print(f"[Pretrain] Augmentation failed ({e}), using raw examples.")
        augmented = all_examples

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\n[Pretrain] Starting supervised pretraining on {len(augmented):,} examples "
          f"for {PRETRAIN_ARGS.epochs} epochs...")
    pi_loss, v_loss, total_loss = pretrain(nnet, augmented, PRETRAIN_ARGS, writer)
    print(f"[Pretrain] Done. Final losses — pi: {pi_loss:.4f} | v: {v_loss:.4f} | total: {total_loss:.4f}")

    # ── Save to all checkpoint locations so every component starts from pretrained weights ──
    os.makedirs(run_dir, exist_ok=True)
    state = {'state_dict': nnet.nnet.state_dict()}

    # 1. pretrained.pth.tar — existence marker so we don't redo pretraining on restart
    torch.save(state, pretrained_path)
    print(f"[Pretrain] Saved pretrained weights → {pretrained_path}")

    # 2. Overwrite checkpoint_0.pth.tar so server serves these weights
    ckpt0_path = os.path.join(run_dir, "checkpoint_0.pth.tar")
    torch.save(state, ckpt0_path)
    print(f"[Pretrain] Overwritten checkpoint_0 → {ckpt0_path}")

    # 3. Overwrite best.pth.tar so workers immediately start from pretrained weights
    best_path = os.path.join(run_dir, "best.pth.tar")
    torch.save(state['state_dict'], best_path)
    print(f"[Pretrain] Overwritten best model → {best_path}")

    # 4. Save checkpoint_candidate.pth.tar for the first evaluation step
    candidate_path = os.path.join(run_dir, "checkpoint_candidate.pth.tar")
    torch.save(state, candidate_path)
    print(f"[Pretrain] Saved candidate model → {candidate_path}")

    return True
