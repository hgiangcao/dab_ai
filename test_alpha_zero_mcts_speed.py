"""
test_alpha_zero_mcts_speed.py
─────────────────────────────
Grid-search benchmark comparing AlphaZero inference speed across:

  • Model configurations : (num_channels × num_res_blocks)
  • MCTS simulation counts: [0, 50, 100, 200, 500]
  • Game stages           : first move (0 pct), mid-game (40 pct), end-game (80 pct filled)

Usage:
    python test_alpha_zero_mcts_speed.py
    python test_alpha_zero_mcts_speed.py --repeats 20 --csv results.csv
    python test_alpha_zero_mcts_speed.py --model_path best.pth.tar
"""

import os, sys, time, csv, argparse, random as _random
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from game import DotsAndBoxesGame
from model import NNetWrapper, dotdict
from mcts import MCTS
import config

# Grid axes
MODEL_CONFIGS = [
    (64,  3), (64,  5),
    (128, 3), (128, 5), (128, 8),
    (192, 5),
    (256, 5), (256, 10),
]
SIM_COUNTS = [0, 50, 100, 200, 500,1000]
BOARD_SIZE  = 5
TOTAL_LINES = 2 * BOARD_SIZE * (BOARD_SIZE + 1)
STAGES = [
    ("first_move", 0.00),
    ("mid_game",   0.40),
    ("end_game",   0.80),
]
DEFAULT_REPEATS = 10
DEFAULT_WARMUP  = 3


def build_nnet(ch, rb, device, model_path):
    dummy = DotsAndBoxesGame(size=BOARD_SIZE)
    args  = dotdict({"lr":1e-3,"epochs":1,"batch_size":512,
                     "num_channels":ch,"num_res_blocks":rb,
                     "l2_reg":1e-4,"lr_scheduler_steps":336,"device":device})
    nnet  = NNetWrapper(dummy, args)
    nnet.nnet.to(torch.device(device))
    if model_path and os.path.exists(model_path):
        try:
            state   = torch.load(model_path, map_location=device, weights_only=False)
            weights = state.get("state_dict", state)
            nnet.nnet.load_state_dict(weights, strict=True)
        except Exception:
            pass
    nnet.nnet.eval()
    return nnet


def get_stage_game(fill_frac, size=BOARD_SIZE, seed=42):
    n_moves = int(2 * size * (size + 1) * fill_frac)
    game = DotsAndBoxesGame(size=size, early_stopping=False)
    rng  = _random.Random(seed)
    for _ in range(n_moves):
        if not game.is_running(): break
        game.execute_move(rng.choice(game.get_valid_moves()))
    return game, n_moves


def count_nn_calls(mcts_obj):
    from collections import deque
    if mcts_obj._root is None: return 0
    visited = set(); q = deque([mcts_obj._root])
    visited.add(id(mcts_obj._root)); calls = 0
    while q:
        node = q.popleft()
        if node.P is not None: calls += 1
        for c in node.children.values():
            if id(c) not in visited:
                visited.add(id(c)); q.append(c)
    return calls


def measure(nnet, n_sims, game_state, repeats, warmup):
    params = {"n_simulations":n_sims,"c_puct":config.MCTS_C_PUCT,
              "dirichlet_eps":0.0,"dirichlet_alpha":config.MCTS_DIRICHLET_ALPHA}
    mcts_obj = MCTS(nnet, params)
    times = []; nn_calls = []
    for i in range(warmup + repeats):
        mcts_obj.reset_tree()
        gs = game_state.clone(track_history=False)
        t0 = time.perf_counter()
        mcts_obj.play(gs, temp=0, last_action=None)
        ms = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(ms)
            nn_calls.append(count_nn_calls(mcts_obj))
    mean_ms = float(np.mean(times))
    return {"mean_ms": mean_ms, "std_ms": float(np.std(times)),
            "ms_per_sim": mean_ms / n_sims if n_sims > 0 else mean_ms,
            "nn_calls": float(np.mean(nn_calls))}


def param_count(nnet):
    return sum(p.numel() for p in nnet.nnet.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats",    type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmup",     type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument("--model_path", type=str, default="best.pth.tar")
    parser.add_argument("--csv",        type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)

    total_lines = 2 * BOARD_SIZE * (BOARD_SIZE + 1)

    print("=" * 76)
    print("  AlphaZero MCTS Speed Grid-Search Benchmark")
    print("=" * 76)
    print(f"  Device : {device}  |  Board: {BOARD_SIZE}x{BOARD_SIZE}  |  "
          f"Repeats: {args.repeats} (+{args.warmup} warm-up)")
    print(f"  Configs: {len(MODEL_CONFIGS)}  |  SIM counts: {SIM_COUNTS}  |  Stages: {len(STAGES)}")
    print()

    # Pre-build stage games
    stage_games = {}
    for sl, ff in STAGES:
        g, nm = get_stage_game(ff, BOARD_SIZE)
        stage_games[sl] = g
        print(f"  Stage '{sl}': {nm}/{total_lines} lines drawn ({int(ff*100)}% filled), "
              f"{total_lines - nm} valid moves remaining")
    print()

    # ── Grid search ───────────────────────────────────────────────────────────
    all_results = {}
    total_cells = len(MODEL_CONFIGS) * len(SIM_COUNTS) * len(STAGES)
    cell_idx = 0

    for ch, rb in MODEL_CONFIGS:
        ck = f"{ch}ch x{rb}blk"
        all_results[ck] = {}
        nnet   = build_nnet(ch, rb, device, args.model_path)
        params = param_count(nnet)
        print(f"\n{'━'*76}")
        print(f"  Model: {ck}  |  Parameters: {params:,}")
        print(f"{'━'*76}")

        for n_sims in SIM_COUNTS:
            all_results[ck][n_sims] = {}
            for sl, ff in STAGES:
                cell_idx += 1
                m = measure(nnet, n_sims, stage_games[sl], args.repeats, args.warmup)
                all_results[ck][n_sims][sl] = {**m, "params": params, "ch": ch, "rb": rb}
                pct = cell_idx / total_cells * 100
                print(f"  [{pct:5.1f}%] sims={n_sims:4d} | {sl:<14} | "
                      f"{m['mean_ms']:8.2f} ms (+-{m['std_ms']:.2f}) | "
                      f"ms/sim={m['ms_per_sim']:.4f} | NN_calls~{m['nn_calls']:.0f}")

    # ── Per-stage mean-ms tables ──────────────────────────────────────────────
    cw = 11; fw = 15

    for sl, ff in STAGES:
        print(f"\n\n{'='*76}")
        print(f"  STAGE: {sl}  ({int(ff*100)}% lines filled)  —  mean move time (ms)")
        print(f"{'='*76}")
        hdr = f"{'Config':<{fw}} {'Params':>10}"
        for s in SIM_COUNTS:
            hdr += f"{'NN' if s==0 else str(s)+'sim':>{cw}}"
        print(hdr)
        print("-" * (fw + 11 + cw * len(SIM_COUNTS)))
        for ch, rb in MODEL_CONFIGS:
            ck = f"{ch}ch x{rb}blk"
            p  = all_results[ck][SIM_COUNTS[0]][sl]["params"]
            row = f"{ck:<{fw}} {p:>10,}"
            for s in SIM_COUNTS:
                row += f"{all_results[ck][s][sl]['mean_ms']:>{cw}.2f}"
            print(row + " ms")

        # Speed-up vs full model
        full = "256ch x10blk"
        print(f"\n  Speed-up vs {full}:")
        hdr2 = f"{'Config':<{fw}} {'Params':>10}"
        for s in SIM_COUNTS:
            hdr2 += f"{'NN' if s==0 else str(s)+'sim':>{cw}}"
        print(hdr2)
        print("-" * (fw + 11 + cw * len(SIM_COUNTS)))
        for ch, rb in MODEL_CONFIGS:
            ck = f"{ch}ch x{rb}blk"
            p  = all_results[ck][SIM_COUNTS[0]][sl]["params"]
            row = f"{ck:<{fw}} {p:>10,}"
            for s in SIM_COUNTS:
                m_c  = all_results[ck][s][sl]["mean_ms"]
                m_f  = all_results[full][s][sl]["mean_ms"]
                su   = m_f / m_c if m_c > 0 else 0
                row += f"{su:>{cw}.2f}x"
            print(row)

    # ── ms/simulation summary ─────────────────────────────────────────────────
    sim_cols = [s for s in SIM_COUNTS if s > 0]
    print(f"\n\n{'='*76}")
    print("  ms / simulation  (mean across all 3 stages)")
    print(f"{'='*76}")
    hdr = f"{'Config':<{fw}}"
    for s in sim_cols: hdr += f"{str(s)+' sims':>{cw}}"
    print(hdr)
    print("-" * (fw + cw * len(sim_cols)))
    for ch, rb in MODEL_CONFIGS:
        ck  = f"{ch}ch x{rb}blk"
        row = f"{ck:<{fw}}"
        for s in sim_cols:
            vals = [all_results[ck][s][sl]["ms_per_sim"] for sl, _ in STAGES]
            row += f"{np.mean(vals):>{cw}.4f}"
        print(row + " ms/sim")

    # ── Best config per (stage, n_sims) ──────────────────────────────────────
    print(f"\n\n{'='*76}")
    print("  Fastest config per (stage, n_sims):")
    print(f"{'='*76}")
    for sl, _ in STAGES:
        for n_sims in SIM_COUNTS:
            best = min(MODEL_CONFIGS,
                       key=lambda c: all_results[f"{c[0]}ch x{c[1]}blk"][n_sims][sl]["mean_ms"])
            bk  = f"{best[0]}ch x{best[1]}blk"
            bms = all_results[bk][n_sims][sl]["mean_ms"]
            sstr = "NN-only" if n_sims == 0 else f"{n_sims:3d} sims"
            print(f"  {sl:<14} | {sstr} | {bk}  ({bms:.2f} ms)")

    # ── CSV ───────────────────────────────────────────────────────────────────
    if args.csv:
        fields = ["config","num_channels","num_res_blocks","params",
                  "n_sims","stage","fill_pct","mean_ms","std_ms","ms_per_sim","nn_calls"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for ch, rb in MODEL_CONFIGS:
                ck = f"{ch}ch x{rb}blk"
                for n_sims in SIM_COUNTS:
                    for sl, ff in STAGES:
                        m = all_results[ck][n_sims][sl]
                        w.writerow({"config":ck,"num_channels":ch,"num_res_blocks":rb,
                                    "params":m["params"],"n_sims":n_sims,"stage":sl,
                                    "fill_pct":int(ff*100),
                                    "mean_ms":round(m["mean_ms"],4),
                                    "std_ms":round(m["std_ms"],4),
                                    "ms_per_sim":round(m["ms_per_sim"],6),
                                    "nn_calls":round(m["nn_calls"],1)})
        print(f"\n  CSV saved to: {args.csv}")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
