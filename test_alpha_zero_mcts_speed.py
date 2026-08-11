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
    params = {"n_simulations": n_sims, "c_puct": config.MCTS_C_PUCT,
              "dirichlet_eps": 0.0, "dirichlet_alpha": config.MCTS_DIRICHLET_ALPHA}
    mcts_obj = MCTS(nnet, params)
    times = []; nn_calls_list = []
    for i in range(warmup + repeats):
        mcts_obj.reset_tree()
        gs = game_state.clone(track_history=False)
        t0 = time.perf_counter()
        mcts_obj.play(gs, temp=0, last_action=None)
        ms = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(ms)
            nn_calls_list.append(count_nn_calls(mcts_obj))
    mean_ms   = float(np.mean(times))
    avg_nn    = float(np.mean(nn_calls_list))
    # Saturation: if NN calls are much less than n_sims, the tree is exhausted
    # and the measure is dominated by re-traversal, NOT inference speed.
    saturated = (n_sims > 0) and (avg_nn < 0.80 * n_sims)
    # ms/NN_call: the true cost of one useful (new-node) evaluation
    ms_per_nn_call = mean_ms / avg_nn if avg_nn > 0 else mean_ms
    return {
        "mean_ms":       mean_ms,
        "std_ms":        float(np.std(times)),
        "ms_per_nn_call": ms_per_nn_call,
        # Keep ms_per_sim for backward compat but don't use as primary metric
        "ms_per_sim":    mean_ms / n_sims if n_sims > 0 else mean_ms,
        "nn_calls":      avg_nn,
        "saturated":     saturated,
    }


def param_count(nnet):
    return sum(p.numel() for p in nnet.nnet.parameters())


def benchmark_nn_inference(ch: int, rb: int, device: str, model_path: str,
                            n_calls: int = 200, warmup: int = 20) -> dict:
    """
    Time raw nnet.predict() calls directly — no MCTS.
    This is the ONLY true measure of pure architecture inference speed.
    Returns mean_ms, std_ms, calls_per_sec.
    """
    nnet = build_nnet(ch, rb, device, model_path)
    dummy_game = DotsAndBoxesGame(size=BOARD_SIZE)

    # Build a fixed board tensor for the timing loop
    size = dummy_game.SIZE
    sp1 = size + 1
    board = np.zeros((4, sp1, sp1), dtype=np.float32)
    board[0, :sp1, :size]  = 0.5   # some content so it's not all-zero
    board[1, :size, :sp1]  = 0.5

    times = []
    for i in range(warmup + n_calls):
        t0 = time.perf_counter()
        nnet.predict(board)
        ms = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(ms)

    return {
        "mean_ms":      float(np.mean(times)),
        "std_ms":       float(np.std(times)),
        "calls_per_sec": 1000.0 / float(np.mean(times)),
        "params":       param_count(nnet),
    }


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

    # ── Step 1: Pure NN inference speed (the true architecture comparison) ────
    print(f"\n{'━'*76}")
    print("  STEP 1 — Pure NN Inference Speed  (raw nnet.predict() calls, no MCTS)")
    print(f"{'━'*76}")
    print(f"  {'Config':<15} {'Params':>12}  {'ms/call':>10}  {'±':>7}  {'calls/sec':>10}")
    print("  " + "-" * 60)

    nn_inference_results = {}
    for ch, rb in MODEL_CONFIGS:
        ck = f"{ch}ch x{rb}blk"
        r  = benchmark_nn_inference(ch, rb, device, args.model_path,
                                    n_calls=200, warmup=20)
        nn_inference_results[ck] = r
        print(f"  {ck:<15} {r['params']:>12,}  "
              f"{r['mean_ms']:>10.3f}  "
              f"±{r['std_ms']:>6.3f}  "
              f"{r['calls_per_sec']:>10.1f} calls/s")

    # Rank by ms/call
    ranked = sorted(MODEL_CONFIGS,
                    key=lambda c: nn_inference_results[f"{c[0]}ch x{c[1]}blk"]["mean_ms"])
    print(f"\n  Fastest: {ranked[0][0]}ch x{ranked[0][1]}blk  "
          f"({nn_inference_results[f'{ranked[0][0]}ch x{ranked[0][1]}blk']['mean_ms']:.3f} ms/call)")
    print(f"  Slowest: {ranked[-1][0]}ch x{ranked[-1][1]}blk  "
          f"({nn_inference_results[f'{ranked[-1][0]}ch x{ranked[-1][1]}blk']['mean_ms']:.3f} ms/call)")
    speedup_nn = (nn_inference_results[f"{ranked[-1][0]}ch x{ranked[-1][1]}blk"]["mean_ms"]
                  / nn_inference_results[f"{ranked[0][0]}ch x{ranked[0][1]}blk"]["mean_ms"])
    print(f"  Inference speed-up (fastest vs slowest): {speedup_nn:.2f}×")
    print()

    # ── Step 2: MCTS grid search ──────────────────────────────────────────────
    print(f"  STEP 2 — MCTS Grid Search  (primary metric: ms / NN-call)")
    print(f"  NOTE: cells marked [SAT] mean the tree is exhausted")
    print(f"  (nn_calls < 80% of n_sims). Those timings measure re-traversal,")
    print(f"  NOT inference speed, and should be IGNORED for model comparison.")

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
                sat_flag = "[SAT]" if m["saturated"] else "     "
                print(f"  [{pct:5.1f}%] {sat_flag} sims={n_sims:4d} | {sl:<14} | "
                      f"{m['mean_ms']:8.2f} ms (+-{m['std_ms']:.2f}) | "
                      f"ms/NN_call={m['ms_per_nn_call']:.4f} | "
                      f"NN_calls~{m['nn_calls']:.0f}/{n_sims}")

    # ── Per-stage ms/NN_call tables ──────────────────────────────────────────
    cw = 12; fw = 15
    SAT = "[SAT]"

    for sl, ff in STAGES:
        print(f"\n\n{'='*76}")
        print(f"  STAGE: {sl}  ({int(ff*100)}% lines filled)")
        print(f"  Primary metric: ms / NN-call  (= mean_ms / actual_nn_calls)")
        print(f"  [SAT] = tree exhausted; cell is invalid for speed comparison")
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
                m = all_results[ck][s][sl]
                if s == 0:
                    # NN-only: ms/call is ms_per_nn_call (one real NN call)
                    row += f"{m['mean_ms']:>{cw}.2f}"
                elif m["saturated"]:
                    row += f"{SAT:>{cw}}"
                else:
                    row += f"{m['ms_per_nn_call']:>{cw}.2f}"
            print(row + " ms/NN")

        # Speed-up vs full model (only on non-saturated cells)
        full = "256ch x10blk"
        print(f"\n  Speed-up vs {full}  (non-[SAT] cells only):")
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
                m_c  = all_results[ck][s][sl]
                m_f  = all_results[full][s][sl]
                if m_c["saturated"] or m_f["saturated"]:
                    row += f"{'[SAT]':>{cw}}"
                else:
                    val_c = m_c["mean_ms"] if s == 0 else m_c["ms_per_nn_call"]
                    val_f = m_f["mean_ms"] if s == 0 else m_f["ms_per_nn_call"]
                    su    = val_f / val_c if val_c > 0 else 0
                    row  += f"{su:>{cw}.2f}x"
            print(row)

    # ── ms/NN_call summary across all configs ─────────────────────────────────
    sim_cols = [s for s in SIM_COUNTS if s > 0]
    print(f"\n\n{'='*76}")
    print("  ms / NN-call  (mean over valid, non-saturated cells across all 3 stages)")
    print(f"  This is the CORRECT metric for comparing model inference speed")
    print(f"{'='*76}")
    hdr = f"{'Config':<{fw}}"
    for s in sim_cols: hdr += f"{str(s)+' sims':>{cw}}"
    print(hdr)
    print("-" * (fw + cw * len(sim_cols)))
    for ch, rb in MODEL_CONFIGS:
        ck  = f"{ch}ch x{rb}blk"
        row = f"{ck:<{fw}}"
        for s in sim_cols:
            valid_vals = [
                all_results[ck][s][sl]["ms_per_nn_call"]
                for sl, _ in STAGES
                if not all_results[ck][s][sl]["saturated"]
            ]
            if valid_vals:
                row += f"{np.mean(valid_vals):>{cw}.4f}"
            else:
                row += f"{'[SAT]':>{cw}}"
        print(row + " ms/NN_call")

    # ── Best config per (stage, n_sims) ──────────────────────────────────────
    print(f"\n\n{'='*76}")
    print("  Fastest config per (stage, n_sims)  — [SAT] cells excluded:")
    print(f"{'='*76}")
    for sl, _ in STAGES:
        for n_sims in SIM_COUNTS:
            valid = [
                c for c in MODEL_CONFIGS
                if not all_results[f"{c[0]}ch x{c[1]}blk"][n_sims][sl]["saturated"]
            ]
            if not valid:
                print(f"  {sl:<14} | {'NN-only' if n_sims==0 else str(n_sims)+' sims':<10} | all cells saturated")
                continue
            metric = "mean_ms" if n_sims == 0 else "ms_per_nn_call"
            best = min(valid, key=lambda c: all_results[f"{c[0]}ch x{c[1]}blk"][n_sims][sl][metric])
            bk   = f"{best[0]}ch x{best[1]}blk"
            bv   = all_results[bk][n_sims][sl][metric]
            unit = "ms" if n_sims == 0 else "ms/NN_call"
            sstr = "NN-only" if n_sims == 0 else f"{n_sims:3d} sims"
            print(f"  {sl:<14} | {sstr:<10} | {bk}  ({bv:.4f} {unit})")

    # ── Pure NN inference ranking (the definitive answer) ─────────────────────
    print(f"\n\n{'='*76}")
    print("  DEFINITIVE RANKING — Pure NN Inference Speed")
    print(f"  (nnet.predict() call, no MCTS overhead)")
    print(f"{'='*76}")
    ranked_all = sorted(
        MODEL_CONFIGS,
        key=lambda c: nn_inference_results[f"{c[0]}ch x{c[1]}blk"]["mean_ms"]
    )
    for rank, (ch, rb) in enumerate(ranked_all, 1):
        ck  = f"{ch}ch x{rb}blk"
        r   = nn_inference_results[ck]
        ref = nn_inference_results[f"{ranked_all[-1][0]}ch x{ranked_all[-1][1]}blk"]["mean_ms"]
        su  = ref / r["mean_ms"]
        print(f"  #{rank}  {ck:<15}  {r['params']:>12,} params  "
              f"{r['mean_ms']:>8.3f} ms  ±{r['std_ms']:.3f}  "
              f"{r['calls_per_sec']:>8.1f} calls/s  ({su:.2f}x vs slowest)")

    print(f"\n  Recommendation for fastest model with acceptable quality:")
    # Suggest the model that is in the top-3 inference AND has >= 128 channels
    candidates = [(ch, rb) for ch, rb in ranked_all if ch >= 128]
    if candidates:
        best_ch, best_rb = candidates[0]
        best_ck = f"{best_ch}ch x{best_rb}blk"
        print(f"  → {best_ck}  ({nn_inference_results[best_ck]['mean_ms']:.3f} ms/call, "
              f"{nn_inference_results[best_ck]['params']:,} params)")

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
