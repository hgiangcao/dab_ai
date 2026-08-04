"""Helper: writes the clean benchmark_mcts.py file."""
import pathlib

content = '''\
import os
import sys
import time
import numpy as np
import torch
import random as _random
from collections import deque

sys.path.append(os.path.join(os.path.dirname(__file__), "distributed"))

from game import DotsAndBoxesGame
from model import NNetWrapper, dotdict
from mcts import MCTS, AZNode
import config


def count_unique_nodes(root):
    """BFS count of all AZNode objects reachable from root."""
    visited = set()
    queue = deque([root])
    visited.add(id(root))
    while queue:
        node = queue.popleft()
        for child in node.children.values():
            if id(child) not in visited:
                visited.add(id(child))
                queue.append(child)
    return len(visited)


def measure_avg_depth(root):
    """Average depth of leaf nodes (BFS)."""
    total_depth = 0
    leaf_count = 0
    queue = deque([(root, 0)])
    while queue:
        node, depth = queue.popleft()
        if not node.children:
            total_depth += depth
            leaf_count += 1
        else:
            for child in node.children.values():
                queue.append((child, depth + 1))
    return total_depth / leaf_count if leaf_count > 0 else 0.0


def get_prefilled_game(num_moves):
    """5x5 board with num_moves random moves (seed=42)."""
    game = DotsAndBoxesGame(size=5, early_stopping=False)
    rng = _random.Random(42)
    for _ in range(num_moves):
        if not game.is_running():
            break
        moves = game.get_valid_moves()
        game.execute_move(rng.choice(moves))
    return game


def run_benchmark_for_state(num_moves, mcts, n_simulations):
    total_lines = 60
    empty_lines = total_lines - num_moves

    print("\\n" + "=" * 65)
    print(f"  BENCHMARK  |  pre-filled moves: {num_moves}  |  empty lines: {empty_lines}")
    print("=" * 65)

    # Test 1: single mcts.play() call
    print(f"\\n[Test 1] Single move via mcts.play() with {n_simulations} sims ...")
    game = get_prefilled_game(num_moves)
    mcts.reset_tree()

    start = time.perf_counter()
    pi = mcts.play(game, temp=0, last_action=None)
    elapsed_ms = (time.perf_counter() - start) * 1000

    root_t1      = mcts._root
    unique_nodes = count_unique_nodes(root_t1)
    avg_depth    = measure_avg_depth(root_t1)
    max_depth    = mcts.max_depth_reached
    chosen_move  = int(np.argmax(pi))

    print(f"  -> Move chosen           : {chosen_move}")
    print(f"  -> Move time             : {elapsed_ms:.2f} ms")
    print(f"  -> Time per simulation   : {elapsed_ms / n_simulations:.4f} ms")
    print(f"  -> Max search depth      : {max_depth}")
    print(f"  -> Avg leaf depth        : {avg_depth:.2f}")
    print(f"  -> Unique nodes explored : {unique_nodes}")

    # Test 2: full game, fresh tree each move
    print(f"\\n[Test 2] Full game timing (fresh tree each move) ...")
    game2 = get_prefilled_game(num_moves)

    move_times      = []
    depths_max      = []
    depths_avg_list = []
    unique_per_move = []

    while game2.is_running():
        mcts.reset_tree()
        t0 = time.perf_counter()
        pi2 = mcts.play(game2, temp=0, last_action=None)
        move_ms = (time.perf_counter() - t0) * 1000
        move_times.append(move_ms)

        root2 = mcts._root
        depths_max.append(mcts.max_depth_reached)
        depths_avg_list.append(measure_avg_depth(root2))
        unique_per_move.append(count_unique_nodes(root2))

        action = int(np.argmax(pi2))
        game2.execute_move(action)

    n_moves = len(move_times)
    print(f"  -> Moves played             : {n_moves}")
    print(f"  -> Avg move time            : {np.mean(move_times):.2f} ms  (std={np.std(move_times):.2f})")
    print(f"  -> Min / Max move time      : {np.min(move_times):.2f} / {np.max(move_times):.2f} ms")
    print(f"  -> Avg max search depth     : {np.mean(depths_max):.2f}  (per move)")
    print(f"  -> Avg leaf depth per move  : {np.mean(depths_avg_list):.2f}")
    print(f"  -> Avg unique nodes/move    : {np.mean(unique_per_move):.1f}")
    print(f"  -> Max unique nodes (move)  : {int(np.max(unique_per_move))}")

    return {
        \'t1_move_ms\':        elapsed_ms,
        \'t1_max_depth\':      max_depth,
        \'t1_avg_depth\':      avg_depth,
        \'t1_unique_nodes\':   unique_nodes,
        \'t2_avg_move_ms\':    float(np.mean(move_times)),
        \'t2_avg_max_depth\':  float(np.mean(depths_max)),
        \'t2_avg_leaf_depth\': float(np.mean(depths_avg_list)),
        \'t2_avg_nodes\':      float(np.mean(unique_per_move)),
        \'t2_max_nodes\':      int(np.max(unique_per_move)),
        \'t2_n_moves\':        n_moves,
    }


def main():
    print("AlphaZero MCTS Benchmark")
    print("=" * 65)

    dummy_game = DotsAndBoxesGame(size=5)
    device = \'cuda\' if torch.cuda.is_available() else \'cpu\'
    print(f"Device : {device}")

    eval_args = dotdict({
        \'lr\':              config.LEARNING_RATE,
        \'epochs\':          config.EPOCHS,
        \'batch_size\':      config.BATCH_SIZE,
        \'num_channels\':    256,
        \'num_res_blocks\':  10,
        \'l2_reg\':          1e-4,
        \'n_simulations\':   100,
        \'c_puct\':          config.MCTS_C_PUCT,
        \'dirichlet_eps\':   0.0,
        \'dirichlet_alpha\': config.MCTS_DIRICHLET_ALPHA,
        \'device\':          device,
    })

    nnet = NNetWrapper(dummy_game, eval_args)

    checkpoint_dir  = config.get_current_model_dir()
    pretrained_path = os.path.join(checkpoint_dir, "pretrained.pth.tar")
    candidate_path  = os.path.join(checkpoint_dir, "checkpoint_candidate.pth.tar")
    best_path       = os.path.join(checkpoint_dir, "best.pth.tar")

    loaded = False
    for path in [pretrained_path, candidate_path, best_path]:
        if os.path.exists(path):
            try:
                print(f"Loading weights from {path} ...")
                state = torch.load(path, map_location=device, weights_only=False)
                nnet.nnet.load_state_dict(state[\'state_dict\'] if \'state_dict\' in state else state)
                loaded = True
                break
            except Exception as e:
                print(f"  Failed: {e}")

    if not loaded:
        print("No checkpoint found - using randomly initialised weights.")

    nnet.nnet.eval()

    print("\\nWarming up ...")
    warmup_params = {
        "n_simulations":   100,
        "c_puct":          config.MCTS_C_PUCT,
        "dirichlet_eps":   0.0,
        "dirichlet_alpha": config.MCTS_DIRICHLET_ALPHA,
    }
    warmup_mcts = MCTS(nnet, warmup_params)
    warmup_root = AZNode(parent=None, s=dummy_game, a=None)
    warmup_mcts.search(warmup_root, is_root=True, current_depth=0)
    warmup_mcts.reset_tree()
    print("Warm-up complete.")

    sim_counts = [100, 200, 500, 1000]
    results = {}

    for n_sims in sim_counts:
        print(f"\\n{\'#\'*65}")
        print(f"#  n_simulations = {n_sims}")
        print(f"{\'#\'*65}")
        mcts_params = {
            "n_simulations":   n_sims,
            "c_puct":          config.MCTS_C_PUCT,
            "dirichlet_eps":   0.0,
            "dirichlet_alpha": config.MCTS_DIRICHLET_ALPHA,
        }
        mcts = MCTS(nnet, mcts_params)
        results[n_sims] = run_benchmark_for_state(num_moves=0, mcts=mcts, n_simulations=n_sims)

    col = 10
    lbl = 38

    print("\\n" + "=" * 78)
    print("          ALPHAZERO BENCHMARK SUMMARY (from empty 5x5 board)")
    print("=" * 78)
    header = f"{\'Metric\':<{lbl}}" + "".join(f"{str(s)+\' sims\':^{col}}" for s in sim_counts)
    print(header)
    print("-" * 78)

    def row(label, fmt_fn):
        return f"{label:<{lbl}}" + "".join(f"{fmt_fn(results[s]):^{col}}" for s in sim_counts)

    print(row("Single move time (ms)",         lambda r: f"{r[\'t1_move_ms\']:.1f}"))
    print(row("  time / simulation (ms)",      lambda r: f"{r[\'t1_move_ms\'] / r[\'t1_unique_nodes\']:.4f}" if r[\'t1_unique_nodes\'] else "N/A"))
    print(row("  max search depth",            lambda r: str(r[\'t1_max_depth\'])))
    print(row("  avg leaf depth",              lambda r: f"{r[\'t1_avg_depth\']:.2f}"))
    print(row("  unique nodes explored",       lambda r: str(r[\'t1_unique_nodes\'])))
    print("-" * 78)
    print(row("Full-game avg move time (ms)",  lambda r: f"{r[\'t2_avg_move_ms\']:.1f}"))
    print(row("  avg max depth / move",        lambda r: f"{r[\'t2_avg_max_depth\']:.2f}"))
    print(row("  avg leaf depth / move",       lambda r: f"{r[\'t2_avg_leaf_depth\']:.2f}"))
    print(row("  avg unique nodes / move",     lambda r: f"{r[\'t2_avg_nodes\']:.1f}"))
    print(row("  max unique nodes (any move)", lambda r: str(r[\'t2_max_nodes\'])))
    print(row("  moves played",               lambda r: str(r[\'t2_n_moves\'])))
    print("=" * 78)


if __name__ == "__main__":
    main()
'''

out = pathlib.Path(r"c:\Users\user\Downloads\dab_ai\benchmark_mcts.py")
out.write_text(content, encoding="utf-8")
print(f"Written {len(content.splitlines())} lines to {out}")
