import os
import sys
import time
import numpy as np
import torch
import random as _random

# Add distributed to path
sys.path.append(os.path.join(os.path.dirname(__file__), "distributed"))

from game import DotsAndBoxesGame
from model import NNetWrapper, dotdict
from mcts import MCTS, AZNode
import config

def get_prefilled_game(num_moves):
    game = DotsAndBoxesGame(size=5,early_stopping=False)
    # Use a seed for reproducibility across different runs/tests
    rng = _random.Random(42)
    for _ in range(num_moves):
        if not game.is_running():
            break
        moves = game.get_valid_moves()
        game.execute_move(rng.choice(moves))
    return game

def run_benchmark_for_state(num_moves, nnet, mcts, eval_args):
    print("\n" + "=" * 60)
    print(f"BENCHMARK WITH {num_moves} PRE-FILLED MOVES ({60 - num_moves} empty lines left)")
    print("=" * 60)

    # ── Test 1: Measure time for 1000 simulations ──
    print("Running Test 1: Time for 1000 MCTS simulations...")
    game = get_prefilled_game(num_moves)
    start_time = time.perf_counter()
    
    test_root_1 = AZNode(parent=None, s=game, a=None)
    mcts.max_depth_reached = 0  # Reset counter
    for _ in range(1000):
        mcts.search(test_root_1, is_root=True, current_depth=0)
        
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    mcts_1000_depth = mcts.max_depth_reached
    print(f"  -> Time taken for 1000 simulations: {elapsed_ms:.2f} ms")
    print(f"  -> Average time per simulation: {elapsed_ms / 1000:.4f} ms")
    print(f"  -> Max depth reached: {mcts_1000_depth}")
    
    # ── Test 2: Measure simulations run in 100ms ──
    print("\nRunning Test 2: Number of simulations completed in 100ms...")
    game = get_prefilled_game(num_moves)
    test_root_2 = AZNode(parent=None, s=game, a=None)
    
    sim_count = 0
    time_limit = 0.1 # 100ms
    mcts.max_depth_reached = 0  # Reset counter
    start_time = time.perf_counter()
    
    while (time.perf_counter() - start_time) < time_limit:
        mcts.search(test_root_2, is_root=True, current_depth=0)
        sim_count += 1
        
    actual_elapsed = (time.perf_counter() - start_time) * 1000
    mcts_100ms_depth = mcts.max_depth_reached
    print(f"  -> Simulations completed: {sim_count}")
    print(f"  -> Actual elapsed time: {actual_elapsed:.2f} ms")
    print(f"  -> Average time per simulation: {actual_elapsed / sim_count:.4f} ms")
    print(f"  -> Max depth reached: {mcts_100ms_depth}")

    # ── Test 3: Measure heuristic MCTS bot simulations in 100ms ──
    print("\nRunning Test 3: Heuristic MCTS bot (bots/mcts_heuristic.py) simulations in 100ms...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bots"))
    from bots.mcts_heuristic import MCTS as HeuristicMCTS
    from bots.mcts_heuristic import Node as HNode

    heuristic_mcts = HeuristicMCTS(time_limit=None, n_simulations=999999)
    heuristic_game = get_prefilled_game(num_moves)
    
    valid_moves = heuristic_game.get_valid_moves()
    root_h = HNode(
        parent=None, move=None,
        player_to_move=heuristic_game.current_player,
        valid_moves=heuristic_mcts._order_expansion_moves(heuristic_game, valid_moves),
        terminal=not heuristic_game.is_running()
    )

    h_sim_count = 0
    time_limit_h = 0.1  # 100ms
    start_time = time.perf_counter()

    while (time.perf_counter() - start_time) < time_limit_h:
        node = root_h
        moves_made = 0

        # SELECTION
        while not node.untried_moves and node.children:
            node = heuristic_mcts._select_child(node)
            heuristic_game.execute_move(node.move)
            moves_made += 1

        # EXPANSION
        if node.untried_moves:
            move = node.untried_moves.pop(0)
            heuristic_game.execute_move(move)
            moves_made += 1
            terminal = not heuristic_game.is_running()
            valid_for_child = heuristic_mcts._order_expansion_moves(heuristic_game, heuristic_game.get_valid_moves()) if not terminal else []
            child = HNode(
                parent=node, move=move,
                player_to_move=heuristic_game.current_player,
                valid_moves=valid_for_child,
                terminal=terminal
            )
            node.children.append(child)
            node = child

        # ROLLOUT
        rollout_player = node.player_to_move
        val, rollout_moves = heuristic_mcts._rollout(heuristic_game, rollout_player)
        moves_made += rollout_moves

        # BACKPROPAGATION
        curr = node
        while curr is not None:
            curr.visits += 1
            curr.value_sum += val
            if curr.parent is not None:
                if curr.parent.player_to_move != curr.player_to_move:
                    val = -val
            curr = curr.parent

        # UNDO ALL MOVES
        for _ in range(moves_made):
            heuristic_game.undo_move()

        h_sim_count += 1

    actual_elapsed_h = (time.perf_counter() - start_time) * 1000
    print(f"  -> Simulations completed: {h_sim_count}")
    print(f"  -> Actual elapsed time: {actual_elapsed_h:.2f} ms")
    print(f"  -> Average time per simulation: {actual_elapsed_h / max(h_sim_count, 1):.4f} ms")

    # ── Test 4: Random move speed (pure game throughput baseline) ──
    print("\nRunning Test 4: Random move speed (full random games in 100ms)...")
    rand_sim_count = 0
    rand_move_count = 0
    time_limit_r = 0.1  # 100ms
    start_time = time.perf_counter()

    while (time.perf_counter() - start_time) < time_limit_r:
        rand_game = get_prefilled_game(num_moves)
        while rand_game.is_running():
            moves = rand_game.get_valid_moves()
            rand_game.execute_move(_random.choice(moves))
            rand_move_count += 1
        rand_sim_count += 1

    actual_elapsed_r = (time.perf_counter() - start_time) * 1000
    avg_moves = rand_move_count / max(rand_sim_count, 1)
    print(f"  -> Full random games completed: {rand_sim_count}")
    print(f"  -> Total moves executed: {rand_move_count}")
    print(f"  -> Average moves per game: {avg_moves:.1f}")
    print(f"  -> Actual elapsed time: {actual_elapsed_r:.2f} ms")
    print(f"  -> Average time per game: {actual_elapsed_r / max(rand_sim_count, 1):.4f} ms")

    # ── Test 5: Alpha-Beta depth in 100ms ──
    print("\nRunning Test 5: Alpha-Beta search depth reached in 100ms...")
    from bots.alpha_beta import AlphaBetaPlayer
    from lookup_board import ZobristHash
    from math import inf

    ab_game = get_prefilled_game(num_moves)
    zobrist = ZobristHash(ab_game.N_LINES)
    initial_hash = zobrist.compute_initial_hash(ab_game.l)

    ab_time_limit = 0.1  # 100ms
    end_time = time.time() + ab_time_limit
    max_possible_depth = len(ab_game.get_valid_moves())
    reached_depth = 0

    start_time_ab = time.perf_counter()
    for current_depth in range(1, max_possible_depth + 1):
        try:
            AlphaBetaPlayer.alpha_beta_search(
                s_node       = ab_game,
                a_latest     = None,
                depth        = current_depth,
                alpha        = -inf,
                beta         = inf,
                maximize     = True,
                zobrist      = zobrist,
                current_hash = initial_hash,
                end_time     = end_time
            )
            reached_depth = current_depth
        except TimeoutError:
            break

    actual_elapsed_ab = (time.perf_counter() - start_time_ab) * 1000
    print(f"  -> Alpha-Beta search depth reached: {reached_depth}")
    print(f"  -> Actual elapsed time: {actual_elapsed_ab:.2f} ms")

    # Return results for printing final summary
    return {
        'elapsed_ms': elapsed_ms,
        'sim_count': sim_count,
        'h_sim_count': h_sim_count,
        'rand_sim_count': rand_sim_count,
        'reached_depth': reached_depth
    }

def main():
    print("MCTS Benchmark Tool (Multi-State Version)")
    print("-" * 50)

    # Setup dummy/base state
    dummy_game = DotsAndBoxesGame(size=5)
    
    # 1. Setup network wrapper
    eval_args = dotdict({
        'lr': config.LEARNING_RATE,
        'epochs': config.EPOCHS,
        'batch_size': config.BATCH_SIZE,
        'num_channels': 256,
        'num_res_blocks': 10,
        'l2_reg': 1e-4,
        'n_simulations': 1000,
        'c_puct': config.MCTS_C_PUCT,
        'dirichlet_eps': 0.0,
        'dirichlet_alpha': config.MCTS_DIRICHLET_ALPHA,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })
    
    print(f"Device being used: {eval_args.device}")
    nnet = NNetWrapper(dummy_game, eval_args)
    
    # Try loading a model checkpoint
    checkpoint_dir = config.get_current_model_dir()
    candidate_path = os.path.join(checkpoint_dir, "checkpoint_candidate.pth.tar")
    best_path = os.path.join(checkpoint_dir, "best.pth.tar")
    
    loaded = False
    for path in [candidate_path, best_path]:
        if os.path.exists(path):
            try:
                print(f"Loading weights from {path}...")
                state = torch.load(path, map_location=eval_args.device, weights_only=False)
                nnet.nnet.load_state_dict(state['state_dict'] if 'state_dict' in state else state)
                loaded = True
                break
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                
    if not loaded:
        print("No checkpoints found. Running benchmark with randomly initialized weights.")
        
    nnet.nnet.eval()
    
    # Initialize MCTS
    mcts_params = {
        "n_simulations": 1000,
        "c_puct": config.MCTS_C_PUCT,
        "dirichlet_eps": 0.0,
        "dirichlet_alpha": config.MCTS_DIRICHLET_ALPHA,
    }
    mcts = MCTS(nnet, mcts_params)
    
    # Warm up MCTS
    print("Warming up model and MCTS...")
    temp_root = AZNode(parent=None, s=dummy_game, a=None)
    mcts.search(temp_root, is_root=True, current_depth=0)
    print("Warm up complete.")

    # Run benchmarks for 0, 30, and 45 moves
    results = {}
    for moves in [0, 30, 45]:
        results[moves] = run_benchmark_for_state(moves, nnet, mcts, eval_args)

    # ── Final Summary Table ──
    print("\n" + "=" * 60)
    print("                     BENCHMARK SUMMARY                     ")
    print("=" * 60)
    print(f"{'Metric':<30} | {'0 Moves':<8} | {'30 Moves':<8} | {'45 Moves':<8}")
    print("-" * 64)
    print(f"{'AlphaZero MCTS (1000 sims time)':<30} | {results[0]['elapsed_ms']:.1f} ms | {results[30]['elapsed_ms']:.1f} ms | {results[45]['elapsed_ms']:.1f} ms")
    print(f"{'AlphaZero MCTS (100ms sims)':<30} | {results[0]['sim_count']:<8} | {results[30]['sim_count']:<8} | {results[45]['sim_count']:<8}")
    print(f"{'Heuristic MCTS (100ms sims)':<30} | {results[0]['h_sim_count']:<8} | {results[30]['h_sim_count']:<8} | {results[45]['h_sim_count']:<8}")
    print(f"{'Random moves (100ms full games)':<30} | {results[0]['rand_sim_count']:<8} | {results[30]['rand_sim_count']:<8} | {results[45]['rand_sim_count']:<8}")
    print(f"{'Alpha-Beta depth (100ms)':<30} | Depth {results[0]['reached_depth']:<2} | Depth {results[30]['reached_depth']:<2} | Depth {results[45]['reached_depth']:<2}")
    print("=" * 60)

if __name__ == "__main__":
    main()
