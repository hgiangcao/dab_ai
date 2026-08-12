import os
import argparse
import random
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    import pandas as pd
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from agent_interface import BaseAgent
from game import DotsAndBoxesGame

from util import create_agent, RandomAgent





def run_single_game(args):
    agent1_name, agent2_name, size, game_index, model_path, n_simulations = args
    from game import DotsAndBoxesGame
    import time

    # Prevent PyTorch from spawning multiple OpenMP threads per worker.
    # With N workers each using K threads, you get N*K threads on K cores → thrashing.
    # Setting 1 thread per worker gives clean utilization.
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    
    agent1 = create_agent(agent1_name, size, model_path, n_simulations)
    agent2 = create_agent(agent2_name, size, model_path, n_simulations)

    # Reset MCTS trees so no stale state leaks across games
    if hasattr(agent1, 'reset'):
        agent1.reset()
    if hasattr(agent2, 'reset'):
        agent2.reset()

    starting_player = 1 if (game_index % 2 == 0) else -1
    game = DotsAndBoxesGame(size=size, starting_player=starting_player, early_stopping=True)
    
    a1_moves = 0
    a2_moves = 0
    a1_time = 0.0
    a2_time = 0.0
    last_move = None  # track last move so MCTS tree-reuse can advance past opponent's turn
    
    while game.is_running():
        total_moves = a1_moves + a2_moves
        if total_moves < 4:
            move = random.choice(game.get_valid_moves())
            if game.current_player == 1:
                a1_moves += 1
            else:
                a2_moves += 1
            last_move = move
            game.execute_move(move)
            continue

        if game.current_player == 1:
            start_t = time.time()
            # Inform agent1's tree of the opponent's last move before searching
            if last_move is not None and hasattr(agent1, '_last_action'):
                agent1._last_action = last_move
            move = agent1.get_move(game)
            a1_time += (time.time() - start_t)
            a1_moves += 1
        else:
            start_t = time.time()
            # Inform agent2's tree of the opponent's last move before searching
            if last_move is not None and hasattr(agent2, '_last_action'):
                agent2._last_action = last_move
            move = agent2.get_move(game)
            a2_time += (time.time() - start_t)
            a2_moves += 1
        last_move = move
        game.execute_move(move)
        
    return agent1_name, agent2_name, game.result, a1_moves, a1_time, a2_moves, a2_time

def main():
    parser = argparse.ArgumentParser(description="Evaluate AlphaZero agent against other bots.")
    parser.add_argument("--size", type=int, default=5, help="Grid size of the Dots and Boxes board (default: 5).")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play per matchup (default: 20).")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes (default: cpu_count - 1).")
    parser.add_argument("--output", type=str, default="alphazero_evaluation.png", help="Path to export the results visualization PNG (default: alphazero_evaluation.png).")
    parser.add_argument("--run", type=str, default=None, help="The run subdirectory name under logs (e.g. run_1) to load best.pth.tar from. If not specified, loads from project root.")
    parser.add_argument("--mcts_sims", type=int, default=200, help="Number of simulations for AlphaZero MCTS (default: 200).")
    args = parser.parse_args()

    import config
    if args.run:
        model_path = os.path.join(config.LOGS_DIR, args.run, "best.pth.tar")
    else:
        model_path = "best.pth.tar"

    opponent_names = [
        # "Random",
        # "Greedy", 
        # "Greedy Chain",
        # "UCLABot_v3",
        "UCLABot_v6",
        # "SimpleBot",
        # "SimpleBot_v2",
        # "UCLA_MCTS_0.1",
        # "UCLA_MCTS_0.2",
        "AlphaZero_100",
        "AlphaZero_200",
        "AlphaZero_300",
        "AlphaZero_500",
    ]

    tasks = []
    # Play AlphaZero (Player 1) vs Opponents (Player 2)
    # Tasks are parallelized per individual game
    for opp in opponent_names:
        for game_idx in range(args.games):
            tasks.append(("AlphaZero", opp, args.size, game_idx, model_path, args.mcts_sims))

    num_cores = args.workers if args.workers is not None else max(1, mp.cpu_count() - 1)
    print(f"Starting AlphaZero evaluation (parallelized per game) against {len(opponent_names)} bots (size={args.size}x{args.size}, games={args.games} per matchup).")
    print(f"Running on {num_cores} worker processes...")
    
    matchup_stats = {opp: {"wins": 0, "losses": 0, "draws": 0} for opp in opponent_names}
    agent_moves = {name: 0 for name in ["AlphaZero"] + opponent_names}
    agent_time = {name: 0.0 for name in ["AlphaZero"] + opponent_names}
    
    with mp.Pool(num_cores) as pool:
        pbar = tqdm(pool.imap_unordered(run_single_game, tasks), total=len(tasks), desc="Running Games")
        for res in pbar:
            a1_name, a2_name, result, a1_moves, a1_time, a2_moves, a2_time = res
            opp = a2_name
            
            agent_moves[a1_name] += a1_moves
            agent_time[a1_name] += a1_time
            agent_moves[a2_name] += a2_moves
            agent_time[a2_name] += a2_time
            
            if result == 1:
                matchup_stats[opp]["wins"] += 1
            elif result == -1:
                matchup_stats[opp]["losses"] += 1
            else:
                matchup_stats[opp]["draws"] += 1
                
            total_played = sum(matchup_stats[opp].values())
            wr = (matchup_stats[opp]["wins"] + 0.5 * matchup_stats[opp]["draws"]) / total_played
            pbar.set_postfix_str(f"Latest {opp} WR: {wr:.2f}")

    # Print move speed results to stdout
    print("\n--- Move Speed Report (moves/second) ---")
    for name in ["AlphaZero"] + opponent_names:
        m_count = agent_moves[name]
        total_t = agent_time[name]
        mps = m_count / total_t if total_t > 0 else 0.0
        print(f"{name:<20} | Moves: {m_count:<6} | Total Time: {total_t:<8.3f}s | Moves/Sec: {mps:.2f}")
    print("-" * 60)

    print("\n--- AlphaZero Evaluation Results ---")
    print(f"{'Opponent':<20} | {'AlphaZero Win Rate':<20} | {'Wins':<6} | {'Losses':<6} | {'Draws':<6}")
    print("-" * 68)
    
    opponents_for_plot = []
    winrates_for_plot = []

    for opp in opponent_names:
        stats = matchup_stats[opp]
        wr = (stats["wins"] + 0.5 * stats["draws"]) / args.games
        opponents_for_plot.append(opp)
        winrates_for_plot.append(wr)
        print(f"{opp:<20} | {wr:>18.2%} | {stats['wins']:<6} | {stats['losses']:<6} | {stats['draws']:<6}")
        
    # Plot results
    plt.figure(figsize=(10, 6))
    bars = plt.bar(opponents_for_plot, winrates_for_plot, color='skyblue')
    plt.axhline(y=0.5, color='r', linestyle='--', label='50% Win Rate')
    plt.ylim(0, 1.05)
    plt.ylabel('AlphaZero Win Rate')
    plt.xlabel('Opponent Bot')
    plt.title(f'AlphaZero ({args.mcts_sims} Sims) vs. Other Bots ({args.games} Games, Size {args.size}x{args.size})')
    plt.legend()
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, f'{height:.2%}', ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()
    print(f"Results chart saved to {args.output}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
