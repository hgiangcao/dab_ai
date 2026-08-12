import os
import argparse
import random
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

# Set matplotlib backend to Agg to prevent GUI popups
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

from util import create_agent, RandomAgent

def run_single_matchup(args):
    agent1_name, agent2_name, opp_idx, size, num_games = args
    from game import DotsAndBoxesGame
    
    # Initialize agents inside the worker process
    print()
    
    a1_wins = 0
    a2_wins = 0
    draws = 0
    game_records = []
    
    a1_moves = 0
    a2_moves = 0
    a1_time = 0.0
    a2_time = 0.0
    import time
    
    for i in tqdm(range(num_games), desc=f"Playing {agent1_name} vs {agent2_name}"):
        agent1 = create_agent(agent1_name, size)
        agent2 = create_agent(agent2_name, size)
        # Alternate who starts to ensure fairness
        starting_player = 1 if (i % 2 == 0) else -1
        game = DotsAndBoxesGame(size=size, starting_player=starting_player,early_stopping=False)
        
        game_moves = []
        game_policies = []
        
        while game.is_running():
            if len(game_moves) < 4:
                import random
                valid_moves = game.get_valid_moves()
                move = random.choice(valid_moves)
                
                if game.current_player == 1:
                    a1_moves += 1
                else:
                    a2_moves += 1
            else:
                if game.current_player == 1:
                    start_t = time.time()
                    move = agent1.get_move(game)
                    a1_time += (time.time() - start_t)
                    a1_moves += 1
                else:
                    start_t = time.time()
                    move = agent2.get_move(game)
                    a2_time += (time.time() - start_t)
                    a2_moves += 1
            
            game_moves.append(int(move))
            dummy_pi = [0.0] * game.N_LINES
            dummy_pi[move] = 1.0
            game_policies.append(dummy_pi)
            
            game.execute_move(move)
            
        if game.result == 1:
            a1_wins += 1
        elif game.result == -1:
            a2_wins += 1
        else:
            draws += 1
            
        game_records.append({
            "winner": int(game.result),
            "moves": game_moves,
            "policies": game_policies
        })
            
    return agent1_name, agent2_name, opp_idx, a1_wins, a2_wins, draws, game_records, a1_moves, a1_time, a2_moves, a2_time

def plot_heatmap(matrix, row_names, col_names, output_path, title):
    plt.figure(figsize=(10, min(8, len(row_names)*1.5 + 2)))
    if HAS_SEABORN:
        df = pd.DataFrame(matrix, index=row_names, columns=col_names)
        sns.heatmap(df, annot=True, cmap="coolwarm", vmin=0, vmax=1, fmt=".2f", cbar_kws={'label': 'Win Rate'})
    else:
        # Fallback to pure matplotlib if seaborn is not available
        plt.imshow(matrix, cmap="coolwarm", vmin=0, vmax=1)
        plt.colorbar(label='Win Rate')
        plt.xticks(np.arange(len(col_names)), col_names, rotation=45, ha="right")
        plt.yticks(np.arange(len(row_names)), row_names)
        for i in range(len(row_names)):
            for j in range(len(col_names)):
                plt.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", color="black")
                
    plt.title(title)
    plt.ylabel("Agent")
    plt.xlabel("Opponent")
    plt.tight_layout()
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Heatmap exported to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Dots and Boxes agents in a 1-vs-all tournament.")
    parser.add_argument("--size", type=int, default=5, help="Grid size of the Dots and Boxes board (default: 3).")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play per matchup (default: 20).")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes (default: cpu_count - 1).")
    parser.add_argument("--output", type=str, default="tournament_heatmap.png", help="Path to export the heatmap PNG (default: tournament_heatmap.png).")
    parser.add_argument("--selected_bot", type=str, default="UCLABot_v6", help="The selected bot to play against all others.")
    args = parser.parse_args()

    # List of agent names to include in the tournament
    agent_names = [
        "Random", 
        "Greedy", 
        "UCLABot_v3",
        "UCLABot_v3",
        "UCLABot_v3",
        "UCLABot_v3",
        "UCLABot_v6",
        "UCLABot_v6",
        "UCLABot_v6",
        "UCLABot_v6",
        "fill_bot"
    ]
    
    opponent_names = agent_names
    n_opponents = len(opponent_names)
    
    # Generate unique display names to treat duplicates as separate agents
    display_names = []
    counts = {}
    for name in opponent_names:
        counts[name] = counts.get(name, 0) + 1
        if opponent_names.count(name) > 1:
            display_names.append(f"{name}_{counts[name]}")
        else:
            display_names.append(name)
    
    tasks = []
    for idx, opponent in enumerate(opponent_names):
        tasks.append((args.selected_bot, opponent, idx, args.size, args.games))

    # Run matchups in parallel
    num_cores = args.workers if args.workers is not None else max(1, mp.cpu_count() - 1)
    print(f"Starting 1-vs-all tournament: {args.selected_bot} vs {n_opponents} opponents, size={args.size}x{args.size}, games={args.games} per matchup.")
    print(f"Running on {num_cores} worker processes...")
    
    selected_bot_moves = 0
    selected_bot_time = 0.0
    opp_moves = [0] * n_opponents
    opp_time = [0.0] * n_opponents
    
    results = []
    with mp.Pool(num_cores) as pool:
        pbar = tqdm(pool.imap_unordered(run_single_matchup, tasks), total=len(tasks), desc="Running Matchups")
        for res in pbar:
            results.append(res)
            a1_name, a2_name, opp_idx, a1_wins, a2_wins, draws, _, _, _, _, _ = res
            a1_wr = (a1_wins + 0.5 * draws) / args.games
            a2_wr = (a2_wins + 0.5 * draws) / args.games
            pbar.set_postfix_str(f"{a1_name} vs {a2_name}: {a1_wr:.1%} vs {a2_wr:.1%} ({a1_wins}W-{a2_wins}L-{draws}D)")
            
    # Compile the winrate matrix (1 row, n_opponents columns)
    winrate_matrix = np.zeros((1, n_opponents))
    
    import json
    log_path = "game_logs_bot.jsonl"
    print(f"Saving played tournament games to {log_path}...")
    with open(log_path, "a") as f_log:
        for res in results:
            a1_name, a2_name, opp_idx, a1_wins, a2_wins, draws, game_records, a1_mvs, a1_t, a2_mvs, a2_t = res
            
            for record in game_records:
                f_log.write(json.dumps(record) + "\n")
                
            selected_bot_moves += a1_mvs
            selected_bot_time += a1_t
            opp_moves[opp_idx] += a2_mvs
            opp_time[opp_idx] += a2_t
            
            wr1 = (a1_wins + 0.5 * draws) / args.games
            winrate_matrix[0][opp_idx] = wr1

    # Print move speed results to stdout
    print("\n--- Move Speed Report (moves/second) ---")
    s_mps = selected_bot_moves / selected_bot_time if selected_bot_time > 0 else 0.0
    print(f"{args.selected_bot:<20} | Moves: {selected_bot_moves:<6} | Total Time: {selected_bot_time:<8.3f}s | Moves/Sec: {s_mps:.2f}")
    
    for idx, d_name in enumerate(display_names):
        m_count = opp_moves[idx]
        total_t = opp_time[idx]
        mps = m_count / total_t if total_t > 0 else 0.0
        print(f"{d_name:<20} | Moves: {m_count:<6} | Total Time: {total_t:<8.3f}s | Moves/Sec: {mps:.2f}")
    print("-" * 60)

    # Print results to stdout
    print(f"\n--- Winrate (Selected Bot: {args.selected_bot} vs Opponents) ---")
    for j, d_name in enumerate(display_names):
        print(f"Opponent: {d_name:<20} | Winrate: {winrate_matrix[0][j]:>6.2%}")
    print("-" * 60)

    # Generate and save the heatmap
    title = f"Win Rate of {args.selected_bot} vs Opponents"
    plot_heatmap(winrate_matrix, [args.selected_bot], display_names, args.output, title)

if __name__ == "__main__":
    mp.freeze_support()
    main()
