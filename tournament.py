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

class RandomAgent(BaseAgent):
    def __init__(self, name: str = "Random"):
        super().__init__(name)

    def get_move(self, game_state) -> int:
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None
        return random.choice(valid_moves)

def create_agent(name: str, size: int):
    if name == "Random":
        return RandomAgent()
    elif name == "Alpha-Beta (1s)":
        from bots.alpha_beta import AlphaBetaPlayer
        return AlphaBetaPlayer(name=name, time_limit=1.0)
    elif name == "MCTS (1s)":
        from bots.mcts_x import MCTSGAgent
        return MCTSGAgent(name=name, time_limit=1.0)
    elif name == "Alpha-Beta (0.1s)":
        from bots.alpha_beta import AlphaBetaPlayer
        return AlphaBetaPlayer(name=name, time_limit=0.1)
    elif name == "MCTS (0.1s)":
        from bots.mcts_x import MCTSGAgent
        return MCTSGAgent(name=name, time_limit=0.1)
    elif name == "Alpha-Beta (0.5s)":
        from bots.alpha_beta import AlphaBetaPlayer
        return AlphaBetaPlayer(name=name, time_limit=0.5)
    elif name == "MCTS (0.5s)":
        from bots.mcts_x import MCTSGAgent
        return MCTSGAgent(name=name, time_limit=0.5)
    elif name == "Greedy":
        from bots.greedy import GreedyPlayer
        return GreedyPlayer(name=name)
    elif name == "Greedy Chain":
        from bots.greedy_improve import GreedyChainPlayer
        return GreedyChainPlayer(name=name)
    else:
        raise ValueError(f"Unknown agent name: {name}")

def run_single_matchup(args):
    agent1_name, agent2_name, size, num_games = args
    from game import DotsAndBoxesGame
    
    # Initialize agents inside the worker process
    print()
    agent1 = create_agent(agent1_name, size)
    agent2 = create_agent(agent2_name, size)
    
    a1_wins = 0
    a2_wins = 0
    draws = 0
    
    for i in tqdm(range(num_games), desc=f"Playing {agent1_name} vs {agent2_name}"):
        # Alternate who starts to ensure fairness
        starting_player = 1 if (i % 2 == 0) else -1
        game = DotsAndBoxesGame(size=size, starting_player=starting_player)
        
        while game.is_running():
            if game.current_player == 1:
                move = agent1.get_move(game)
            else:
                move = agent2.get_move(game)
            game.execute_move(move)
            
        if game.result == 1:
            a1_wins += 1
        elif game.result == -1:
            a2_wins += 1
        else:
            draws += 1
            
    return agent1_name, agent2_name, a1_wins, a2_wins, draws

def plot_heatmap(matrix, agent_names, output_path):
    plt.figure(figsize=(10, 8))
    if HAS_SEABORN:
        df = pd.DataFrame(matrix, index=agent_names, columns=agent_names)
        sns.heatmap(df, annot=True, cmap="coolwarm", vmin=0, vmax=1, fmt=".2f", cbar_kws={'label': 'Win Rate'})
    else:
        # Fallback to pure matplotlib if seaborn is not available
        plt.imshow(matrix, cmap="coolwarm", vmin=0, vmax=1)
        plt.colorbar(label='Win Rate')
        plt.xticks(np.arange(len(agent_names)), agent_names, rotation=45, ha="right")
        plt.yticks(np.arange(len(agent_names)), agent_names)
        for i in range(len(agent_names)):
            for j in range(len(agent_names)):
                plt.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", color="black")
                
    plt.title("Agent Win Rate Matrix (Row vs Column)\n(Value represents Row Agent's Win Rate vs Column Opponent)")
    plt.ylabel("Agent")
    plt.xlabel("Opponent")
    plt.tight_layout()
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Heatmap exported to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Dots and Boxes agents in a round-robin tournament.")
    parser.add_argument("--size", type=int, default=5, help="Grid size of the Dots and Boxes board (default: 3).")
    parser.add_argument("--games", type=int, default=40, help="Number of games to play per matchup (default: 20).")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes (default: cpu_count - 1).")
    parser.add_argument("--output", type=str, default="tournament_heatmap.png", help="Path to export the heatmap PNG (default: tournament_heatmap.png).")
    args = parser.parse_args()

    # List of agent names to include in the tournament
    agent_names = ["Random", "Alpha-Beta (0.1s)", "MCTS (0.1s)", "Alpha-Beta (0.5s)", "MCTS (0.5s)", "Greedy", "Greedy Chain"]
    n_agents = len(agent_names)
    
    # Create the tasks list for all distinct pairs (i < j)
    tasks = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            tasks.append((agent_names[i], agent_names[j], args.size, args.games))

    # Run matchups in parallel
    num_cores = args.workers if args.workers is not None else max(1, mp.cpu_count() - 1)
    print(f"Starting tournament: {n_agents} agents, size={args.size}x{args.size}, games={args.games} per matchup.")
    print(f"Running on {num_cores} worker processes...")
    
    results = []
    with mp.Pool(num_cores) as pool:
        for res in tqdm(pool.imap_unordered(run_single_matchup, tasks), total=len(tasks), desc="Running Matchups"):
            results.append(res)
            
    # Compile the winrate matrix
    winrate_matrix = np.zeros((n_agents, n_agents))
    # Self-winrate is 0.5
    np.fill_diagonal(winrate_matrix, 0.5)
    
    # Map name to matrix index
    name_to_idx = {name: idx for idx, name in enumerate(agent_names)}
    
    # Fill in matchup results
    for a1_name, a2_name, a1_wins, a2_wins, draws in results:
        idx1 = name_to_idx[a1_name]
        idx2 = name_to_idx[a2_name]
        
        # Winrate of Agent 1 against Agent 2
        wr1 = (a1_wins + 0.5 * draws) / args.games
        # Winrate of Agent 2 against Agent 1
        wr2 = (a2_wins + 0.5 * draws) / args.games
        
        winrate_matrix[idx1][idx2] = wr1
        winrate_matrix[idx2][idx1] = wr2

    # Print results to stdout
    print("\n--- Winrate Matrix (Row Agent vs Column Opponent) ---")
    header = f"{'':<20}" + "".join(f"{name:>18}" for name in agent_names)
    print(header)
    for i, name in enumerate(agent_names):
        row_str = f"{name:<20}"
        for j in range(n_agents):
            row_str += f"{winrate_matrix[i][j]:>18.2%}"
        print(row_str)
    print("-" * len(header))

    # Generate and save the heatmap
    plot_heatmap(winrate_matrix, agent_names, args.output)

if __name__ == "__main__":
    mp.freeze_support()
    main()
