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

# Import bots from tournament.py list
from bots.greedy import GreedyPlayer
from bots.greedy_improve import GreedyChainPlayer
from bots.ucla_bot import UCLABot, UCLABot_v2, UCLABot_v3
from bots.ucla_bot_heuristic import UCLAGreedyBot
from bots.ucla_alpha_beta import UCLAAlphaBeta
from bots.ucla_mcts import UCLAMCTS
from bots.simple_bot import SimpleBot
from bots.mcts_x import MCTSGAgent

class RandomAgent(BaseAgent):
    def __init__(self, name: str = "Random"):
        super().__init__(name)

    def get_move(self, game_state) -> int:
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None
        return random.choice(valid_moves)

_nnet_cache = {}

class AlphaZeroAgent(BaseAgent):
    def __init__(self, name: str = "AlphaZero", n_simulations=500, model_path="best.pth.tar"):
        super().__init__(name)
        import torch
        import config
        from model import NNetWrapper, dotdict
        from mcts import MCTS
        
        self.eval_args = dotdict({
            'lr': config.LEARNING_RATE,
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'num_channels': 256,
            'num_res_blocks': 10, 
            'l2_reg': 1e-4,
            'n_simulations': n_simulations,
            'c_puct': config.MCTS_C_PUCT,
            'dirichlet_eps': 0.0,
            'dirichlet_alpha': config.MCTS_DIRICHLET_ALPHA,
            'device': 'cpu'
        })
        
        global _nnet_cache
        if model_path not in _nnet_cache:
            self.game_ref = DotsAndBoxesGame(size=5)
            net = NNetWrapper(self.game_ref, self.eval_args)
            if os.path.exists(model_path):
                try:
                    state = torch.load(model_path, map_location='cpu', weights_only=False)
                    net.nnet.load_state_dict(state['state_dict'] if 'state_dict' in state else state)
                    print(f"Loaded AlphaZero model from {model_path}")
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
            else:
                print(f"Warning: AlphaZero model not found at {model_path}")
            net.nnet.eval()
            _nnet_cache[model_path] = net
            
        self.net = _nnet_cache[model_path]
        self.mcts = MCTS(self.net, self.eval_args)

    def get_move(self, game_state: DotsAndBoxesGame) -> int:
        pi = self.mcts.play(game_state, temp=0)
        return int(np.argmax(pi))

def create_agent(name: str, size: int, model_path: str = "best.pth.tar"):
    if name == "Random":
        return RandomAgent()
    elif name == "AlphaZero":
        return AlphaZeroAgent(model_path=model_path)
    elif name == "MCTS (100sims)":
        return MCTSGAgent(name=name, n_simulations=100)
    elif name == "MCTS (200sims)":
        return MCTSGAgent(name=name, n_simulations=200)
    elif name == "Greedy":
        return GreedyPlayer(name=name)
    elif name == "Greedy Chain":
        return GreedyChainPlayer(name=name)
    elif name == "UCLABot":
        return UCLABot(name=name)
    elif name == "UCLABot_v2":
        return UCLABot_v2(name=name)
    elif name == "UCLABot_v3":
        return UCLABot_v3(name=name)
    elif name == "UCLAGreedyBot":
        return UCLAGreedyBot(name=name)
    elif name == "UCLAAlphaBeta":
        return UCLAAlphaBeta(name=name)
    elif name == "UCLA_MCTS_100":
        return UCLAMCTS(name=name, n_simulations=100, time_limit=None)
    elif name == "UCLA_MCTS_200":
        return UCLAMCTS(name=name, n_simulations=200, time_limit=None)
    elif name == "SimpleBot":
        return SimpleBot(name=name)
    else:
        raise ValueError(f"Unknown agent name: {name}")

def run_single_game(args):
    agent1_name, agent2_name, size, game_index, model_path = args
    from game import DotsAndBoxesGame
    
    agent1 = create_agent(agent1_name, size, model_path)
    agent2 = create_agent(agent2_name, size, model_path)
    starting_player = 1 if (game_index % 2 == 0) else -1
    game = DotsAndBoxesGame(size=size, starting_player=starting_player, early_stopping=True)
    
    while game.is_running():
        if game.current_player == 1:
            move = agent1.get_move(game)
        else:
            move = agent2.get_move(game)
        game.execute_move(move)
        
    return agent1_name, agent2_name, game.result

def main():
    parser = argparse.ArgumentParser(description="Evaluate AlphaZero agent against other bots.")
    parser.add_argument("--size", type=int, default=5, help="Grid size of the Dots and Boxes board (default: 5).")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play per matchup (default: 20).")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes (default: cpu_count - 1).")
    parser.add_argument("--output", type=str, default="alphazero_evaluation.png", help="Path to export the results visualization PNG (default: alphazero_evaluation.png).")
    parser.add_argument("--run", type=str, default=None, help="The run subdirectory name under logs (e.g. run_1) to load best.pth.tar from. If not specified, loads from project root.")
    args = parser.parse_args()

    import config
    if args.run:
        model_path = os.path.join(config.LOGS_DIR, args.run, "best.pth.tar")
    else:
        model_path = "best.pth.tar"

    opponent_names = [
        # "Random",
        "Greedy", 
        "Greedy Chain",
        "UCLABot_v3",
        # "UCLA_MCTS_100",
        "SimpleBot"
    ]

    tasks = []
    # Play AlphaZero (Player 1) vs Opponents (Player 2)
    # Tasks are parallelized per individual game
    for opp in opponent_names:
        for game_idx in range(args.games):
            tasks.append(("AlphaZero", opp, args.size, game_idx, model_path))

    num_cores = args.workers if args.workers is not None else max(1, mp.cpu_count() - 1)
    print(f"Starting AlphaZero evaluation (parallelized per game) against {len(opponent_names)} bots (size={args.size}x{args.size}, games={args.games} per matchup).")
    print(f"Running on {num_cores} worker processes...")
    
    matchup_stats = {opp: {"wins": 0, "losses": 0, "draws": 0} for opp in opponent_names}
    
    with mp.Pool(num_cores) as pool:
        pbar = tqdm(pool.imap_unordered(run_single_game, tasks), total=len(tasks), desc="Running Games")
        for res in pbar:
            a1_name, a2_name, result = res
            opp = a2_name
            if result == 1:
                matchup_stats[opp]["wins"] += 1
            elif result == -1:
                matchup_stats[opp]["losses"] += 1
            else:
                matchup_stats[opp]["draws"] += 1
                
            total_played = sum(matchup_stats[opp].values())
            wr = (matchup_stats[opp]["wins"] + 0.5 * matchup_stats[opp]["draws"]) / total_played
            pbar.set_postfix_str(f"Latest {opp} WR: {wr:.2f}")

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
    plt.title(f'AlphaZero vs. Other Bots ({args.games} Games, Size {args.size}x{args.size})')
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
