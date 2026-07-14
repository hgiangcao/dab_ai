import os
import json
import random
import time
import multiprocessing as mp
from tqdm import tqdm

from game import DotsAndBoxesGame
from bots.mcts_x import MCTS100Agent, MCTS1000Agent
from bots.alpha_beta import AlphaBetaPlayer

def generate_single_game(args):
    """Worker function to generate one game."""
    game_idx, target_fill_pct = args
    
    # Initialize bots inside the worker process
    bot_mcts100 = MCTS100Agent()
    bot_mcts1000 = MCTS1000Agent()
    bot_alphabeta = AlphaBetaPlayer(depth=2)
    
    game = DotsAndBoxesGame(size=5)
    bot_idx = game_idx % 3
    bot = bot_mcts100 if bot_idx == 0 else (bot_mcts1000 if bot_idx == 1 else bot_alphabeta)
    
    game_moves = []
    game_policies = []
    
    # Phase 1: Random fill
    total_lines_to_fill = int(game.N_LINES * target_fill_pct)
    while len(game_moves) < total_lines_to_fill and game.is_running():
        valid_moves = game.get_valid_moves()
        if not valid_moves: break
        move = random.choice(valid_moves)
        game.execute_move(move)
        
        game_moves.append(int(move))
        dummy_pi = [0.0] * game.N_LINES
        dummy_pi[move] = 1.0
        game_policies.append(dummy_pi)
        
    # Phase 2: Strong bot finishes
    while game.is_running():
        move = bot.get_move(game)
        game_moves.append(int(move))
        
        pi = [0.0] * game.N_LINES
        pi[move] = 1.0
        game_policies.append(pi)
        game.execute_move(move)
        
    return {
        "winner": int(game.result),
        "moves": game_moves,
        "policies": game_policies
    }

def generate_backward_data_parallel(output_filepath: str, num_games: int, target_fill_pct: float = 0.75, num_workers: int = None):
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # Setup multiprocessing arguments
    tasks = [(i, target_fill_pct) for i in range(num_games)]
    num_cores = num_workers if num_workers is not None else max(1, mp.cpu_count() - 1)
    
    print(f"Starting parallel generation of {num_games} matches using {num_cores} cores...")
    
    with mp.Pool(num_cores) as pool, open(output_filepath, 'a') as f:
        # imap_unordered is fastest; order of generated games doesn't matter
        for record in tqdm(pool.imap_unordered(generate_single_game, tasks), total=num_games, desc="Generating Games"):
            f.write(json.dumps(record) + '\n')
            f.flush()

if __name__ == "__main__":
    OUTPUT_FILE = "game_logs.jsonl"
    TOTAL_GAMES = 10000
    FILL_PERCENTAGE = 0.75
    NUM_WORKERS = None #None  # Set an integer (e.g. 4) to limit CPU cores, or None to use all-but-one core
    
    # Requires this wrapper on Windows/macOS for multiprocessing
    mp.freeze_support() 
    
    generate_backward_data_parallel(
        output_filepath=OUTPUT_FILE, 
        num_games=TOTAL_GAMES, 
        target_fill_pct=FILL_PERCENTAGE,
        num_workers=NUM_WORKERS
    )
    print(f"Done! Data appended to {OUTPUT_FILE}")