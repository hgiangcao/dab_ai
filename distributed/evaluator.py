import os
import sys
import numpy as np
import concurrent.futures
from tqdm import tqdm
import multiprocessing
import torch

# Add parent dir to path to import game, model, mcts
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import config
import model_manager
from game import DotsAndBoxesGame
from model import NNetWrapper, dotdict
from mcts import MCTS

# Default evaluation arguments
eval_args = dotdict({
    'lr': config.LEARNING_RATE,
    'epochs': config.EPOCHS,
    'batch_size': config.BATCH_SIZE,
    'num_channels': 256,
    'num_res_blocks': 10, 
    'n_simulations': 100,
    'c_puct': 1.0,
    'device': 'cpu'  # CPU allows us to run evaluation in parallel efficiently
})

def _worker_play_single(worker_args):
    """
    Isolated worker to execute a single evaluation match.
    Runs one game in parallel.
    """
    candidate_path, best_path, p1_starts = worker_args
    
    game = DotsAndBoxesGame(size=5)
    
    # 1. Candidate Model
    cand_net = NNetWrapper(game, eval_args)
    cand_state = torch.load(candidate_path, map_location='cpu', weights_only=True)
    cand_net.nnet.load_state_dict(cand_state['state_dict'] if 'state_dict' in cand_state else cand_state)
    cand_net.nnet.eval()
    mcts_cand = MCTS(cand_net, eval_args)
    
    def agent_cand(g):
        pi = mcts_cand.play(g, temp=0)
        return np.argmax(pi)
        
    # 2. Best Model (or fallback to random if none exists)
    if os.path.exists(best_path):
        best_net = NNetWrapper(game, eval_args)
        best_state = torch.load(best_path, map_location='cpu', weights_only=True)
        best_net.nnet.load_state_dict(best_state['state_dict'] if 'state_dict' in best_state else best_state)
        best_net.nnet.eval()
        mcts_best = MCTS(best_net, eval_args)
        
        def agent_best(g):
            pi = mcts_best.play(g, temp=0)
            return np.argmax(pi)
    else:
        import random
        def agent_best(g):
            valid = g.get_valid_moves()
            return random.choice(valid) if valid else None
            
    # Assign turns
    players = {1: agent_cand, -1: agent_best} if p1_starts else {1: agent_best, -1: agent_cand}
    
    # Play
    while game.is_running():
        cur_player = game.current_player
        action = players[cur_player](game)
        game.execute_move(action)
        
    # Return 1 if candidate won, -1 if best won, 0 for draw
    return game.result if p1_starts else -game.result


def play_game(model1_path, model2_path, p1_starts=True):
    """
    Play one game.
    Return:
        1 for MODEL1 (candidate)
       -1 for MODEL2 (best)
        0 for DRAW
    """
    return _worker_play_single((model1_path, model2_path, p1_starts))


def evaluate_model(candidate_model_path, best_model_path, num_games):
    """
    Run evaluation games in parallel across multiple CPU cores.
    Return a summary dictionary.
    """
    wins = 0
    losses = 0
    draws = 0
    
    worker_args_list = []
    half_games = num_games // 2
    for idx in range(num_games):
        p1_starts = idx < half_games
        worker_args_list.append((candidate_model_path, best_model_path, p1_starts))
        
    mp_context = multiprocessing.get_context('spawn')
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Evaluating candidate model over {num_games} arena games using {num_workers} processes...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
        futures = [executor.submit(_worker_play_single, arg) for arg in worker_args_list]
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_games, desc="Arena Eval"):
            try:
                res = future.result()
                if res == 1:
                    wins += 1
                elif res == -1:
                    losses += 1
                else:
                    draws += 1
            except Exception as e:
                print(f"Match execution failed: {e}")
                
    total_decisive = wins + losses
    win_rate = wins / total_decisive if total_decisive > 0 else 0.5
    
    return {
        "wins": wins,
        "loss": losses,
        "draw": draws,
        "win_rate": win_rate
    }


def should_promote(result):
    """
    Decide if candidate becomes best based on win_rate > PROMOTION_THRESHOLD
    """
    return result["win_rate"] >= config.PROMOTION_THRESHOLD


def evaluate_new_model():
    """
    High-level evaluation:
    1. Load latest candidate
    2. Load best
    3. Run games
    4. Decide promotion
    """
    latest_path = model_manager.get_latest_model_path()
    best_path = model_manager.get_best_model_path()
    
    # In distributed mode, trainer saves candidate model temporarily 
    # to evaluate before officially committing it as a version.
    candidate_path = os.path.join(config.get_current_model_dir(), "checkpoint_candidate.pth.tar")
    if not os.path.exists(candidate_path):
        candidate_path = latest_path
        
    if not os.path.exists(candidate_path):
        print(f"No candidate model found at {candidate_path} for evaluation.")
        return False
        
    result = evaluate_model(candidate_path, best_path, config.EVAL_GAMES)
    
    print("\n================ EVALUATION RESULTS ================")
    print(f" Wins:   {result['wins']}")
    print(f" Losses: {result['loss']}")
    print(f" Draws:  {result['draw']}")
    print(f" Win Rate: {result['win_rate']:.2%}")
    print("====================================================")
    
    if should_promote(result):
        print(f"Result exceeds threshold ({config.PROMOTION_THRESHOLD:.2%}). Model promoted to BEST.")
        model_manager.promote_best_model()
        return True
    else:
        print("Model rejected.")
        return False

if __name__ == "__main__":
    evaluate_new_model()