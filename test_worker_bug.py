import traceback
from DotsAndBoxesAlphaZero import DotsAndBoxesGame
import numpy as np

def run_test():
    try:
        from coach import worker_execute_episode_chunk
        from config import PHASES_CONFIG
        
        # Test selfplay logic for phase 6
        game_size = 5
        # No game sequence
        episode_specs = [
            (None, 0.0, "self", None)
        ] * 10
        
        import torch
        from evaluate_Alphzero_agent import AlphaZeroAgent
        from model import NNetWrapper, dotdict
        from mcts import MCTS
        
        args = dotdict({
            'num_eps': 10,
            'temp_threshold': 15,
            'temperature_initial': 1.0,
            'temperature_final': 0.0,
            'n_simulations': 25,
            'c_puct': 1.0,
            'dirichlet_alpha': 0.3,
            'dirichlet_eps': 0.25,
            'num_channels': 64,
            'dropout': 0.3,
            'kernel_size': 3,
            'num_res_blocks': 5,
            'lr': 0.001,
            'l2_reg': 1e-4
        })
        
        dummy_game = DotsAndBoxesGame(size=game_size, starting_player=1, early_stopping=True)
        net = NNetWrapper(dummy_game, args)
        
        import io
        torch.save({'state_dict': net.nnet.state_dict()}, "test_best_model.pth")
        
        worker_args = (
            game_size, "test_best_model.pth", MCTS, args, episode_specs
        )
        
        print("Running worker...")
        results = worker_execute_episode_chunk(worker_args)
        print("Worker finished without error.")
    except Exception as e:
        print("Error encountered:")
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
