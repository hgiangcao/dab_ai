import os
import sys
import time
import numpy as np
import torch

# Add distributed to path
sys.path.append(os.path.join(os.path.dirname(__file__), "distributed"))

from game import DotsAndBoxesGame
from model import NNetWrapper, dotdict
from mcts import MCTS, AZNode
import config

def main():
    print("MCTS Benchmark Tool")
    print("-" * 30)

    game = DotsAndBoxesGame(size=5)
    
    # 1. Setup network wrapper
    # We can load the latest model if it exists, otherwise use a random one
    eval_args = dotdict({
        'lr': config.LEARNING_RATE,
        'epochs': config.EPOCHS,
        'batch_size': config.BATCH_SIZE,
        'num_channels': 256,
        'num_res_blocks': 10,
        'l2_reg': 1e-4,
        'n_simulations': 200,
        'c_puct': config.MCTS_C_PUCT,
        'dirichlet_eps': 0.0,
        'dirichlet_alpha': config.MCTS_DIRICHLET_ALPHA,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })
    
    print(f"Device being used: {eval_args.device}")
    nnet = NNetWrapper(game, eval_args)
    
    # Try loading a model checkpoint if available
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
        "n_simulations": 200,
        "c_puct": config.MCTS_C_PUCT,
        "dirichlet_eps": 0.0,
        "dirichlet_alpha": config.MCTS_DIRICHLET_ALPHA,
    }
    mcts = MCTS(nnet, mcts_params)
    
    # Warm up (run a search to initialize CUDA/PyTorch graph states)
    print("Warming up model and MCTS...")
    temp_root = AZNode(parent=None, s=game, a=None)
    mcts.search(temp_root, is_root=True, current_depth=0)
    print("Warm up complete.\n")
    
    # ── Test 1: Measure time for 200 simulations ──
    print("Running Test 1: Time for 200 MCTS simulations...")
    start_time = time.perf_counter()
    
    test_root_1 = AZNode(parent=None, s=game, a=None)
    # Perform Leaf Expansion for root
    # Note: search expands if P is None
    for _ in range(200):
        mcts.search(test_root_1, is_root=True, current_depth=0)
        
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    print(f"  -> Time taken for 200 simulations: {elapsed_ms:.2f} ms")
    print(f"  -> Average time per simulation: {elapsed_ms / 200:.4f} ms")
    
    # ── Test 2: Measure simulations run in 100ms ──
    print("\nRunning Test 2: Number of simulations completed in 100ms...")
    test_root_2 = AZNode(parent=None, s=game, a=None)
    
    sim_count = 0
    time_limit = 0.1 # 100ms
    start_time = time.perf_counter()
    
    while (time.perf_counter() - start_time) < time_limit:
        mcts.search(test_root_2, is_root=True, current_depth=0)
        sim_count += 1
        
    actual_elapsed = (time.perf_counter() - start_time) * 1000
    print(f"  -> Simulations completed: {sim_count}")
    print(f"  -> Actual elapsed time: {actual_elapsed:.2f} ms")
    print(f"  -> Average time per simulation: {actual_elapsed / sim_count:.4f} ms")

if __name__ == "__main__":
    main()
