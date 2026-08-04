import torch
import numpy as np
from model import NNetWrapper, dotdict
from game import DotsAndBoxesGame

def run_overfit_test():
    args = dotdict({
        'lr': 0.05,
        'epochs': 3000,
        'batch_size': 4,
        'num_channels': 32,
        'num_res_blocks': 2,
        'l2_reg': 0.0,
        'device': 'cpu'
    })
    
    # 2x2 board gives 12 possible actions (2*3*2 = 12 lines)
    game = DotsAndBoxesGame(size=2)
    nnet = NNetWrapper(game, args)
    
    board_x, board_y = game.getBoardSize() if hasattr(game, 'getBoardSize') else (game.SIZE, game.SIZE)
    action_size = game.getActionSize() if hasattr(game, 'getActionSize') else game.N_LINES
    
    # Create a small fixed batch of 4 examples
    examples = []
    for i in range(4):
        lines = np.zeros(action_size, dtype=np.float32)
        lines[i % action_size] = 1.0
        
        boxes = np.zeros((game.SIZE, game.SIZE), dtype=np.float32)
        
        pi = np.zeros(action_size, dtype=np.float32)
        pi[i % action_size] = 1.0
        
        v = np.float32(1.0 if i % 2 == 0 else -1.0)
        examples.append((lines, boxes, pi, v))
        
    print("Starting 1000-epoch single-batch overfit test...")
    # Train directly
    pi_loss, v_loss, total_loss, entropy = nnet.train(examples, epochs=args.epochs)
    
    unweighted_total_loss = pi_loss + v_loss
    
    print(f"\nResult after {args.epochs} epochs:")
    print(f"Policy Loss: {pi_loss:.6f}")
    print(f"Value Loss:  {v_loss:.6f}")
    print(f"Weighted Total Loss: {total_loss:.6f}")
    print(f"Unweighted Total Loss: {unweighted_total_loss:.6f}")
    print(f"Entropy:     {entropy:.6f}")
    
    if unweighted_total_loss < 1e-4:
        print("Test PASSED: Unweighted total loss effectively reached 0.0")
    else:
        print("Test FAILED: Unweighted total loss did not reach 0.0. Loss calculation or backprop may be broken.")

if __name__ == "__main__":
    run_overfit_test()
