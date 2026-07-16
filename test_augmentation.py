import os
import json
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "distributed"))
import torch
from game import DotsAndBoxesGame
from distributed.pretrained import game_record_to_examples, GameDataset

def main():
    print("Testing Augmentation logic...")
    
    # 1. Load an example from game_logs.jsonl
    log_path = "game_logs.jsonl"
    if not os.path.exists(log_path):
        print(f"File {log_path} not found.")
        return

    with open(log_path, "r") as f:
        line = f.readline()
        record = json.loads(line)

    examples = game_record_to_examples(record)
    print(f"Loaded {len(examples)} examples from first game record.")
    
    board_state, pi, value = examples[5] # Pick a random state
    
    size = board_state.shape[1] - 1
    
    # 2. Check transformations directly
    dataset = GameDataset([examples[5]])
    
    assert len(dataset) == 8, f"Expected 8 augmentations, got {len(dataset)}"
    print("Augmentation count is 8. (OK)")
    
    for i in range(8):
        aug_board, aug_pi, aug_val = dataset[i]
        aug_board = aug_board.numpy()
        aug_pi = aug_pi.numpy()
        aug_val = aug_val.item()
        
        assert aug_val == value, "Value target must be unchanged."
        
        # Check alignment of policy and board
        h_ch = aug_board[0, :size + 1, :size]
        v_ch = aug_board[1, :size, :size + 1]
        
        aug_lines = DotsAndBoxesGame.h_v_to_l(h_ch, v_ch)
        
        # Valid moves on aug_board are where aug_lines == 0
        valid_moves = np.where(aug_lines == 0)[0]
        
        # Check that aug_pi only has non-zero probability on valid moves
        # (Actually, MCTS might explore a bit, but pi should be mostly 0 for drawn lines)
        # In our case, drawn lines MUST have exactly 0 probability.
        invalid_moves = np.where(aug_lines != 0)[0]
        if len(invalid_moves) > 0:
            prob_on_invalid = aug_pi[invalid_moves].sum()
            assert prob_on_invalid < 1e-5, f"Transform {i}: Found probability {prob_on_invalid} on invalid moves!"
        
        # Check box correctness
        boxes_p1 = aug_board[2, :size, :size]
        boxes_m1 = aug_board[3, :size, :size]
        boxes = boxes_p1 - boxes_m1
        
        # Reconstruct boxes from lines to verify alignment
        # This checks if boxes and lines transformed together correctly
        game = DotsAndBoxesGame(size=size)
        game.l = aug_lines
        
        # For a given line state, let's see if the boxes match
        # Actually, reconstructing boxes exactly from lines alone isn't trivial if we don't know who captured them
        # But we can at least check if completed boxes have non-zero value, and incomplete have 0.
        for r in range(size):
            for c in range(size):
                lines_of_box = game.get_lines_of_box((r, c))
                filled_lines = np.count_nonzero(aug_lines[lines_of_box])
                if filled_lines < 4:
                    assert boxes[r, c] == 0, f"Transform {i}: Box {r},{c} is incomplete but marked as captured."
                else:
                    # In this game state, if 4 lines are drawn, it must be captured by someone
                    assert boxes[r, c] != 0, f"Transform {i}: Box {r},{c} is complete but not marked as captured."

    print("All tests passed! Data augmentation alignment is correct.")

if __name__ == "__main__":
    main()
