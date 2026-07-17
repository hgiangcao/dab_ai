import numpy as np
import random
from game import DotsAndBoxesGame

def run_5x5_simulation():
    print("\n--- Running 5x5 Simulation with identical move sequence ---")
    size = 5
    
    # We want to play two games with the exact same sequence of moves.
    # To do this, we'll first play a game without early stopping, record the move sequence,
    # and then replay it on a game with early stopping.
    
    # 1. Play game without early stopping to completion
    game_no_early = DotsAndBoxesGame(size=size, early_stopping=False, starting_player=1)
    # Fix the random seed for reproducible random moves
    random.seed(42)
    np.random.seed(42)
    
    moves_played = []
    while game_no_early.is_running():
        valid_moves = game_no_early.get_valid_moves()
        move = random.choice(valid_moves)
        moves_played.append(move)
        game_no_early.execute_move(move)
        
    total_moves_no_early = len(moves_played)
    p1_score_no_early = int(np.sum(game_no_early.b == 1))
    p2_score_no_early = int(np.sum(game_no_early.b == -1))
    winner_no_early = game_no_early.result
    
    print(f"[No Early Stopping] Game finished. Total moves: {total_moves_no_early}")
    print(f"                    Final Score -> P1: {p1_score_no_early} | P2: {p2_score_no_early} | Winner: {winner_no_early}")
    
    # 2. Replay the same moves in a game WITH early stopping
    game_yes_early = DotsAndBoxesGame(size=size, early_stopping=True, starting_player=1)
    
    moves_played_yes_early = 0
    for move in moves_played:
        if not game_yes_early.is_running():
            break
        game_yes_early.execute_move(move)
        moves_played_yes_early += 1
        
    p1_score_yes_early = int(np.sum(game_yes_early.b == 1))
    p2_score_yes_early = int(np.sum(game_yes_early.b == -1))
    winner_yes_early = game_yes_early.result
    
    print(f"[With Early Stopping] Game finished. Total moves: {moves_played_yes_early}")
    print(f"                     Score at finish -> P1: {p1_score_yes_early} | P2: {p2_score_yes_early} | Winner: {winner_yes_early}")
    
    # Assert correctness
    assert winner_yes_early == winner_no_early, "Winner mismatch between early stopping and full completion!"
    assert moves_played_yes_early < total_moves_no_early, "Early stopping should have finished in fewer moves!"
    print("5x5 Early Stopping correctness & move counts verified successfully!")

def test_early_stopping():
    print("Running early stopping checks...")

    # Scenario 1: early_stopping=False (Default)
    # The game should NOT stop early even if mathematically decided.
    size = 3
    game_default = DotsAndBoxesGame(size=size, early_stopping=False)
    
    # Let's say Player 1 gets 6 boxes, Player 2 gets 0 boxes, 3 remain.
    # Player 1 has already won mathematically (6 > 0 + 3), but game is not finished yet.
    game_default.b = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0]
    ])
    
    game_default.check_finished()
    print(f"[Default, Early Stopping = False] Leads 6 to 0 (3 remaining). Result: {game_default.result} (Expected: None)")
    assert game_default.result is None, "Should not terminate early when early_stopping=False"

    # Fill up the rest of the boxes.
    # Player 2 captures the remaining 3 boxes. P1 has 6, P2 has 3. P1 wins.
    game_default.b = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [-1, -1, -1]
    ])
    game_default.check_finished()
    print(f"[Default, Early Stopping = False] Fully finished (6 to 3). Result: {game_default.result} (Expected: 1)")
    assert game_default.result == 1, f"Winner should be 1, got {game_default.result}"

    # Scenario 2: early_stopping=True
    # The game should stop early as soon as the result is mathematically decided.
    game_early = DotsAndBoxesGame(size=size, early_stopping=True)
    
    # Case A: Player 1 leads mathematically
    # Player 1 has 5, Player 2 has 0, 4 remaining.
    # 5 > 0 + 4 -> Player 1 wins mathematically.
    game_early.b = np.array([
        [1, 1, 1],
        [1, 1, 0],
        [0, 0, 0]
    ])
    game_early.check_finished()
    print(f"[Early Stopping = True] Player 1 leads 5 to 0 (4 remaining). Result: {game_early.result} (Expected: 1)")
    assert game_early.result == 1, f"Expected 1, got {game_early.result}"

    # Case B: Player 2 leads mathematically
    # Player 2 has 6, Player 1 has 1, 2 remaining.
    # 6 > 1 + 2 -> Player 2 wins mathematically.
    game_early_p2 = DotsAndBoxesGame(size=size, early_stopping=True)
    game_early_p2.b = np.array([
        [-1, -1, -1],
        [-1, -1, -1],
        [1, 0, 0]
    ])
    game_early_p2.check_finished()
    print(f"[Early Stopping = True] Player 2 leads 6 to 1 (2 remaining). Result: {game_early_p2.result} (Expected: -1)")
    assert game_early_p2.result == -1, f"Expected -1, got {game_early_p2.result}"

    # Case C: Not yet mathematically decided
    # Player 1 has 4, Player 2 has 1, 4 remaining.
    # 4 is not > 1 + 4 (which is 5). Opponent could still win or draw.
    game_undecided = DotsAndBoxesGame(size=size, early_stopping=True)
    game_undecided.b = np.array([
        [1, 1, 1],
        [1, -1, 0],
        [0, 0, 0]
    ])
    game_undecided.check_finished()
    print(f"[Early Stopping = True] Undecided 4 to 1 (4 remaining). Result: {game_undecided.result} (Expected: None)")
    assert game_undecided.result is None, f"Expected None, got {game_undecided.result}"

    # Scenario 3: Draw verification (fully finished)
    game_draw = DotsAndBoxesGame(size=2, early_stopping=True)
    # A 2x2 board has 4 boxes. Player 1 gets 2, Player 2 gets 2.
    game_draw.b = np.array([
        [1, -1],
        [1, -1]
    ])
    game_draw.check_finished()
    print(f"[Draw verification] Finished 2 to 2. Result: {game_draw.result} (Expected: 0)")
    assert game_draw.result == 0, f"Expected 0, got {game_draw.result}"

    print("All checks completed successfully!")

if __name__ == "__main__":
    test_early_stopping()
    run_5x5_simulation()
