### Task: Implement Reverse Curriculum from JSON Game Logs

Modify the AlphaZero training loop to initialize self-play games from states loaded from a JSON file logs/game_logs.jsonl, gradually decaying the starting depth to 0%.
(reference to check_health_logs.py if neeeded)

**1. Curriculum Schedule Setup**
*   Define `start_fill_pct = 0.70` (70% of the game sequence).
*   Define `decay_iterations = int(total_iterations * 0.5)`. 
*   Calculate `decay_step = 0.70 / decay_iterations`.
*   At the end of each training iteration, update `start_fill_pct = max(0.0, start_fill_pct - decay_step)`.

**2. Episode Initialization (`execute_episode`)**
*   At the beginning of each self-play episode, check if `start_fill_pct > 0`.
*   If yes:
    1. Randomly sample one game sequence from the loaded JSON logs.
    2. Calculate `target_move_index = int(len(json_game_moves) * start_fill_pct)`.
    3. Apply the moves from `index 0` to `target_move_index` directly to the `DotsAndBoxesGame` state. Do not use MCTS and do not store these pre-filled moves in the replay buffer.
*   Once the board reaches the target state, hand control to AlphaZero to finish the game using MCTS. 
*   Only append the states, policies, and values generated *after* the handoff to the training replay buffer.
*   If `start_fill_pct == 0.0`, start from a completely empty board as normal.