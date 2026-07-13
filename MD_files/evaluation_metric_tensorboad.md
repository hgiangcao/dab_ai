# AlphaZero TensorBoard Logging Metrics

## 1. Loss Metrics
*   **`Loss/Policy_Loss`**: Difference between network predicted probabilities and MCTS search probabilities. Tracks policy convergence.
*   **`Loss/Value_Loss`**: Mean Squared Error between predicted outcome and actual game result. Tracks value head accuracy.
*   **`Loss/Total_Loss`**: Sum of policy, value, and regularization losses.

## 2. Self-Play Metrics
*   **`SelfPlay/Game_Length`**: Average number of moves per game. Useful for spotting early resignation or invalid move loops.
*   **`SelfPlay/Memory_Size`**: Current count of training examples in the replay buffer. Confirms correct memory/queue management.

## 3. Evaluation Metrics
*   **`Arena/Win_Rate_Vs_Old`**: Win rate of the new network against the previous best network. Determines model promotion.
*   **`Baseline/Win_Rate_Vs_Heuristic`**: Win rate against fixed baselines (e.g., MCTS1000 or Alpha-Beta). Measures absolute skill progression over time.

## 4. Search Dynamics
*   **`MCTS/Average_Tree_Depth`**: Average maximum depth reached by MCTS. A rising value indicates the network's value predictions are effectively narrowing the search space.