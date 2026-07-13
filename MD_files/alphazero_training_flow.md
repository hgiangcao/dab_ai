# AlphaZero Training Flow and TensorBoard Interpretation Guide

This guide details the step-by-step training pipeline for the Dots and Boxes AlphaZero agent, how metrics are tracked, and how to analyze learning health using TensorBoard.

---

## 1. Step-by-Step Training Flow

The AlphaZero trainer (`AlphaZeroTrainer.learn()`) runs for a configured number of iterations (`num_iters`). Each iteration performs five distinct phases:

### Phase 1: Self-Play (`execute_episode`)
1. The current neural network plays a series of games (`num_eps`) against itself.
2. During each turn, MCTS executes search simulations (`n_simulations`). 
3. The neural network's policy head predicts move probabilities, while the value head evaluates state valuations to guide the MCTS search.
4. Each move stores a state tuple: `(board_representation, move_probabilities_from_mcts, active_player)`. 
5. When the game ends, the terminal reward (1 for win, -1 for loss, 0 for draw) is back-propagated to value all states collected during that episode from the perspective of the player who made the move.

### Phase 2: Symmetry Augmentation
To prevent spatial/direction bias and multiply the training data by 8x, every collected board state is transformed into 8 structurally equivalent orientations:
* Three $90^\circ$ rotations.
* Horizontal reflection + its three rotations.

The augmented states are stored in a rolling history buffer (`maxlen_queue`).

### Phase 3: Neural Network Training (`model.train`)
The active neural network trains on batches of states randomly sampled from the history buffer. The optimizer minimizes a combined loss function:
* **Policy Loss ($L_{pi}$)**: Cross-entropy measuring how closely the network's move predictions match the search probabilities discovered by MCTS.
* **Value Loss ($L_v$)**: Mean squared error measuring how closely the network's win/loss evaluations match the actual game outcomes.
* **Total Loss**: The sum of Policy Loss + Value Loss ($L_{pi} + L_v$).

### Phase 4: Model Comparison (The Arena)
The newly trained network is evaluated competitively:
1. **Vs. Previous Best (`pnet`)**: Plays a series of games (`arena_games`). If the new network wins at least the `update_threshold` (e.g., 55%) of decisive games, it is **accepted** and replaces the previous best network. If not, the new network is **rejected**, and its weights are reset to match the old network.
2. **Vs. Baseline (`AlphaBetaPlayer`)**: Plays 10 games against a deterministic Alpha-Beta heuristic bot to track progress against a static minimax opponent.

### Phase 5: Checkpointing
Saves the current iteration model (`checkpoint_i.pth.tar`). If the model was accepted in Phase 4, it also overwrites `best.pth.tar`.

---

## 2. Evaluation Metrics Logged to TensorBoard

Metrics are written to TensorBoard at the end of every training iteration:

| Tag Name | Type | Description |
| :--- | :--- | :--- |
| **`Loss/Policy_Loss`** | Optimization | Average cross-entropy loss of the policy head. |
| **`Loss/Value_Loss`** | Optimization | Average mean squared error of the value head. |
| **`Loss/Total_Loss`** | Optimization | Combined loss ($L_{pi} + L_v$). |
| **`SelfPlay/Game_Length`** | Game | Average number of turns to complete a self-play game. |
| **`SelfPlay/Memory_Size`** | Memory | Total count of training examples currently stored in the rolling buffer. |
| **`Arena/Win_Rate_Vs_Old`** | Arena | Win rate (0.0 to 1.0) of the new network against the previous best network. |
| **`Baseline/Win_Rate_Vs_Heuristic`** | Arena | Win rate (0.0 to 1.0) of the new network against the deterministic Alpha-Beta baseline. |
| **`MCTS/Average_Tree_Depth`** | MCTS | The average maximum depth reached by MCTS search trees during self-play. |

---

## 3. How to Read TensorBoard Results

Start TensorBoard (automatically done by `./start_training.sh`) and navigate to `http://localhost:6006`.

### Healthy Learning Signals
* **`Loss/Total_Loss` steadily decreases**: The network is extracting patterns. Policy loss usually drops rapidly, while value loss decreases gradually.
* **`Baseline/Win_Rate_Vs_Heuristic` trends upward**: Your agent is successfully shifting from random moves to tactical gameplay, outplaying standard heuristics.
* **`Arena/Win_Rate_Vs_Old` hovers around 0.50 - 0.60**: This is healthy! Because the opponent (`pnet`) updates to match the new network whenever it gets better, the agent is continuously playing a stronger version of itself. A flat line around 50% means the network is continually improving.

### Unhealthy Learning Signals (Troubleshooting)
* **`Arena/Win_Rate_Vs_Old` falls to 0.0 consistently**: The new network is losing to the old network, resulting in perpetual rejections. This suggests the learning rate (`lr`) might be too high, causing gradient explosion, or the training dataset size is too small.
* **`MCTS/Average_Tree_Depth` is extremely low (e.g. 1 or 2)**: The network's policy head is predicting highly biased/peaked probabilities too early, preventing MCTS from exploring branches. Consider increasing Dirichlet noise parameters (`dirichlet_eps`) to encourage exploration.
