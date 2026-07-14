# Project Structure & AlphaZero Training Components

This document outlines the purpose of each file and directory in the repository to ensure all components of the AlphaZero training pipeline, evaluation tools, and benchmark bots are present and correctly integrated.

---

## Core AlphaZero Training Pipeline

These files form the primary loop, model definition, game logic, and search algorithms required for self-play reinforcement learning.

### 1. Training Orchestration & Setup
*   **[main.py](file:///home/user/dab_ai/main.py)**: The main execution entrypoint. It configures the training hyperparameters, instantiates the game, wraps the neural network, and runs the training loop via `Coach`.
*   **[coach.py](file:///home/user/dab_ai/coach.py)**: Contains the `Coach` class. It manages self-play episodes (`execute_episode`), saves checkpoints, and handles the backward/curriculum training logic (pre-filling boards to decay start states) to train the network iteratively.
*   **[start_training.sh](file:///home/user/dab_ai/start_training.sh)**: A helper shell script to launch or resume the AlphaZero training loop.

### 2. Neural Network
*   **[model.py](file:///home/user/dab_ai/model.py)**: Contains the neural network architecture and wrapper.
    *   `DotsAndBoxesNet`: A PyTorch CNN with residual blocks, a policy head (predicting move probabilities), and a value head (predicting expected game outcomes).
    *   `NNetWrapper`: Handles training epochs, loss computation (MSE value loss + cross-entropy policy loss), batching, device placement (CPU/CUDA), and saving/loading model weights.

### 3. Search Algorithm (MCTS)
*   **[mcts.py](file:///home/user/dab_ai/mcts.py)**: Contains the Monte Carlo Tree Search logic.
    *   `AZNode`: Represents a state and statistics (Q-values, visit counts, priors) in the search tree.
    *   `MCTS`: Executes simulations using the network's predictions for leaf evaluation, including Dirichlet noise on the root node for exploration.
    *   `MCTSAgent`: A player wrapper that uses MCTS to select the best move.

### 4. Game Engine
*   **[game.py](file:///home/user/dab_ai/game.py)**: Core engine for Dots and Boxes. Contains logic for move execution (`execute_move`), board validation (`get_valid_moves`), winner checks, rotations/reflections for data augmentation, and canonical board formatting (4-plane representation: horizontal lines, vertical lines, P1 boxes, P2 boxes) expected by the neural network.
*   **[DotsAndBoxesAlphaZero.py](file:///home/user/dab_ai/DotsAndBoxesAlphaZero.py)**: An adapter class (`DotsAndBoxesAlphaZeroWrapper`) designed to wrap the game engine to match generic AlphaZero pipeline APIs (e.g., `getInitBoard`, `getNextState`, `getValidMoves`, `getGameEnded`).

---


## Benchmark Players & Bots (`bots/`)

These agents are used as baseline players, training opponents, or data generation tools.

*   **[agent_interface.py](file:///home/user/dab_ai/agent_interface.py)**: Declares the common interface `BaseAgent` and the `HumanAgent` class which all bots implement.
*   **[bots/alpha_beta.py](file:///home/user/dab_ai/bots/alpha_beta.py)**: A minimax agent using depth-limited Alpha-Beta pruning search and basic box-count difference heuristics.
*   **[bots/alpha_beta_v1.py](file:///home/user/dab_ai/bots/alpha_beta_v1.py)**: Legacy/first iteration of the Alpha-Beta minimax agent.
*   **[bots/alpha_beta_v2.py](file:///home/user/dab_ai/bots/alpha_beta_v2.py)**: Optimised second iteration of the Alpha-Beta minimax agent.
*   **[bots/greedy.py](file:///home/user/dab_ai/bots/greedy.py)**: A simple greedy agent that completes any boxes when possible and otherwise avoids making moves that open boxes.
*   **[bots/greedy_improve.py](file:///home/user/dab_ai/bots/greedy_improve.py)**: An improved greedy bot with more robust move filtering and double-cross avoidance.
*   **[bots/mcts_heuristic.py](file:///home/user/dab_ai/bots/mcts_heuristic.py)**: Implements `MCTSHeuristicAgent`, combining neural-network-guided MCTS with hand-crafted rules (e.g., immediate box completion, chain-handling using Double-Cross strategy, and filtering unsafe moves).
*   **[bots/mcts_x.py](file:///home/user/dab_ai/bots/mcts_x.py)**: Exposes agents configured with custom simulation levels (e.g., `MCTS100Agent`, `MCTS1000Agent`) by reloading weight checkpoints.
*   **[download_bots.sh](file:///home/user/dab_ai/download_bots.sh)**: Script to download reference bots for evaluation.

---

