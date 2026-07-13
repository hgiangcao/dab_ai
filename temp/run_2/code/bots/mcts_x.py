from bots.mcts_heuristic import MCTSHeuristicAgent


class MCTSGAgent(MCTSHeuristicAgent):
    """
    Generalized MCTS Heuristic Agent that accepts a custom number of simulations.
    Fully heuristic-based with Monte Carlo rollouts (no neural networks).
    """
    def __init__(self, name: str = "MCTS_X", n_simulations: int = 100, size: int = None, time_limit: float = None):
        mcts_parameters = {
            "n_simulations": n_simulations,
            "time_limit": time_limit,
            "c_puct": 1.0  # Acts as exploration constant (c) in pure UCT
        }
        super().__init__(name, mcts_parameters)

    def get_move(self, game_state) -> int:
        return super().get_move(game_state)


class MCTS100Agent(MCTSGAgent):
    def __init__(self, name: str = "MCTS_100", size: int = None):
        super().__init__(name, n_simulations=100, size=size)


class MCTS1000Agent(MCTSGAgent):
    def __init__(self, name: str = "MCTS_1000", size: int = None):
        # Pass a time limit of 2.0 seconds instead of fixed simulations
        super().__init__(name, n_simulations=1000, size=size, time_limit=2.0)
