import random
from agent_interface import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, name: str = "Random"):
        super().__init__(name)

    def get_move(self, game_state) -> int:
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None
        return random.choice(valid_moves)

def create_agent(name: str, size: int, model_path: str = "best.pth.tar", n_simulations: int = 200):
    if name == "Random":
        return RandomAgent()
    elif name == "Greedy":
        from bots.greedy import GreedyPlayer
        return GreedyPlayer(name=name)
    elif name == "Greedy Chain":
        from bots.greedy_improve import GreedyChainPlayer
        return GreedyChainPlayer(name=name)
    elif name == "UCLABot":
        from bots.ucla_bot import UCLABot
        return UCLABot(name=name)
    elif name == "UCLABot_v2":
        from bots.ucla_bot import UCLABot_v2
        return UCLABot_v2(name=name)
    elif name == "UCLABot_v3":
        from bots.ucla_bot import UCLABot_v3
        return UCLABot_v3(name=name)
    elif name == "UCLABot_v4":
        from bots.ucla_bot_v4 import UCLABot_v4
        return UCLABot_v4(name=name)
    elif name == "UCLABot_v5":
        from bots.ucla_bot_v5 import UCLABot_v5
        return UCLABot_v5(name=name)
    elif name == "UCLABot_v6":
        from bots.ucla_bot_v6 import UCLABot_v6
        return UCLABot_v6(name=name)
    elif name == "UCLAGreedyBot":
        from bots.ucla_bot_heuristic import UCLAGreedyBot
        return UCLAGreedyBot(name=name)
    elif name == "UCLAAlphaBeta":
        from bots.ucla_alpha_beta import UCLAAlphaBeta
        return UCLAAlphaBeta(name=name)
    elif name == "UCLA_MCTS_0.1":
        from bots.ucla_mcts import UCLAMCTSBot
        return UCLAMCTSBot(name=name, time_limit=0.1)
    elif name == "UCLA_MCTS_0.2":
        from bots.ucla_mcts import UCLAMCTSBot
        return UCLAMCTSBot(name=name, time_limit=0.2)
    elif name == "UCLA_MCTS_0.5":
        from bots.ucla_mcts import UCLAMCTSBot
        return UCLAMCTSBot(name=name, time_limit=0.5)
    elif name == "SimpleBot":
        from bots.simple_bot import SimpleBot
        return SimpleBot(name=name)
    elif name == "SimpleBot_v2":
        from bots.simple_bot_v2 import SimpleBot_v2
        return SimpleBot_v2(name=name)
    elif name == "ARM_bot":
        from bots.arm_bot import ArmandoBot
        return ArmandoBot(name=name, ply=2)
    elif name == "fill_bot":
        from bots.fill_bot import FillBot
        return FillBot(name=name)
    elif name.startswith("AlphaZero"):
        from bots.alphazero_agent import AlphaZeroAgent
        if name == "AlphaZero":
            return AlphaZeroAgent(name=name, n_simulations=n_simulations, model_path=model_path)
        else:
            # Pattern: AlphaZero_<N>_D  (dynamic sims)
            #          AlphaZero_<N>_F  (fixed sims)
            #          AlphaZero_<N>    (legacy fixed, no suffix)
            import re
            m = re.fullmatch(r"AlphaZero_(\d+)(?:_(D|F))?", name)
            if m:
                n_sims = int(m.group(1))
                dynamic = (m.group(2) == "D")
                return AlphaZeroAgent(name=name, n_simulations=n_sims,
                                      model_path=model_path,
                                      dynamic_simulations=dynamic)
            return AlphaZeroAgent(name=name, model_path=model_path)
    elif name == "MCTS (100sims)":
        from bots.mcts_x import MCTSGAgent
        return MCTSGAgent(name=name, n_simulations=100)
    elif name == "MCTS (200sims)":
        from bots.mcts_x import MCTSGAgent
        return MCTSGAgent(name=name, n_simulations=200)
    elif name == "MCTS (300sims)":
        from bots.mcts_x import MCTSGAgent
        return MCTSGAgent(name=name, n_simulations=300)
    else:
        raise ValueError(f"Unknown agent name: {name}")
