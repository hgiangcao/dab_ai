import os
from bots.mcts_heuristic import MCTSHeuristicAgent


# 1. Initialize the shared state at the module level
_shared_mcts_model = None

class MCTSGAgent(MCTSHeuristicAgent):
    """
    Generalized MCTS Heuristic Agent that accepts a custom number of simulations.
    """
    def __init__(self, name: str = "MCTS_X", n_simulations: int = 100):
        global _shared_mcts_model
        
        if _shared_mcts_model is None:
            from model import NNetWrapper, dotdict
            from game import DotsAndBoxesGame
            
            args = dotdict({
                'lr': 0.001,
                'epochs': 10,
                'batch_size': 64,
                'num_channels': 256,
                'num_res_blocks': 10, 
                'l2_reg': 1e-4,
            })
            game = DotsAndBoxesGame(size=5)
            _shared_mcts_model = NNetWrapper(game, args)
            
            checkpoint_dir = './temp/'
            
            # Helper to safely attempt loading a file path
            def try_load(filename):
                path = os.path.join(checkpoint_dir, filename)
                if os.path.exists(path):
                    try:
                        _shared_mcts_model.load_checkpoint(folder=checkpoint_dir, filename=filename)
                        # Put network in evaluation mode
                        if hasattr(_shared_mcts_model, 'nnet'):
                            _shared_mcts_model.nnet.eval()
                        print(f"-> Successfully loaded weights from {path}")
                        return True
                    except Exception as e:
                        print(f"Could not load checkpoint {filename}: {e}")
                return False

            # Attempt loading priority chain
            if not try_load('best.pth.tar'):
                try_load('checkpoint.pth.tar')

        mcts_parameters = {
            "n_simulations": n_simulations,
            "c_puct": 1.0,
            "dirichlet_eps": 0.25,
            "dirichlet_alpha": 0.3
        }
        
        super().__init__(name, _shared_mcts_model, mcts_parameters)


class MCTS100Agent(MCTSGAgent):
    def __init__(self, name: str = "MCTS_100"):
        super().__init__(name, n_simulations=100)


class MCTS1000Agent(MCTSGAgent):
    def __init__(self, name: str = "MCTS_1000"):
        super().__init__(name, n_simulations=1000)
