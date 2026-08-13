import os
import numpy as np
from agent_interface import BaseAgent
from game import DotsAndBoxesGame

_nnet_cache = {}

class AlphaZeroAgent(BaseAgent):
    def __init__(self, name: str = "AlphaZero", n_simulations=200, model_path="best.pth.tar",
                 dynamic_simulations: bool = False):
        super().__init__(name)
        import torch
        import config
        from model import NNetWrapper, dotdict
        from mcts import MCTS
        
        self.eval_args = dotdict({
            'lr': config.LEARNING_RATE,
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'num_channels': 256,
            'num_res_blocks': 10, 
            'l2_reg': 1e-4,
            'n_simulations': n_simulations,
            'c_puct': config.MCTS_C_PUCT,
            'dirichlet_eps': 0.0,
            'dirichlet_alpha': config.MCTS_DIRICHLET_ALPHA,
            'device': 'cpu',
            'dynamic_simulations': dynamic_simulations,
        })
        
        global _nnet_cache
        if model_path not in _nnet_cache:
            self.game_ref = DotsAndBoxesGame(size=5)
            net = NNetWrapper(self.game_ref, self.eval_args)
            if os.path.exists(model_path):
                try:
                    state = torch.load(model_path, map_location='cpu', weights_only=False)
                    net.nnet.load_state_dict(state['state_dict'] if 'state_dict' in state else state)
                    print(f"Loaded AlphaZero model from {model_path}")
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
            else:
                print(f"Warning: AlphaZero model not found at {model_path}")
            net.nnet.eval()
            _nnet_cache[model_path] = net
            
        self.net = _nnet_cache[model_path]
        self.mcts = MCTS(self.net, self.eval_args)
        self._last_action: int = None

    def reset(self):
        """Reset MCTS tree. Must be called at the start of each new game."""
        self.mcts.reset_tree()
        self._last_action = None

    def get_move(self, game_state: DotsAndBoxesGame) -> int:
        pi = self.mcts.play(game_state, temp=0, last_action=self._last_action)
        action = int(np.argmax(pi))
        self._last_action = action
        return action
