import os
from game import DotsAndBoxesGame
from model import NNetWrapper, dotdict
from coach import AlphaZeroTrainer
from mcts import MCTS

if __name__ == "__main__":
    args = dotdict({
        'lr': 0.001,
        'epochs': 1,
        'batch_size': 8,
        'num_channels': 32,
        'num_res_blocks': 2, 
        'l2_reg': 1e-4,
        'num_iters': 1,
        'num_eps': 1,
        'temp_threshold': 15,
        'maxlen_queue': 200000,
        'start_fill_pct': 0.8,
        'fill_decay': 0.05,
        'keep_history_iters': 20,
        'checkpoint_dir': './temp/',
        'n_simulations': 10,
        'c_puct': 1.0,
        'dirichlet_eps': 0.25,
        'dirichlet_alpha': 0.3,

        # Added parameters
        'arena_games': 2,
        'update_threshold': 0.55,
        'device': 'cpu' # force CPU for quick test
    })
    
    game_size = 2 # small board for quick game
    dummy_game = DotsAndBoxesGame(size=game_size)
    nnet = NNetWrapper(dummy_game, args)
    pnet = NNetWrapper(dummy_game, args)
    pnet.nnet.load_state_dict(nnet.nnet.state_dict())
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    coach = AlphaZeroTrainer(game_size, nnet, pnet, MCTS, args)
    
    print("Starting MINI AlphaZero Training Loop...")
    coach.learn()
