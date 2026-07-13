from game import DotsAndBoxesGame
from model import NNetWrapper, dotdict
from coach import Coach

if __name__ == "__main__":
    args = dotdict({
        'lr': 0.001,
        'epochs': 10,
        'batch_size': 64,
        'num_channels': 256,
        'num_res_blocks': 10, 
        'l2_reg': 1e-4,
        'num_iters': 100,
        'num_eps': 50,
        'temp_threshold': 15,
        'maxlen_queue': 200000,
        'start_fill_pct': 0.8,
        'fill_decay': 0.05,
        'keep_history_iters': 20,
        'checkpoint_dir': './temp/',
        'n_simulations': 100,
        'c_puct': 1.0,
        'dirichlet_eps': 0.25,
        'dirichlet_alpha': 0.3,

        # Added parameters
        'arena_games': 40,
        'update_threshold': 0.55,
        'device': 'cuda' # or 'cpu'
})
    game = DotsAndBoxesGame(size=5)
    nnet = NNetWrapper(game, args)
    
    coach = Coach(game, nnet, args)
    coach.learn()