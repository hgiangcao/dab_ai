import os
import shutil
import glob
from game import DotsAndBoxesGame
from model import NNetWrapper, dotdict
from coach import AlphaZeroTrainer
from mcts import MCTS

def get_next_run_dir(base_dir='./logs'):
    os.makedirs(base_dir, exist_ok=True)
    run_idx = 1
    while os.path.exists(os.path.join(base_dir, f'run_{run_idx}')):
        run_idx += 1
    return os.path.join(base_dir, f'run_{run_idx}')

if __name__ == "__main__":
    run_dir = get_next_run_dir()
    
    args = dotdict({
        'lr': 0.0005,
        'epochs': 10,
        'batch_size': 64,
        'num_channels': 256,
        'num_res_blocks': 10, 
        'l2_reg': 1e-4,
        'num_iters': 1200,
        'num_eps': 100,
        'temp_threshold': 15,
        'maxlen_queue': 200000,
        'start_fill_pct': 0.8,
        'fill_decay': 0.05,
        'keep_history_iters': 20,
        'checkpoint_dir': run_dir,
        'n_simulations': 100,
        'c_puct': 1.0,
        'dirichlet_eps': 0.25,
        'dirichlet_alpha': 0.3,

        # Added parameters
        'arena_games': 40,
        'update_threshold': 0.55,
        'device': 'cuda' # or 'cpu'
    })
    
    # Initialize the base game size (the game instances will be created per match)
    game_size = 5
    
    # Needs a dummy game object just to query dimensions for Neural Net init
    dummy_game = DotsAndBoxesGame(size=game_size)
    
    # Initialize active neural network
    nnet = NNetWrapper(dummy_game, args)
    
    # Initialize previous neural network for arena comparison
    pnet = NNetWrapper(dummy_game, args)
    pnet.nnet.load_state_dict(nnet.nnet.state_dict())
    
    # Ensure checkpoint directory exists and copy source code
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    code_dir = os.path.join(args.checkpoint_dir, 'code')
    os.makedirs(code_dir, exist_ok=True)
    for f in glob.glob('*.py'):
        shutil.copy(f, code_dir)
    if os.path.exists('bots'):
        shutil.copytree('bots', os.path.join(code_dir, 'bots'), dirs_exist_ok=True)
    
    # Initialize the AlphaZero Trainer
    coach = AlphaZeroTrainer(game_size, nnet, pnet, MCTS, args)
    
    print("Starting AlphaZero Training Loop...")
    coach.learn()