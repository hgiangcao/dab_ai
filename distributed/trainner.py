import os
import sys
import time
import torch
import numpy as np
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import config
import model_manager
import replay_manager
import evaluator
from game import DotsAndBoxesGame
from model import NNetWrapper, dotdict
from torch.utils.tensorboard import SummaryWriter

# Re-use args from config (augmented with model architecture defaults)
train_args = dotdict({
    'lr': config.LEARNING_RATE,
    'epochs': config.EPOCHS,
    'batch_size': config.BATCH_SIZE,
    'num_channels': 256,
    'num_res_blocks': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
})

def augment_data(train_examples):
    """
    Apply symmetry augmentation and format the data to the 4-channel representation.
    Extracts identical logic from coach.py AlphaZeroTrainer.augment_data
    """
    data_augmented = []
    # train_examples is a list of (lines, boxes, p, v)
    for lines, boxes, p, v in train_examples:
        for aug_lines, aug_boxes, aug_p, aug_v in zip(
            DotsAndBoxesGame.get_rotations_and_reflections_lines(lines),
            DotsAndBoxesGame.get_rotations_and_reflections_boxes(boxes),
            DotsAndBoxesGame.get_rotations_and_reflections_lines(np.asarray(p)),
            [v] * 8
        ):
            h, v_mat = DotsAndBoxesGame.l_to_h_v(aug_lines)
            size = DotsAndBoxesGame.n_lines_to_size(len(aug_lines))
            
            c1 = np.zeros((size+1, size+1))
            c1[:size+1, :size] = h
            
            c2 = np.zeros((size+1, size+1))
            c2[:size, :size+1] = v_mat
            
            c3 = np.zeros((size+1, size+1))
            c3[:size, :size] = (aug_boxes == 1).astype(float)
            
            c4 = np.zeros((size+1, size+1))
            c4[:size, :size] = (aug_boxes == -1).astype(float)
            
            board_state = np.stack([c1, c2, c3, c4])
            data_augmented.append((board_state, aug_p, aug_v))
            
    return data_augmented

def train_network(replay_data, output_model_path):
    """
    Train AlphaZero network.
    
    Input:
        replay_file: merged self-play games (loaded as tuples)
        output_model: path to save trained model
    """
    game_size = 5
    dummy_game = DotsAndBoxesGame(size=game_size)
    nnet = NNetWrapper(dummy_game, train_args)
    
    # Load best model if exists to continue training
    best_model_path = model_manager.get_best_model_path()
    state_dict = model_manager.load_model(best_model_path)
    if state_dict:
        nnet.nnet.load_state_dict(state_dict)
        print(f"Loaded existing best model weights from {best_model_path}.")
    else:
        print("No previous best model found. Initializing randomly.")
        
    print(f"Augmenting dataset of {len(replay_data)} raw states...")
    augmented_memory = augment_data(replay_data)
    
    print(f"Training on {len(augmented_memory)} samples...")
    pi_loss, v_loss, total_loss = nnet.train(augmented_memory)
    
    print(f"Training complete. Loss -> Policy: {pi_loss:.4f} | Value: {v_loss:.4f} | Total: {total_loss:.4f}")
    
    # Save the trained candidate model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    model_manager.save_latest_model(nnet)
    print(f"Candidate model saved to {output_model_path}")
    
    return {"pi_loss": pi_loss, "v_loss": v_loss, "total_loss": total_loss}

def load_training_checkpoint():
    """Resume interrupted training. (Currently handled dynamically in train_network)"""
    pass

def save_training_checkpoint():
    """Save training states. (Currently handled by model_manager.save_latest_model)"""
    pass

def run_training_iteration(writer=None, iteration=0):
    """
    Execute one complete iteration:
    1. Validate and promote incoming → ready
    2. Claim ready → training (anti-race)
    3. Load replay buffer from training/
    4. Train candidate network
    5. Evaluate candidate vs best
    6. Mark training files as used
    """
    print("\n" + "="*60)
    print("STARTING NEW TRAINING ITERATION")
    print("="*60)
    
    # 1. Validate and promote incoming files to ready
    promoted = replay_manager.validate_and_promote()
    print(f"Promoted {len(promoted)} new files to ready/")
    
    # 2. Claim files atomically to prevent race conditions
    claimed_files = replay_manager.claim_for_training()
    if not claimed_files:
        print("No files in ready/ to claim. Waiting for workers...")
        return False
        
    # 3. Load the replay buffer from training/
    replay_data = replay_manager.load_replay_buffer(claimed_files)
    if len(replay_data) < config.MIN_REPLAY_SIZE:
        print(f"Replay buffer size ({len(replay_data)}) below minimum ({config.MIN_REPLAY_SIZE}). Waiting...")
        # Move claimed files back to ready/ so they aren't lost
        for f in claimed_files:
            fname = os.path.basename(f)
            import shutil
            shutil.move(f, os.path.join(config.REPLAY_READY, fname))
        return False
        
    # 4. Train candidate network
    merged_path = replay_manager.merge_replay(claimed_files)
    candidate_path = os.path.join(config.get_current_model_dir(), "checkpoint_candidate.pth.tar")
    losses = train_network(replay_data, candidate_path)
    
    if writer:
        writer.add_scalar('Log/Policy_Loss', losses["pi_loss"], iteration)
        writer.add_scalar('Log/Value_Loss', losses["v_loss"], iteration)
        writer.add_scalar('Log/Total_Loss', losses["total_loss"], iteration)
        writer.add_scalar('Log/Memory_Size', len(replay_data), iteration)
    
    # 5. Evaluate and update
    print("\nEvaluating candidate model against best model...")
    promoted_model, win_rate = evaluator.evaluate_new_model()
    
    if writer:
        writer.add_scalar('Evaluation/Win_Rate_Vs_Old', win_rate, iteration)
        writer.flush()
    
    if promoted_model:
        print("Model promoted! Cleaning up old data...")
        
    # 6. Mark files as used
    replay_manager.mark_used(claimed_files)
    
    # 7. Maintain disk space
    replay_manager.cleanup_old(days=7)
    
    return True

def training_loop():
    """
    Main infinite training loop.
    Monitors replay files and runs iterations continuously.
    """
    print("Distributed Trainer Daemon Started. Monitoring storage/replay...")
    
    # Initialize TensorBoard writer
    log_dir = config.get_current_model_dir()
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    iteration = 1
    
    # Backup source code for reproducibility
    import glob
    import shutil
    code_dir = os.path.join(log_dir, 'code')
    os.makedirs(code_dir, exist_ok=True)
    
    for f in glob.glob(os.path.join(PROJECT_ROOT, '*.py')):
        shutil.copy(f, code_dir)
        
    dist_dir = os.path.join(code_dir, 'distributed')
    os.makedirs(dist_dir, exist_ok=True)
    for f in glob.glob(os.path.join(PROJECT_ROOT, 'distributed', '*.py')):
        shutil.copy(f, dist_dir)
        
    bots_src = os.path.join(PROJECT_ROOT, 'bots')
    if os.path.exists(bots_src):
        shutil.copytree(bots_src, os.path.join(code_dir, 'bots'), dirs_exist_ok=True)
    
    while True:
        try:
            # Check how many replay files are in incoming/ and ready/
            import glob
            incoming = glob.glob(os.path.join(config.REPLAY_INCOMING, "*.npz"))
            ready = glob.glob(os.path.join(config.REPLAY_READY, "*.npz"))
            total_files = len(incoming) + len(ready)
            
            # Start iteration if we have multiple new files to merge
            if total_files >= 5: 
                success = run_training_iteration(writer, iteration)
                if not success:
                    print("Iteration skipped. Sleeping 60s...")
                    time.sleep(60)
                else:
                    iteration += 1
            else:
                print(f"Not enough replay files found (incoming+ready = {total_files}). Sleeping 60s...")
                time.sleep(60)
                
        except Exception as e:
            print(f"Error during training iteration: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(30)

if __name__ == "__main__":
    training_loop()