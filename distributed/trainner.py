import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
from pretrained import run_pretraining

# Re-use args from config (augmented with model architecture defaults)
train_args = dotdict({
    'lr': config.LEARNING_RATE,
    'epochs': config.EPOCHS,
    'batch_size': config.BATCH_SIZE,
    'num_channels': 256,
    'num_res_blocks': 10,
    'l2_reg': 1e-4,
    # 7 iter/hour × 48 hours = 336 total iterations → LR reaches eta_min exactly at stop time
    'lr_scheduler_steps': 336,
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

def train_network(replay_data, output_model_path, nnet, epochs=None):
    """
    Train AlphaZero network.
    
    Input:
        replay_data: merged self-play games (loaded as tuples)
        output_model_path: path to save trained model
        nnet: the single persistent instance of the Neural Network
        epochs: number of epochs to train
    """
        
    print(f"Augmenting dataset of {len(replay_data)} raw states...")
    augmented_memory = augment_data(replay_data)
    
    print(f"Training on {len(augmented_memory)} samples...")
    pi_loss, v_loss, total_loss = nnet.train(augmented_memory, epochs=epochs)
    
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

def run_training_iteration(writer=None, iteration=0, nnet=None, replay_buffer=None):
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
    print("STARTING NEW TRAINING ITERATION", iteration)
    print("="*60)
    
    # 1. Validate and promote incoming files to ready
    promoted = replay_manager.validate_and_promote()
    print(f"Promoted {len(promoted)} new files to ready/")
    
    # 2. Claim files atomically to prevent race conditions
    claimed_files = replay_manager.claim_for_training()
    if not claimed_files:
        print("No files in ready/ to claim. Trainer acting as fallback worker to generate 50 games...")
        from selfplay import SelfPlayGenerator
        import shutil
        
        generator = SelfPlayGenerator()
        candidate_path = os.path.join(config.get_current_model_dir(), "checkpoint_candidate.pth.tar")
        if not os.path.exists(candidate_path):
            candidate_path = model_manager.get_best_model_path()
            
        if os.path.exists(candidate_path):
            generator.load_model(candidate_path)
            
        replay_file = generator.play_games(
            num_games=50,
            save_dir=config.REPLAY_READY,
            worker_id="trainer_fallback",
            model_version=iteration,
            current_phase=model_manager.get_current_phase(),
            epoch=iteration
        )
        if replay_file and os.path.exists(replay_file):
            print(f"Trainer fallback generated {replay_file}.")
            fname = os.path.basename(replay_file)
            dest = os.path.join(config.REPLAY_TRAINING, fname)
            shutil.move(replay_file, dest)
            claimed_files.append(dest)
        else:
            print("Trainer fallback failed to generate games.")
            return False
        
    # 2.5 Check Curriculum Phase Winrates
    client_winrates = []
    current_server_phase = model_manager.get_current_phase()
    
    import json
    aggregated_bot_stats = {}
    
    for f in claimed_files:
        try:
            data = np.load(f, allow_pickle=False)
            if 'phase_winrate' in data and 'current_phase' in data:
                file_phase = data['current_phase'][0]
                # Only use winrates generated by clients running the CURRENT phase
                if file_phase == current_server_phase:
                    client_winrates.append(data['phase_winrate'][0])
                    
            if 'bot_stats' in data:
                bot_stats_str = str(data['bot_stats'][0])
                file_bot_stats = json.loads(bot_stats_str)
                for opp, stats in file_bot_stats.items():
                    if opp not in aggregated_bot_stats:
                        aggregated_bot_stats[opp] = {'games': 0, 'wins': 0, 'decisive': 0}
                    aggregated_bot_stats[opp]['games'] += stats['games']
                    aggregated_bot_stats[opp]['wins'] += stats['wins']
                    aggregated_bot_stats[opp]['decisive'] += stats['decisive']
                    
        except Exception as e:
            print(f"Error parsing {f}: {e}")
            pass
            
    if client_winrates:
        max_client_winrate = max(client_winrates)
        current_phase = model_manager.get_current_phase()
        print ("max_client_winrate ,current_phase",max_client_winrate ,current_phase)
        if writer:
            writer.add_scalar('Curriculum/Phase_Winrate', max_client_winrate, iteration)
            writer.add_scalar('Training_Winrate', max_client_winrate, iteration)
            writer.add_scalar('Curriculum/Current_Phase', current_phase, iteration)
            
            # Log Bot Stats
            total_games_all_bots = sum(s['games'] for s in aggregated_bot_stats.values())
            for opp, stats in aggregated_bot_stats.items():
                if total_games_all_bots > 0:
                    pct = stats['games'] / total_games_all_bots
                    writer.add_scalar(f'Curriculum_Bots/{opp}_Percentage', pct, iteration)
                if stats['decisive'] > 0:
                    wr = stats['wins'] / stats['decisive']
                    writer.add_scalar(f'Curriculum_Bots/{opp}_WinRate', wr, iteration)
            
        # Advance phase only when the model is genuinely winning, 
        # with a backstop of 20 iterations so training never stalls forever.
        phase_iterations = iteration % 20  # rough estimate of iters spent in current phase
        if (max_client_winrate >= 0.60 or phase_iterations == 0) and current_phase < 5:
            reason = f"Winrate {max_client_winrate:.1%} >= 60%" if max_client_winrate >= 0.60 else "Max phase iterations reached"
            print(f"\n===========================================================")
            print(f"Phase {current_phase} cleared ({reason})!")
            print(f"Advancing to Phase {current_phase + 1}...")
            print(f"===========================================================\n")
            model_manager.advance_curriculum_phase()
        
    # 3. Load the replay buffer from training/
    new_data = replay_manager.load_replay_buffer(claimed_files)
    
    is_first_time = (replay_buffer is None) or (len(replay_buffer) == 0)
    
    if is_first_time and len(new_data) < config.MIN_REPLAY_SIZE:
        print(f"New data size ({len(new_data)}) below minimum ({config.MIN_REPLAY_SIZE}). Waiting...")
        # Move claimed files back to ready/ so they aren't lost
        for f in claimed_files:
            fname = os.path.basename(f)
            import shutil
            shutil.move(f, os.path.join(config.REPLAY_READY, fname))
        return False
    
    # Accumulate into rolling replay buffer (experience replay window)
    if replay_buffer is not None:
        replay_buffer.extend(new_data)
        replay_data = list(replay_buffer)
        print(f"Replay buffer: {len(new_data)} new + {len(replay_data) - len(new_data)} retained = {len(replay_data)} total samples")
    else:
        replay_data = new_data
        
    # 4. Train candidate network
    merged_path = replay_manager.merge_replay(claimed_files)
    candidate_path = os.path.join(config.get_current_model_dir(), "checkpoint_candidate.pth.tar")
    dynamic_epochs = 2 * len(claimed_files)
    losses = train_network(replay_data, candidate_path, nnet, epochs=dynamic_epochs)
    
    if writer:
        current_lr = nnet.optimizer.param_groups[0]['lr']
        writer.add_scalar('Log/Policy_Loss', losses["pi_loss"], iteration)
        writer.add_scalar('Log/Value_Loss', losses["v_loss"], iteration)
        writer.add_scalar('Log/Total_Loss', losses["total_loss"], iteration)
        writer.add_scalar('Log/Memory_Size', len(replay_data), iteration)
        writer.add_scalar('Log/Learning_Rate', current_lr, iteration)
    
    # 5. Evaluate and update
    print("\nEvaluating candidate model against best model and baselines...")
    promoted_model, win_rate, baseline_win_rates, avg_depth = evaluator.evaluate_new_model(iteration)
    
    if writer:
        writer.add_scalar('Evaluation/Win_Rate_Vs_Old', win_rate, iteration)
        writer.add_scalar('Evaluation/MCTS_Avg_Depth', avg_depth, iteration)
        for opp_name, rate in baseline_win_rates.items():
            writer.add_scalar(f'Evaluation/Win_Rate_Vs_{opp_name}', rate, iteration)
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
    
    # Initialize the neural network ONCE for continuous use
    print("Initializing Neural Network for continuous training...")
    dummy_game = DotsAndBoxesGame(size=5)
    global_nnet = NNetWrapper(dummy_game, train_args)

    # ── Supervised pretraining from bot game logs (runs once, skipped on restart) ──
    print("\n" + "=" * 60)
    print("PHASE 0: SUPERVISED PRETRAINING FROM BOT GAME LOGS")
    print("=" * 60)
    did_pretrain = run_pretraining(global_nnet, log_dir, writer)
    if did_pretrain:
        print("[Pretrain] Pretraining complete. Proceeding to AlphaZero self-play.\n")
    else:
        print("[Pretrain] Pretraining skipped. Using existing weights.\n")
        
    # Signal to workers that pretraining is done
    model_manager.set_pretrain_finished()

    
    # Try to load full checkpoint (weights + optimizer + scheduler state)
    candidate_path = os.path.join(config.get_current_model_dir(), "checkpoint_candidate.pth.tar")
    if os.path.exists(candidate_path):
        load_path = candidate_path
        print("Load pretrained model checkpoint_candidate")
    else:
        load_path = model_manager.get_latest_model_path()
        
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)
        weights = checkpoint.get('state_dict', checkpoint)
        global_nnet.nnet.load_state_dict(weights)
        if 'optimizer' in checkpoint:
            global_nnet.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Restored optimizer state from {load_path}.")
        if 'scheduler' in checkpoint:
            global_nnet.scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"Restored scheduler state from {load_path}.")
        else:
            # No saved scheduler state (pre-scheduler checkpoint).
            # Fast-forward the scheduler to match iterations already completed
            # so LR resumes from the correct position instead of restarting at peak.
            elapsed = model_manager.get_current_version()
            for _ in range(elapsed):
                global_nnet.scheduler.step()
            resumed_lr = global_nnet.optimizer.param_groups[0]['lr']
            print(f"No scheduler state found. Fast-forwarded {elapsed} steps. Resumed LR: {resumed_lr:.2e}")
        print(f"Loaded model weights from {load_path}.")
    else:
        print("No previous model found. Initializing randomly.")
        
    

    # Initialize rolling experience replay buffer capped at MAX_REPLAY_SIZE
    print(f"Initializing experience replay buffer (max size: {config.MAX_REPLAY_SIZE:,})...")
    replay_buffer = deque(maxlen=config.MAX_REPLAY_SIZE)
    
    print("PHASE 1: ALPHA ZERO TRAINING")

    while True:
        try:
            # Check how many replay files are in incoming/ and ready/
            success = run_training_iteration(writer, iteration, global_nnet, replay_buffer)
            if not success:
                print("Iteration skipped. Sleeping 60s...")
                time.sleep(60)
            else:
                iteration += 1
                
        except Exception as e:
            print(f"Error during training iteration: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(30)

if __name__ == "__main__":
    training_loop()