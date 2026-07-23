import os
import uuid
import sys
import json
import random
import multiprocessing
import concurrent.futures
import numpy as np
from tqdm import tqdm

# Add parent directory to path so we can import from the main project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import config
from model import dotdict
from mcts import MCTS
from coach import build_worker_chunks, worker_execute_episode_chunk, get_selfplay_executor

# Imported from config: config.PHASES_CONFIG

class SelfPlayGenerator:

    def __init__(self):
        self.game_size = 5
        self.args = dotdict({
            'lr': config.LEARNING_RATE,
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'num_channels': 256,
            'num_res_blocks': 10, 
            'l2_reg': 1e-4,
            'num_iters': 1200,
            'num_eps': 100,
            'temp_threshold': 40,
            'temperature_initial': 1.0,
            'temperature_medium': 0.5,
            'temperature_final': 0.0,
            'temperature_drop_move': 40,
            'temperature_medium_end_move': 55,
            'maxlen_queue': config.MAX_REPLAY_SIZE,
            'start_fill_pct': 0.8,
            'fill_decay': 0.05,
            'keep_history_iters': 20,
            'checkpoint_dir': './logs',
            'n_simulations': config.MCTS_NUM_SIMULATIONS,
            'c_puct': config.MCTS_C_PUCT,
            'dirichlet_eps': config.MCTS_DIRICHLET_EPS,
            'dirichlet_alpha': config.MCTS_DIRICHLET_ALPHA,
            'arena_games': config.EVAL_GAMES,
            'update_threshold': config.PROMOTION_THRESHOLD,
            'device': 'cpu'  # Workers use CPU for highly parallel self-play
        })
        self.latest_model_path = None
        
        # Load Reverse Curriculum Logs
        self.json_logs = []
        log_path = os.path.join(PROJECT_ROOT, "game_logs.jsonl")
        expected_moves = 2 * self.game_size * (self.game_size + 1)
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            moves = record.get("moves", [])
                            if moves and len(moves) == expected_moves:
                                self.json_logs.append(moves)
                        except Exception:
                            pass
        print(f"SelfPlayGenerator initialized. Loaded {len(self.json_logs)} historical sequences.")
 
    def load_model(self, checkpoint):
        """Loads the path to the latest model to be used by the multiprocessing workers."""
        self.latest_model_path = os.path.abspath(checkpoint)
 
    def play_games(self, num_games, save_dir, worker_id="worker", model_version=0, current_phase=0,epoch =0):
        print(f"epoch {epoch} - Starting generation of {num_games} games at Phase {current_phase}...")
 
        # Determine opponent pool for the current phase
        current_pool = list(config.PHASES_CONFIG[current_phase])
            
        total_prob = sum(p for _, p in current_pool)
        normalized_probs = [p / total_prob for _, p in current_pool]
        
        # Reverse Curriculum Fill % approximation based on model version (iterations)
        start_fill_pct = max(0.0, 0.70 - (0.70 / 10) * epoch)
        
        if self.json_logs and start_fill_pct >= 0.001:
            sampled_sequences = [random.choice(self.json_logs) for _ in range(num_games)]
        else:
            sampled_sequences = [None] * num_games
        
        print(start_fill_pct,"RANDOM FILLED MOVES")
        
        # 3. Setup episode specs for chunked multiprocessing
        episode_specs = []
        for seq in sampled_sequences:
            opp_type = np.random.choice([name for name, _ in current_pool], p=normalized_probs)
            # Workers don't necessarily have server's "best" or "past" checkpoints readily available.
            # Revert them to "self" to guarantee execution.
            if opp_type in ["best", "past"]:
                opp_type = "self"
                
            episode_specs.append((
                seq,
                start_fill_pct,
                opp_type,
                None # opp_path is None because we don't have past checkpoints locally
            ))
            
        # 4. Generate games using multiprocessing pool
        iteration_data = []
        phase_wins = 0
        phase_decisive = 0
        bot_stats = {}
        mp_context = multiprocessing.get_context('spawn')
        
        max_workers = max(1, min(config.MAX_WORKERS, multiprocessing.cpu_count() - 1))
        worker_args_list = build_worker_chunks(
            self.game_size,
            self.latest_model_path,
            MCTS,
            self.args,
            episode_specs,
            max_workers=max_workers,
        )
        if not worker_args_list:
            print("No games requested; no replay file generated.")
            return None

        print(f"Using {len(worker_args_list)} self-play worker processes for {len(episode_specs)} games.")
        executor = get_selfplay_executor(
            self.latest_model_path,
            MCTS,
            self.args,
            self.game_size,
            max_workers=max_workers,
            mp_context=mp_context,
        )
        futures = {executor.submit(worker_execute_episode_chunk, arg): len(arg[4]) for arg in worker_args_list}
        with tqdm(total=len(episode_specs), desc=f"Self-Play (Phase {current_phase})") as pbar:
            for future in concurrent.futures.as_completed(futures):
                chunk_size = futures[future]
                try:
                    chunk_results = future.result()
                    for examples, length, depth, opp_type, latest_won, latest_drawn in chunk_results:
                        iteration_data.extend(examples)
                        if opp_type != "self":
                            if not latest_drawn:
                                phase_decisive += 1
                                if latest_won:
                                    phase_wins += 1
                                
                        if opp_type not in bot_stats:
                            bot_stats[opp_type] = {'games': 0, 'wins': 0, 'decisive': 0}
                        bot_stats[opp_type]['games'] += 1
                        if not latest_drawn:
                            bot_stats[opp_type]['decisive'] += 1
                            if latest_won:
                                bot_stats[opp_type]['wins'] += 1
                                
                except Exception as e:
                    import traceback
                    print(f"Worker execution failed: {e}")
                    traceback.print_exc()
                finally:
                    pbar.update(chunk_size)

        phase_winrate = phase_wins / phase_decisive if phase_decisive > 0 else 0.5

        # 5. Save generated examples to local storage (.npz)
        if not iteration_data:
            print("No valid training examples were generated.")
            return None
            
        lines_data = []
        boxes_data = []
        pis_data = []
        vals_data = []
        
        for lines, boxes, pi, val in iteration_data:
            lines_data.append(lines)
            boxes_data.append(boxes)
            pis_data.append(pi)
            vals_data.append(val)
            
        lines_data = np.array(lines_data, dtype=np.float32)
        boxes_data = np.array(boxes_data, dtype=np.float32)
        pis_data = np.array(pis_data, dtype=np.float32)
        vals_data = np.array(vals_data, dtype=np.float32)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Structured filename: {worker_id}_v{model_version}_{timestamp}.npz
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"{worker_id}_v{model_version}_{timestamp}.npz")
        

        print ("Phase:", current_phase, "winrate", phase_winrate)
        
        import json
        bot_stats_json = json.dumps(bot_stats)

        np.savez_compressed(
            filename,
            lines=lines_data,
            boxes=boxes_data,
            pis=pis_data,
            vals=vals_data,
            # Embedded metadata
            worker_id=np.array(worker_id),
            model_version=np.array(model_version, dtype=np.int32),
            game_count=np.array(len(lines_data), dtype=np.int32),
            phase_winrate=np.array([phase_winrate], dtype=np.float32),
            current_phase=np.array([current_phase], dtype=np.int32),
            bot_stats=np.array([bot_stats_json], dtype=str)
        )
        
        print(f"Saved {len(iteration_data)} examples to {filename}")
        return filename

if __name__ == "__main__":
    import argparse
    import torch
    import model_manager
    
    parser = argparse.ArgumentParser(description="Test SelfPlayGenerator locally")
    parser.add_argument("--phase", type=int, default=6, help="Phase index to test")
    parser.add_argument("--games", type=int, default=2, help="Number of games to generate")
    args = parser.parse_args()
    
    print(f"Testing SelfPlayGenerator locally (Phase {args.phase}, Games {args.games})...")
    
    generator = SelfPlayGenerator()
    
    # Try to find an existing model
    latest_path = model_manager.get_latest_model_path()
    best_path = model_manager.get_best_model_path()
    
    test_model_path = None
    if os.path.exists(latest_path):
        test_model_path = latest_path
    elif os.path.exists(best_path):
        test_model_path = best_path
        
    print(f"Loading model from: {test_model_path}")
    if test_model_path:
        generator.load_model(test_model_path)
    
    # Test generation of games
    test_save_dir = os.path.join(PROJECT_ROOT, "storage", "test_replay")
    print(f"Running {args.games} self-play games to test worker pipeline...")
    saved_file = generator.play_games(
        num_games=args.games, 
        save_dir=test_save_dir,
        current_phase=args.phase
    )
    
    if saved_file and os.path.exists(saved_file):
        print(f"\\nSUCCESS! Test replay saved to {saved_file}")
        
        # Verify the saved data integrity
        data = np.load(saved_file)
        print("Saved Replay Data Shapes:")
        print(f"- Lines:  {data['lines'].shape}")
        print(f"- Boxes:  {data['boxes'].shape}")
        print(f"- Pis:    {data['pis'].shape}")
        print(f"- Vals:   {data['vals'].shape}")
    else:
        print("\\nFAILURE! Failed to generate games.")