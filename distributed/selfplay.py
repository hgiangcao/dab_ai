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

from model import dotdict
from mcts import MCTS
from coach import worker_execute_episode

class SelfPlayGenerator:

    def __init__(self):
        self.game_size = 5
        self.args = dotdict({
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
            'checkpoint_dir': './logs',
            'n_simulations': 100,
            'c_puct': 1.0,
            'dirichlet_eps': 0.25,
            'dirichlet_alpha': 0.3,
            'arena_games': 40,
            'update_threshold': 0.55,
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

    def play_games(self, num_games, save_dir, worker_id="worker", model_version=0):
        # 1. Get current phase from version.txt
        version_txt_path = os.path.join(PROJECT_ROOT, "version.txt")
        current_phase = 0
        if os.path.exists(version_txt_path):
            try:
                with open(version_txt_path, "r") as f:
                    for line in f:
                        if line.startswith("current_phase:"):
                            current_phase = int(line.split(":")[1].strip())
            except Exception as e:
                print(f"Error reading version.txt: {e}")
        
        print(f"Starting generation of {num_games} games at Phase {current_phase}...")

        # 2. Determine opponent pool based on phase (matching coach.py configuration)
        phases_config = [
            [("random", 0.01)],
            [("greedy", 0.1)],
            [("greedy_chain", 0.1)],
            [("alpha_beta_0.1s", 0.1)],
            [("mcts_0.1s", 0.1)],
            [("self", 0.4), ("best", 0.1), ("past", 0.1)]
        ]
        
        current_pool = []
        for phase_idx in range(min(current_phase + 1, len(phases_config))):
            current_pool.extend(phases_config[phase_idx])
            
        total_prob = sum(p for _, p in current_pool)
        normalized_probs = [p / total_prob for _, p in current_pool]
        
        # Reverse Curriculum Fill % approximation based on phase
        start_fill_pct = max(0.0, 0.70 - (0.70 / 10) * current_phase)
        
        if self.json_logs and start_fill_pct >= 0.001:
            sampled_sequences = [random.choice(self.json_logs) for _ in range(num_games)]
        else:
            sampled_sequences = [None] * num_games
        print(start_fill_pct,"RANDOM FILLED MOVES")
        # 3. Setup arguments for multiprocessing
        worker_args_list = []
        for seq in sampled_sequences:
            opp_type = np.random.choice([name for name, _ in current_pool], p=normalized_probs)
            # Workers don't necessarily have server's "best" or "past" checkpoints readily available.
            # Revert them to "self" to guarantee execution.
            if opp_type in ["best", "past"]:
                opp_type = "self"
                
            worker_args_list.append((
                self.game_size, 
                self.latest_model_path, 
                MCTS, 
                self.args, 
                seq, 
                start_fill_pct, 
                opp_type, 
                None # opp_path is None because we don't have past checkpoints locally
            ))
            
        # 4. Generate games using multiprocessing pool
        iteration_data = []
        phase_wins = 0
        phase_decisive = 0
        mp_context = multiprocessing.get_context('spawn')
        
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
            futures = [executor.submit(worker_execute_episode, arg) for arg in worker_args_list]
            for future in tqdm(concurrent.futures.as_completed(futures), total=num_games, desc=f"Self-Play (Phase {current_phase})"):
                try:
                    examples, length, depth, latest_won, latest_drawn = future.result()
                    iteration_data.extend(examples)
                    if not latest_drawn:
                        phase_decisive += 1
                        if latest_won:
                            phase_wins += 1
                except Exception as e:
                    print(f"Worker execution failed: {e}")

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
            current_phase=np.array([current_phase], dtype=np.int32)
        )
        
        print(f"Saved {len(iteration_data)} examples to {filename}")
        return filename

if __name__ == "__main__":
    import torch
    import model_manager
    print("Testing SelfPlayGenerator locally...")
    
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
    generator.load_model(test_model_path)
    
    # Test generation of 2 games
    test_save_dir = os.path.join(PROJECT_ROOT, "storage", "test_replay")
    print(f"Running 2 self-play games to test worker pipeline...")
    saved_file = generator.play_games(num_games=2, save_dir=test_save_dir)
    
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