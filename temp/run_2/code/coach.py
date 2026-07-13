import os
import copy
import json
import random
import numpy as np
from collections import deque
from tqdm import tqdm
import multiprocessing
import concurrent.futures
import io
import torch
from torch.utils.tensorboard import SummaryWriter

def worker_execute_episode(worker_args):
    """Generates self-play data using an isolated worker process."""
    game_size, nnet_bytes, mcts_class, args, game_sequence, start_fill_pct = worker_args
    
    # Must import inside worker to avoid pickle issues
    from game import DotsAndBoxesGame
    from model import NNetWrapper
    
    dummy_game = DotsAndBoxesGame(size=game_size)
    nnet = NNetWrapper(dummy_game, args)
    
    # Load from bytes safely
    buffer = io.BytesIO(nnet_bytes)
    state_dict = torch.load(buffer, weights_only=True)
    nnet.nnet.load_state_dict(state_dict)
    nnet.nnet.eval()
    
    train_examples = []
    game = DotsAndBoxesGame(size=game_size, starting_player=1)
    
    # Reverse Curriculum Logic: Pre-fill the board using the historical log sequence
    if game_sequence is not None and start_fill_pct >= 0.001:
        target_move_index = int(len(game_sequence) * start_fill_pct)
        for move in game_sequence[:target_move_index]:
            game.execute_move(move)
            
    episode_step = 0
    depths = []
    
    mcts = mcts_class(nnet, args)

    while game.is_running():
        episode_step += 1
        temp = int(episode_step < args.temp_threshold)
        
        # play() returns probabilities
        pi = mcts.play(game, temp=temp)
        depths.append(mcts.max_depth_reached)
        
        # Store data from the perspective of the current player
        canonical_lines = game.get_canonical_lines()
        canonical_boxes = game.get_canonical_boxes()
        
        train_examples.append([canonical_lines, canonical_boxes, game.current_player, pi, None])

        action = np.random.choice(len(pi), p=pi)
        game.execute_move(action)

    r = game.result
    avg_depth = sum(depths) / len(depths) if depths else 0
    
    # Process final reward from perspective of player who played that turn
    final_examples = []
    for x in train_examples:
        val = r if x[2] == 1 else -r
        final_examples.append((x[0], x[1], x[3], val))
        
    return final_examples, episode_step, avg_depth

def worker_play_single(worker_args):
    """Plays a single arena match in an isolated worker process."""
    game_size, nnet_bytes, pnet_bytes, mcts_class, args, p1_starts, use_baseline = worker_args
    
    from game import DotsAndBoxesGame
    from model import NNetWrapper
    
    dummy_game = DotsAndBoxesGame(size=game_size)
    
    # Init new network (Agent 1)
    nnet = NNetWrapper(dummy_game, args)
    buffer = io.BytesIO(nnet_bytes)
    state_dict = torch.load(buffer, weights_only=True)
    nnet.nnet.load_state_dict(state_dict)
    nnet.nnet.eval()
    mcts1 = mcts_class(nnet, args)
    def agent1(g):
        pi = mcts1.play(g, temp=0)
        return np.argmax(pi)
        
    # Init opponent (Agent 2)
    if use_baseline:
        from bots.alpha_beta import AlphaBetaPlayer
        baseline = AlphaBetaPlayer(name="Baseline", time_limit=0.5)
        def agent2(g):
            return baseline.get_move(copy.deepcopy(g))
    else:
        pnet = NNetWrapper(dummy_game, args)
        pbuffer = io.BytesIO(pnet_bytes)
        pstate_dict = torch.load(pbuffer, weights_only=True)
        pnet.nnet.load_state_dict(pstate_dict)
        pnet.nnet.eval()
        mcts2 = mcts_class(pnet, args)
        def agent2(g):
            pi = mcts2.play(g, temp=0)
            return np.argmax(pi)
            
    game = DotsAndBoxesGame(size=game_size, starting_player=1)
    players = {1: agent1, -1: agent2} if p1_starts else {1: agent2, -1: agent1}
    
    while game.is_running():
        cur_player = game.current_player
        action = players[cur_player](game)
        game.execute_move(action)
        
    # Return 1 if agent1 won, -1 if agent2 won, 0 for draw
    if p1_starts:
        return game.result
    else:
        return -game.result


class AlphaZeroTrainer:
    def __init__(self, game_size, nnet, pnet, mcts_class, args):
        self.game_size = game_size
        self.nnet = nnet
        self.pnet = pnet
        self.MCTS = mcts_class
        self.args = args
        self.train_examples_history = deque([], maxlen=self.args.keep_history_iters)
        
        # Load Reverse Curriculum Logs
        self.json_logs = []
        log_path = "logs/game_logs.jsonl"
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            moves = record.get("moves", [])
                            if moves:
                                self.json_logs.append(moves)
                        except Exception:
                            pass
        print(f"Loaded {len(self.json_logs)} historical game sequences for Reverse Curriculum.")
        
        # TensorBoard logging setup
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.args.checkpoint_dir, 'logs'))

    def learn(self):
        """
        Training loop matching the continuous-update and baseline-evaluation approach.
        """
        starting_iteration = 1
        
        # Force 'spawn' context to safely use CUDA in worker processes
        mp_context = multiprocessing.get_context('spawn')
        
        # Reverse Curriculum Setup
        start_fill_pct = 0.70
        decay_iterations = int(self.args.num_iters * 0.5)
        decay_step = 0.70 / decay_iterations if decay_iterations > 0 else 0
        
        for i in range(starting_iteration, self.args.num_iters + 1):
            print(f"\n#################### Iteration {i}/{self.args.num_iters} ####################")
            
            self.writer.add_scalar("Curriculum/StartFillPct", start_fill_pct, i)
            
            # Serialize state dict to bytes to pass to workers safely without PyTorch FD sharing issues
            buffer = io.BytesIO()
            torch.save(self.nnet.nnet.state_dict(), buffer)
            nnet_bytes = buffer.getvalue()
            
            # Sample random sequences for each episode
            if self.json_logs and start_fill_pct >= 0.001:
                sampled_sequences = [random.choice(self.json_logs) for _ in range(self.args.num_eps)]
            else:
                sampled_sequences = [None] * self.args.num_eps
            
            worker_args_list = [
                (self.game_size, nnet_bytes, self.MCTS, self.args, seq, start_fill_pct) 
                for seq in sampled_sequences
            ]
            
            # 1. Self-Play (Parallelized)
            print(f"------------ Self-Play (Fill: {start_fill_pct*100:.1f}%) ------------")
            iteration_data = []
            episode_lengths = []
            episode_depths = []
            
            with concurrent.futures.ProcessPoolExecutor(mp_context=mp_context) as executor:
                futures = [executor.submit(worker_execute_episode, arg) for arg in worker_args_list]
                for future in tqdm(concurrent.futures.as_completed(futures), total=self.args.num_eps, desc="Self Play"):
                    examples, length, depth = future.result()
                    iteration_data.append(examples)
                    episode_lengths.append(length)
                    episode_depths.append(depth)
                
            # Log SelfPlay Metrics
            avg_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0
            avg_depth = sum(episode_depths) / len(episode_depths) if episode_depths else 0
            self.writer.add_scalar('SelfPlay/Game_Length', avg_length, i)
            self.writer.add_scalar('MCTS/Average_Tree_Depth', avg_depth, i)
            
            # 2. Augment Data & Add to Memory
            augmented_data = self.augment_data(iteration_data)
            flat_data = [example for game in augmented_data for example in game]
            self.memory.extend(flat_data)
            
            self.writer.add_scalar('SelfPlay/Memory_Size', len(self.memory), i)
            
            # 3. Neural Network Training
            print("\n---------- Neural Network Training -----------")
            pi_loss, v_loss, total_loss = self.nnet.train(list(self.memory))
            
            self.writer.add_scalar('Loss/Policy_Loss', pi_loss, i)
            self.writer.add_scalar('Loss/Value_Loss', v_loss, i)
            self.writer.add_scalar('Loss/Total_Loss', total_loss, i)
            
            # 4. Model Comparison (Parallelized)
            print("\n-------------- Model Comparison (Multiprocessing) --------------")
            
            pwins, plosses, pdraws = 0, 0, 0
            
            pbuffer = io.BytesIO()
            torch.save(self.pnet.nnet.state_dict(), pbuffer)
            pnet_bytes = pbuffer.getvalue()
            
            # a. Play against Previous Network
            print("Pitting against Previous Network (pnet)...")
            half_games = self.args.arena_games // 2
            match_args_pnet = []
            for idx in range(self.args.arena_games):
                p1_starts = idx < half_games
                match_args_pnet.append((self.game_size, nnet_bytes, pnet_bytes, self.MCTS, self.args, p1_starts, False))
                
            with concurrent.futures.ProcessPoolExecutor(mp_context=mp_context) as executor:
                futures = [executor.submit(worker_play_single, arg) for arg in match_args_pnet]
                for future in tqdm(concurrent.futures.as_completed(futures), total=self.args.arena_games, desc="Vs PNet"):
                    res = future.result()
                    if res == 1: pwins += 1
                    elif res == -1: plosses += 1
                    else: pdraws += 1
            
            # Decay Curriculum
            start_fill_pct = max(0.0, start_fill_pct - decay_step)
            
            print(f"\nIteration {i} complete. Vs PNet -> Wins: {pwins} | Losses: {plosses} | Draws: {pdraws}")
            
            total_decisive = pwins + plosses
            win_rate_vs_old = pwins / total_decisive if total_decisive > 0 else 0.5
            self.writer.add_scalar('Arena/Win_Rate_Vs_Old', win_rate_vs_old, i)
            
            if total_decisive > 0 and win_rate_vs_old >= self.args.update_threshold:
                print("ACCEPTING NEW MODEL")
                self.pnet.nnet.load_state_dict(self.nnet.nnet.state_dict())
                self.nnet.save_checkpoint(folder=self.args.checkpoint_dir, filename='best.pth.tar')
            else:
                print("REJECTING NEW MODEL")
                self.nnet.nnet.load_state_dict(self.pnet.nnet.state_dict())
                
            # b. Play against Baseline Heuristic (AlphaBeta)
            print("Pitting against Baseline (AlphaBeta)...")
            bwins, blosses, bdraws = 0, 0, 0
            baseline_games = 10
            match_args_baseline = []
            for idx in range(baseline_games):
                p1_starts = idx < (baseline_games // 2)
                match_args_baseline.append((self.game_size, nnet_bytes, None, self.MCTS, self.args, p1_starts, True))
                
            with concurrent.futures.ProcessPoolExecutor(mp_context=mp_context) as executor:
                futures = [executor.submit(worker_play_single, arg) for arg in match_args_baseline]
                for future in tqdm(concurrent.futures.as_completed(futures), total=baseline_games, desc="Vs Baseline"):
                    res = future.result()
                    if res == 1: bwins += 1
                    elif res == -1: blosses += 1
                    else: bdraws += 1
                    
            print(f"Vs Baseline -> Wins: {bwins} | Losses: {blosses} | Draws: {bdraws}")
            
            baseline_decisive = bwins + blosses
            win_rate_vs_heuristic = bwins / baseline_decisive if baseline_decisive > 0 else 0.5
            self.writer.add_scalar('Baseline/Win_Rate_Vs_Heuristic', win_rate_vs_heuristic, i)
            
            # 5. Save Checkpoints
            print("\\nSaving Checkpoint...")
            self.nnet.save_checkpoint(folder=self.args.checkpoint_dir, filename=f'checkpoint_{i}.pth.tar')

    @staticmethod
    def augment_data(train_examples_per_game: list):
        data_augmented = []
        from game import DotsAndBoxesGame
        for train_examples in train_examples_per_game:
            train_examples_augmented = []
            for lines, boxes, p, v in train_examples:
                train_examples_augmented.extend(zip(
                    DotsAndBoxesGame.get_rotations_and_reflections_lines(lines),
                    DotsAndBoxesGame.get_rotations_and_reflections_boxes(boxes),
                    DotsAndBoxesGame.get_rotations_and_reflections_lines(np.asarray(p)),
                    [v] * 8
                ))
            
            formatted_augmented = []
            for aug_lines, aug_boxes, aug_p, aug_v in train_examples_augmented:
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
                formatted_augmented.append((board_state, aug_p, aug_v))
                
            data_augmented.append(formatted_augmented)

        return data_augmented