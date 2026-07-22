import os
import copy
import json
import random
import numpy as np
from collections import deque
from tqdm import tqdm
import multiprocessing
import config
import concurrent.futures
import io
import torch

_WORKER_SHARED_STATE = None
_SELFPLAY_EXECUTOR = None
_SELFPLAY_EXECUTOR_MODEL_SIGNATURE = None
_SELFPLAY_EXECUTOR_MAX_WORKERS = None


def _get_model_signature(model_path):
    if not model_path or not os.path.exists(model_path):
        return (model_path, None, None)
    return (model_path, os.path.getmtime(model_path), os.path.getsize(model_path))


def init_worker_process(latest_model_path, mcts_class, args, game_size):
    """Load the model once per worker process and reuse it across tasks."""
    global _WORKER_SHARED_STATE

    from game import DotsAndBoxesGame
    from model import NNetWrapper

    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    dummy_game = DotsAndBoxesGame(size=game_size)
    nnet = NNetWrapper(dummy_game, args)

    state_dict = torch.load(latest_model_path, map_location='cpu', weights_only=False)
    nnet.nnet.load_state_dict(state_dict['state_dict'] if 'state_dict' in state_dict else state_dict)
    nnet.nnet.eval()
    mcts_latest = mcts_class(nnet, args)

    _WORKER_SHARED_STATE = {
        "dummy_game": dummy_game,
        "mcts_latest": mcts_latest,
    }


def get_selfplay_executor(latest_model_path, mcts_class, args, game_size, max_workers=None, mp_context=None):
    """Create or reuse a process pool for self-play batches."""
    global _SELFPLAY_EXECUTOR, _SELFPLAY_EXECUTOR_MODEL_SIGNATURE, _SELFPLAY_EXECUTOR_MAX_WORKERS

    if max_workers is None:
        max_workers = max(1, min(config.MAX_WORKERS, multiprocessing.cpu_count() - 1))

    model_signature = _get_model_signature(latest_model_path)

    if (
        _SELFPLAY_EXECUTOR is not None
        and _SELFPLAY_EXECUTOR_MODEL_SIGNATURE == model_signature
        and _SELFPLAY_EXECUTOR_MAX_WORKERS == max_workers
    ):
        return _SELFPLAY_EXECUTOR

    if _SELFPLAY_EXECUTOR is not None:
        _SELFPLAY_EXECUTOR.shutdown(wait=False, cancel_futures=True)

    if mp_context is None:
        mp_context = multiprocessing.get_context('spawn')

    _SELFPLAY_EXECUTOR = concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_context,
        initializer=init_worker_process,
        initargs=(latest_model_path, mcts_class, args, game_size),
    )
    _SELFPLAY_EXECUTOR_MODEL_SIGNATURE = model_signature
    _SELFPLAY_EXECUTOR_MAX_WORKERS = max_workers
    return _SELFPLAY_EXECUTOR


def build_worker_chunks(
    game_size,
    latest_model_path,
    mcts_class,
    args,
    episode_specs,
    max_workers=None,
    target_games_per_worker=4,
):
    """Create chunked self-play worker args so each process can reuse its model load."""
    if not episode_specs:
        return []

    cpu_workers = max(1, min(config.MAX_WORKERS, multiprocessing.cpu_count() - 1))
    if max_workers is not None:
        cpu_workers = min(cpu_workers, max_workers)

    target_games_per_worker = max(1, target_games_per_worker)
    chunk_limited_workers = (len(episode_specs) + target_games_per_worker - 1) // target_games_per_worker
    num_workers = max(1, min(len(episode_specs), cpu_workers, chunk_limited_workers))
    chunk_size = (len(episode_specs) + num_workers - 1) // num_workers
    return [
        (game_size, latest_model_path, mcts_class, args, episode_specs[i:i + chunk_size])
        for i in range(0, len(episode_specs), chunk_size)
    ]


def worker_execute_episode(worker_args):
    """Compatibility wrapper for code that still submits one episode per task."""
    game_size, latest_model_path, mcts_class, args, game_sequence, start_fill_pct, opp_type, opp_path = worker_args
    results = worker_execute_episode_chunk((
        game_size,
        latest_model_path,
        mcts_class,
        args,
        [(game_sequence, start_fill_pct, opp_type, opp_path)],
    ))
    return results[0]


def worker_execute_episode_chunk(worker_args):
    """Generates multiple self-play games after loading models once in this process."""
    game_size, latest_model_path, mcts_class, args, episode_specs = worker_args

    # Must import inside worker to avoid pickle issues under spawn.
    import copy
    import random
    from game import DotsAndBoxesGame
    from model import NNetWrapper

    global _WORKER_SHARED_STATE
    if _WORKER_SHARED_STATE is None:
        init_worker_process(latest_model_path, mcts_class, args, game_size)

    dummy_game = _WORKER_SHARED_STATE["dummy_game"]
    mcts_latest = _WORKER_SHARED_STATE["mcts_latest"]

    opponent_cache = {}

    def get_opponent(opp_type, opp_path):
        if opp_type == "self":
            return None

        key = (opp_type, opp_path)
        if key in opponent_cache:
            return opponent_cache[key]

        if opp_type in ["best", "past"]:
            opp_net = NNetWrapper(dummy_game, args)
            opp_state_dict = torch.load(opp_path, map_location='cpu', weights_only=False)
            opp_net.nnet.load_state_dict(opp_state_dict['state_dict'] if 'state_dict' in opp_state_dict else opp_state_dict)
            opp_net.nnet.eval()
            mcts_opp = mcts_class(opp_net, args)

            def agent_opp(g, t):
                pi = mcts_opp.play(g, temp=t, add_root_noise=True)
                return pi, mcts_opp.max_depth_reached

        elif opp_type == "alpha_beta_0.1s":
            from bots.alpha_beta import AlphaBetaPlayer
            baseline = AlphaBetaPlayer(name="AlphaBeta", time_limit=0.1)

            def agent_opp(g, t):
                move = baseline.get_move(copy.deepcopy(g))
                pi = np.zeros(g.N_LINES, dtype=np.float32)
                pi[move] = 1.0
                return pi, 0

        elif opp_type == "mcts_0.1s":
            from bots.mcts_x import MCTSGAgent
            baseline = MCTSGAgent(name="MCTS", time_limit=0.1)

            def agent_opp(g, t):
                move = baseline.get_move(copy.deepcopy(g))
                pi = np.zeros(g.N_LINES, dtype=np.float32)
                pi[move] = 1.0
                return pi, 0

        elif opp_type == "random":
            import random as rand

            def agent_opp(g, t):
                valid_moves = g.get_valid_moves()
                move = rand.choice(valid_moves) if valid_moves else 0
                pi = np.zeros(g.N_LINES, dtype=np.float32)
                pi[move] = 1.0
                return pi, 0

        elif opp_type == "greedy":
            from bots.greedy import GreedyPlayer
            baseline = GreedyPlayer(name="Greedy")

            def agent_opp(g, t):
                move = baseline.get_move(copy.deepcopy(g))
                pi = np.zeros(g.N_LINES, dtype=np.float32)
                pi[move] = 1.0
                return pi, 0

        elif opp_type == "greedy_chain":
            from bots.greedy_improve import GreedyChainPlayer
            baseline = GreedyChainPlayer(name="GreedyChain")

            def agent_opp(g, t):
                move = baseline.get_move(copy.deepcopy(g))
                pi = np.zeros(g.N_LINES, dtype=np.float32)
                pi[move] = 1.0
                return pi, 0

        elif opp_type == "ucla_bot_v3":
            from bots.ucla_bot import UCLABot_v3
            baseline = UCLABot_v3(name="UCLABot_v3")

            def agent_opp(g, t):
                move = baseline.get_move(copy.deepcopy(g))
                pi = np.zeros(g.N_LINES, dtype=np.float32)
                pi[move] = 1.0
                return pi, 0

        elif opp_type == "simple_bot":
            from bots.simple_bot import SimpleBot
            baseline = SimpleBot(name="SimpleBot")

            def agent_opp(g, t):
                move = baseline.get_move(copy.deepcopy(g))
                pi = np.zeros(g.N_LINES, dtype=np.float32)
                pi[move] = 1.0
                return pi, 0

        elif opp_type == "simple_bot_v2":
            from bots.simple_bot_v2 import SimpleBotV2
            baseline = SimpleBotV2(name="SimpleBotV2")

            def agent_opp(g, t):
                move = baseline.get_move(copy.deepcopy(g))
                pi = np.zeros(g.N_LINES, dtype=np.float32)
                pi[move] = 1.0
                return pi, 0

        else:
            raise ValueError(f"Unknown opponent type: {opp_type}")

        opponent_cache[key] = agent_opp
        return agent_opp

    def run_episode(game_sequence, start_fill_pct, opp_type, opp_path):
        if opp_type in ["best", "past"] and opp_path is None:
            opp_type = "self"

        agent_opp = get_opponent(opp_type, opp_path)
        p1_is_latest = random.choice([True, False])

        train_examples = []
        game = DotsAndBoxesGame(size=game_size, starting_player=1, early_stopping=True)

        # Reverse Curriculum Logic: Pre-fill the board using the historical log sequence.
        if game_sequence is not None and start_fill_pct >= 0.001:
            target_move_index = int(len(game_sequence) * start_fill_pct)
            for move in game_sequence[:target_move_index]:
                game.execute_move(move)

        episode_step = 0
        depths = []
        temperature_initial = getattr(args, "temperature_initial", 1.0)
        temperature_final = getattr(args, "temperature_final", 0.0)
        drop_move = getattr(args, "temperature_drop_move", getattr(args, "temp_threshold", 15))

        while game.is_running():
            episode_step += 1
            move_number = int(np.count_nonzero(game.l))
            temp = float(temperature_initial if move_number < drop_move else temperature_final)

            is_latest_turn = (game.current_player == 1) == p1_is_latest

            if is_latest_turn or opp_type == "self":
                pi = mcts_latest.play(game, temp=temp, add_root_noise=True)
                depths.append(mcts_latest.max_depth_reached)

                canonical_lines = game.get_canonical_lines()
                canonical_boxes = game.get_canonical_boxes()
                train_examples.append([canonical_lines, canonical_boxes, game.current_player, pi, None])

                action = np.random.choice(len(pi), p=pi)
            else:
                pi, depth = agent_opp(game, temp)
                if depth > 0:
                    depths.append(depth)

                # We only train the network on data generated by the latest network policy.
                action = np.random.choice(len(pi), p=pi)

            game.execute_move(action)

        r = game.result
        avg_depth = sum(depths) / len(depths) if depths else 0

        final_examples = []
        for x in train_examples:
            val = r if x[2] == 1 else -r
            final_examples.append((x[0], x[1], x[3], val))

        latest_won = (r == 1 if p1_is_latest else r == -1)
        latest_drawn = (r == 0)
        return final_examples, episode_step, avg_depth, opp_type, latest_won, latest_drawn

    return [run_episode(*episode_spec) for episode_spec in episode_specs]


def worker_play_single(worker_args):
    """Plays a single arena match in an isolated worker process."""
    game_size, latest_model_path, pnet_bytes, mcts_class, args, p1_starts, opponent_type = worker_args
    
    from game import DotsAndBoxesGame
    from model import NNetWrapper
    import copy
    
    dummy_game = DotsAndBoxesGame(size=game_size)
    
    # Init new network (Agent 1)
    nnet = NNetWrapper(dummy_game, args)
    state_dict = torch.load(latest_model_path, map_location='cpu', weights_only=False)
    nnet.nnet.load_state_dict(state_dict['state_dict'] if 'state_dict' in state_dict else state_dict)
    nnet.nnet.eval()
    mcts1 = mcts_class(nnet, args)
    def agent1(g):
        pi = mcts1.play(g, temp=0, add_root_noise=False)
        return np.argmax(pi)
        
    # Init opponent (Agent 2)
    if opponent_type == "random":
        import random
        def agent2(g):
            valid_moves = g.get_valid_moves()
            return random.choice(valid_moves) if valid_moves else None
    elif opponent_type == "alpha_beta_0.1s":
        from bots.alpha_beta import AlphaBetaPlayer
        baseline = AlphaBetaPlayer(name="AlphaBeta_0.1s", time_limit=0.1)
        def agent2(g):
            return baseline.get_move(copy.deepcopy(g))
    elif opponent_type == "mcts_0.1s":
        from bots.mcts_x import MCTSGAgent
        baseline = MCTSGAgent(name="MCTS_0.1s", time_limit=0.1)
        def agent2(g):
            return baseline.get_move(copy.deepcopy(g))
    elif opponent_type == "greedy":
        from bots.greedy import GreedyPlayer
        baseline = GreedyPlayer(name="Greedy")
        def agent2(g):
            return baseline.get_move(copy.deepcopy(g))
    elif opponent_type == "greedy_chain":
        from bots.greedy_improve import GreedyChainPlayer
        baseline = GreedyChainPlayer(name="GreedyChain")
        def agent2(g):
            return baseline.get_move(copy.deepcopy(g))
    elif opponent_type == "ucla_bot_v3":
        from bots.ucla_bot import UCLABot_v3
        baseline = UCLABot_v3(name="UCLABot_v3")
        def agent2(g):
            return baseline.get_move(copy.deepcopy(g))
    elif opponent_type == "simple_bot":
        from bots.simple_bot import SimpleBot
        baseline = SimpleBot(name="SimpleBot")
        def agent2(g):
            return baseline.get_move(copy.deepcopy(g))
    elif opponent_type == "simple_bot_v2":
        from bots.simple_bot_v2 import SimpleBotV2
        baseline = SimpleBotV2(name="SimpleBotV2")
        def agent2(g):
            return baseline.get_move(copy.deepcopy(g))
    else: # "pnet"
        pnet = NNetWrapper(dummy_game, args)
        pbuffer = io.BytesIO(pnet_bytes)
        pstate_dict = torch.load(pbuffer, weights_only=False)
        pnet.nnet.load_state_dict(pstate_dict)
        pnet.nnet.eval()
        mcts2 = mcts_class(pnet, args)
        def agent2(g):
            pi = mcts2.play(g, temp=0, add_root_noise=False)
            return np.argmax(pi)
            
    game = DotsAndBoxesGame(size=game_size, starting_player=1, early_stopping=True)
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
        self.memory = deque(maxlen=self.args.maxlen_queue)
        self.train_examples_history = deque([], maxlen=self.args.keep_history_iters)
        
        self.current_phase = 0
        self.phase_wins = 0
        self.phase_decisive = 0
        self.iterations_in_current_phase = 0
        
        # Load Reverse Curriculum Logs
        self.json_logs = []
        log_path = "game_logs.jsonl"
        expected_moves = 2 * self.game_size * (self.game_size + 1)
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            moves = record.get("moves", [])
                            # Only load games that match our current board size exactly
                            if moves and len(moves) == expected_moves:
                                self.json_logs.append(moves)
                        except Exception:
                            pass
        print(f"Loaded {len(self.json_logs)} historical game sequences of size {self.game_size}x{self.game_size} for Reverse Curriculum.")
        
        # TensorBoard logging setup
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.args.checkpoint_dir)

    def learn(self):
        """
        Training loop matching the continuous-update and baseline-evaluation approach.
        """
        starting_iteration = 1
        
        # Force 'spawn' context to safely use CUDA in worker processes
        mp_context = multiprocessing.get_context('spawn')
        
        # Reverse Curriculum Setup
        start_fill_pct = 0.70
        decay_iterations = 10
        decay_step = 0.70 / decay_iterations if decay_iterations > 0 else 0
        
        for i in range(starting_iteration, self.args.num_iters + 1):
            print(f"\n#################### Iteration {i}/{self.args.num_iters} ####################")
            
            self.writer.add_scalar("Log/Start_Sequence_Fill", start_fill_pct, i)
            
            import glob
            # Save latest model to temp file for workers to avoid 136MB IPC transfer
            temp_latest_path = os.path.join(self.args.checkpoint_dir, 'temp_latest.pth.tar')
            self.nnet.save_checkpoint(folder=self.args.checkpoint_dir, filename='temp_latest.pth.tar')
            
            # Find past checkpoints
            past_checkpoints = sorted(glob.glob(os.path.join(self.args.checkpoint_dir, 'checkpoint_*.pth.tar')))
            past_checkpoints = past_checkpoints[-5:] if past_checkpoints else []
            best_path = os.path.join(self.args.checkpoint_dir, 'best.pth.tar')
            
            # Sample random sequences for each episode
            if self.json_logs and start_fill_pct >= 0.001:
                sampled_sequences = [random.choice(self.json_logs) for _ in range(self.args.num_eps)]
            else:
                sampled_sequences = [None] * self.args.num_eps
            
            episode_specs = []
            
            phases_config = [
                [("greedy", 0.05)],
                [("greedy_chain", 0.05)],
                [("simple_bot", 0.05)],
                [("simple_bot_v2", 0.05)],
                [("ucla_bot_v3", 0.05)],
                [("self", 0.5), ("best", 0.1), ("past", 0.15)]
            ]
                
            current_pool = []
            for phase_idx in range(self.current_phase + 1):
                current_pool.extend(phases_config[phase_idx])
                
            total_prob = sum(p for _, p in current_pool)
            normalized_probs = [p / total_prob for _, p in current_pool]
            
            for seq in sampled_sequences:
                opp_type = np.random.choice([name for name, _ in current_pool], p=normalized_probs)
                opp_path = None
                
                if opp_type == "best":
                    opp_path = best_path if os.path.exists(best_path) else None
                    if opp_path is None: opp_type = "self"
                elif opp_type == "past":
                    opp_path = random.choice(past_checkpoints) if past_checkpoints else None
                    if opp_path is None: opp_type = "self"
                    
                episode_specs.append((seq, start_fill_pct, opp_type, opp_path))
            
            # 1. Self-Play (Parallelized)
            print(f"------------ Self-Play (Fill: {start_fill_pct*100:.1f}%) ------------")
            iteration_data = []
            episode_lengths = []
            episode_depths = []
            
            worker_args_list = build_worker_chunks(
                self.game_size,
                temp_latest_path,
                self.MCTS,
                self.args,
                episode_specs,
                max_workers=8,
            )
            print(f"Using {len(worker_args_list)} self-play worker processes for {len(episode_specs)} games.")

            with concurrent.futures.ProcessPoolExecutor(max_workers=len(worker_args_list), mp_context=mp_context) as executor:
                futures = {executor.submit(worker_execute_episode_chunk, arg): len(arg[4]) for arg in worker_args_list}
                with tqdm(total=len(episode_specs), desc="Self Play") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        chunk_size = futures[future]
                        chunk_results = future.result()
                        for examples, length, depth, latest_won, latest_drawn in chunk_results:
                            iteration_data.append(examples)
                            episode_lengths.append(length)
                            episode_depths.append(depth)
                            if not latest_drawn:
                                self.phase_decisive += 1
                                if latest_won:
                                    self.phase_wins += 1
                        pbar.update(chunk_size)
                            
            # Calculate and log phase winrate
            phase_winrate = self.phase_wins / self.phase_decisive if self.phase_decisive > 0 else 0.5
            self.writer.add_scalar('Curriculum/Current_Phase', self.current_phase, i)
            self.writer.add_scalar('Curriculum/Phase_Winrate', phase_winrate, i)
            print(f"Phase {self.current_phase} Winrate: {phase_winrate:.1%} ({self.phase_wins}/{self.phase_decisive})")
            
            self.iterations_in_current_phase += 1
            if (phase_winrate >= 0.60 or self.iterations_in_current_phase >= 200) and self.current_phase < 5:
                reason = "winrate >= 60%" if phase_winrate >= 0.60 else "max iterations reached"
                print(f"Phase {self.current_phase} cleared ({reason})! Advancing to Phase {self.current_phase + 1}...")
                self.current_phase += 1
                self.phase_wins = 0
                self.phase_decisive = 0
                self.iterations_in_current_phase = 0
                
            # Log SelfPlay Metrics
            avg_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0
            avg_depth = sum(episode_depths) / len(episode_depths) if episode_depths else 0
            self.writer.add_scalar('Log/Game_Length', avg_length, i)
            self.writer.add_scalar('Log/Average_Tree_Depth', avg_depth, i)
            
            # 2. Add Data to Memory
            flat_data = [example for game in iteration_data for example in game]
            self.memory.extend(flat_data)
            
            self.writer.add_scalar('Log/Memory_Size', len(self.memory), i)
            
            # 3. Neural Network Training
            print("\n---------- Neural Network Training -----------")
            pi_loss, v_loss, total_loss = self.nnet.train(list(self.memory))
            
            self.writer.add_scalar('Log/Policy_Loss', pi_loss, i)
            self.writer.add_scalar('Log/Value_Loss', v_loss, i)
            self.writer.add_scalar('Log/Total_Loss', total_loss, i)
            
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
                match_args_pnet.append((self.game_size, temp_latest_path, pnet_bytes, self.MCTS, self.args, p1_starts, "pnet"))
                
            with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=mp_context) as executor:
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
            self.writer.add_scalar('Evaluation/Win_Rate_Vs_Old', win_rate_vs_old, i)
            
            if total_decisive > 0 and win_rate_vs_old >= self.args.update_threshold:
                print("ACCEPTING NEW MODEL")
                self.pnet.nnet.load_state_dict(self.nnet.nnet.state_dict())
                self.nnet.save_checkpoint(folder=self.args.checkpoint_dir, filename='best.pth.tar')
                
                # Update version.txt with the new accepted checkpoint id
                version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version.txt")
                if os.path.exists(version_file):
                    try:
                        with open(version_file, "r") as f:
                            v_lines = f.readlines()
                        with open(version_file, "w") as f:
                            for line in v_lines:
                                if line.startswith("last_updated_model:"):
                                    f.write(f"last_updated_model: {i}\n")
                                else:
                                    f.write(line)
                    except Exception as e:
                        print(f"Error updating version.txt: {e}")
            else:
                print("REJECTING NEW MODEL")
                self.nnet.nnet.load_state_dict(self.pnet.nnet.state_dict())
                
            # b. Play against Evaluators
            eval_opponents = {
                "greedy": "Greedy",
                "greedy_chain": "GreedyChain",
                "ucla_bot_v3": "UCLABot_v3",
                "simple_bot": "SimpleBot",
                "simple_bot_v2": "SimpleBotV2"
            }
            
            for opp_type, opp_name in eval_opponents.items():
                print(f"Pitting against {opp_name}...")
                ewins, elosses, edraws = 0, 0, 0
                eval_games = 10
                match_args_eval = []
                for idx in range(eval_games):
                    p1_starts = idx < (eval_games // 2)
                    match_args_eval.append((self.game_size, temp_latest_path, None, self.MCTS, self.args, p1_starts, opp_type))
                    
                with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=mp_context) as executor:
                    futures = [executor.submit(worker_play_single, arg) for arg in match_args_eval]
                    for future in tqdm(concurrent.futures.as_completed(futures), total=eval_games, desc=f"Vs {opp_name}"):
                        res = future.result()
                        if res == 1: ewins += 1
                        elif res == -1: elosses += 1
                        else: edraws += 1
                        
                print(f"Vs {opp_name} -> Wins: {ewins} | Losses: {elosses} | Draws: {edraws}")
                
                eval_decisive = ewins + elosses
                win_rate_vs_eval = ewins / eval_decisive if eval_decisive > 0 else 0.5
                self.writer.add_scalar(f'Evaluation/Win_Rate_Vs_{opp_name}', win_rate_vs_eval, i)
            
            # 5. Save Checkpoints
            print("\\nSaving Checkpoint...")
            self.nnet.save_checkpoint(folder=self.args.checkpoint_dir, filename=f'checkpoint_{i}.pth.tar')
            self.writer.flush()
