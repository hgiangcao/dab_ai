import os
import sys
import numpy as np
import concurrent.futures
from tqdm import tqdm
import multiprocessing
import torch

# Add parent dir to path to import game, model, mcts
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import config
import model_manager
from game import DotsAndBoxesGame
from model import NNetWrapper, dotdict
from mcts import MCTS

# Default evaluation arguments
eval_args = dotdict({
    'lr': config.LEARNING_RATE,
    'epochs': config.EPOCHS,
    'batch_size': config.BATCH_SIZE,
    'num_channels': 256,
    'num_res_blocks': 10, 
    'l2_reg': 1e-4,
    'n_simulations': config.MCTS_NUM_SIMULATIONS,
    'c_puct': config.MCTS_C_PUCT,
    'dirichlet_eps': 0.0,
    'dirichlet_alpha': config.MCTS_DIRICHLET_ALPHA,
    'time_limit': None,  # Use simulation-count mode (MCTS_NUM_SIMULATIONS), never time-limited
    'device': 'cpu'  # CPU allows us to run evaluation in parallel efficiently
})

# ─── Per-process model cache (populated once by pool initializer) ──────────────
_EVAL_WORKER_STATE = None

def _init_eval_worker(candidate_path, opp_identifier):
    """
    Load models once per worker process and cache them globally.
    Called by ProcessPoolExecutor initializer — never called manually.
    """
    global _EVAL_WORKER_STATE
    import copy

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    game = DotsAndBoxesGame(size=5, early_stopping=True)

    # Always load candidate
    cand_net = NNetWrapper(game, eval_args)
    cand_state = torch.load(candidate_path, map_location='cpu', weights_only=False)
    cand_net.nnet.load_state_dict(cand_state['state_dict'] if 'state_dict' in cand_state else cand_state)
    cand_net.nnet.eval()
    mcts_cand = MCTS(cand_net, eval_args)

    # Load opponent model or prepare baseline
    opp_agent = None
    if opp_identifier in ("random", "greedy", "greedy_chain", "alpha_beta_0.1s", "mcts_0.1s"):
        opp_agent = opp_identifier  # resolved at play time
    elif os.path.exists(opp_identifier):
        best_net = NNetWrapper(game, eval_args)
        best_state = torch.load(opp_identifier, map_location='cpu', weights_only=False)
        best_net.nnet.load_state_dict(best_state['state_dict'] if 'state_dict' in best_state else best_state)
        best_net.nnet.eval()
        mcts_best = MCTS(best_net, eval_args)
        opp_agent = mcts_best
    else:
        opp_agent = "random"  # fallback

    _EVAL_WORKER_STATE = {
        "mcts_cand": mcts_cand,
        "opp_agent": opp_agent,
        "opp_identifier": opp_identifier,
    }


def _worker_play_single(p1_starts):
    """
    Play one game using the models already loaded in this worker process.
    """
    import copy, random as _rand

    global _EVAL_WORKER_STATE
    state = _EVAL_WORKER_STATE
    mcts_cand = state["mcts_cand"]
    opp_agent = state["opp_agent"]
    opp_identifier = state["opp_identifier"]

    def agent_cand(g):
        pi = mcts_cand.play(g, temp=0)
        return int(np.argmax(pi))

    if isinstance(opp_agent, str):
        if opp_agent == "random":
            def agent_opp(g):
                valid = g.get_valid_moves()
                return _rand.choice(valid) if valid else None
        elif opp_agent == "alpha_beta_0.1s":
            from bots.alpha_beta import AlphaBetaPlayer
            _baseline = AlphaBetaPlayer(name="AlphaBeta_0.1s", time_limit=0.1)
            def agent_opp(g):
                return _baseline.get_move(copy.deepcopy(g))
        elif opp_agent == "mcts_0.1s":
            from bots.mcts_x import MCTSGAgent
            _baseline = MCTSGAgent(name="MCTS_0.1s", time_limit=0.1)
            def agent_opp(g):
                return _baseline.get_move(copy.deepcopy(g))
        elif opp_agent == "greedy":
            from bots.greedy import GreedyPlayer
            _baseline = GreedyPlayer(name="Greedy")
            def agent_opp(g):
                return _baseline.get_move(copy.deepcopy(g))
        elif opp_agent == "greedy_chain":
            from bots.greedy_improve import GreedyChainPlayer
            _baseline = GreedyChainPlayer(name="GreedyChain")
            def agent_opp(g):
                return _baseline.get_move(copy.deepcopy(g))
        else:
            def agent_opp(g):
                valid = g.get_valid_moves()
                return _rand.choice(valid) if valid else None
    else:
        # opp_agent is an MCTS instance
        _mcts_opp = opp_agent
        def agent_opp(g):
            pi = _mcts_opp.play(g, temp=0)
            return int(np.argmax(pi))

    game = DotsAndBoxesGame(size=5, early_stopping=True)
    players = {1: agent_cand, -1: agent_opp} if p1_starts else {1: agent_opp, -1: agent_cand}

    while game.is_running():
        cur_player = game.current_player
        action = players[cur_player](game)
        game.execute_move(action)

    # Return 1 if candidate won, -1 if opponent won, 0 for draw
    return game.result if p1_starts else -game.result


def play_game(model1_path, model2_path, p1_starts=True):
    """
    Play one game (sequential, for debugging).
    """
    _init_eval_worker(model1_path, model2_path)
    return _worker_play_single(p1_starts)


def _run_pool(candidate_path, opp_identifier, num_games, desc, num_workers):
    """
    Helper: spin up a pool pre-initialised with models, run num_games, return (wins, losses, draws).
    Each worker loads the models ONCE via the initializer — no per-game reloads.
    """
    wins, losses, draws = 0, 0, 0
    half = num_games // 2
    p1_starts_list = [idx < half for idx in range(num_games)]

    mp_context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp_context,
        initializer=_init_eval_worker,
        initargs=(candidate_path, opp_identifier),
    ) as executor:
        futures = [executor.submit(_worker_play_single, p1) for p1 in p1_starts_list]
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_games, desc=desc):
            try:
                res = future.result()
                if res == 1:
                    wins += 1
                elif res == -1:
                    losses += 1
                else:
                    draws += 1
            except Exception as e:
                print(f"Match execution failed: {e}")

    return wins, losses, draws


def evaluate_model(candidate_model_path, best_model_path, num_games):
    """
    Run evaluation games in parallel across multiple CPU cores.
    Each worker loads both models once (via initializer), then plays multiple games.
    Return a summary dictionary.
    """
    # Each worker holds both models in RAM; capped by config.MAX_WORKERS to prevent resource exhaustion.
    num_workers = max(1, min(num_games, config.MAX_WORKERS, multiprocessing.cpu_count() - 1))
    print(f"Evaluating candidate model over {num_games} arena games using {num_workers} processes...")

    wins, losses, draws = _run_pool(
        candidate_model_path, best_model_path, num_games,
        desc="Arena Eval", num_workers=num_workers
    )

    total_decisive = wins + losses
    win_rate = wins / total_decisive if total_decisive > 0 else 0.5

    return {
        "wins": wins,
        "loss": losses,
        "draw": draws,
        "win_rate": win_rate
    }


def should_promote(result):
    """
    Decide if candidate becomes best based on win_rate > PROMOTION_THRESHOLD
    """
    return result["win_rate"] >= config.PROMOTION_THRESHOLD


def evaluate_baselines(candidate_model_path, num_games=10):
    """
    Run evaluation games against heuristic baselines.
    Each worker loads only the candidate model (no opponent model needed for baselines).
    """
    baselines = {
        "random": "Random",
        "alpha_beta_0.1s": "AlphaBeta_0.1s",
        "mcts_0.1s": "MCTS_0.1s",
        "greedy": "Greedy",
        "greedy_chain": "GreedyChain"
    }

    # Baselines only need the candidate model per worker, but we still cap at MAX_WORKERS.
    num_workers = max(1, min(num_games, config.MAX_WORKERS, multiprocessing.cpu_count() - 1))

    baseline_win_rates = {}
    for opp_id, opp_name in baselines.items():
        print(f"Evaluating candidate against {opp_name}...")
        wins, losses, draws = _run_pool(
            candidate_model_path, opp_id, num_games,
            desc=f"Vs {opp_name}", num_workers=num_workers
        )
        total_decisive = wins + losses
        win_rate = wins / total_decisive if total_decisive > 0 else 0.5
        print(f"Vs {opp_name} -> Wins: {wins} | Losses: {losses} | Draws: {draws} (Win Rate: {win_rate:.2%})")
        baseline_win_rates[opp_name] = win_rate

    return baseline_win_rates


def evaluate_new_model():
    """
    High-level evaluation:
    1. Load latest candidate
    2. Load best
    3. Run games vs best
    4. Run games vs baselines
    5. Decide promotion
    """
    best_path = model_manager.get_best_model_path()

    # In distributed mode, trainer saves candidate model temporarily
    # to evaluate before officially committing it as a version.
    candidate_path = os.path.join(config.get_current_model_dir(), "checkpoint_candidate.pth.tar")

    if not os.path.exists(candidate_path):
        print(f"No candidate model found at {candidate_path} for evaluation.")
        return False, 0.0, {}

    print(f"Evaluating candidate model: {candidate_path}")

    print(f"\n================ EVALUATION VS BEST ({config.EVAL_GAMES} games) ================")
    result = evaluate_model(candidate_path, best_path, config.EVAL_GAMES)

    print(f" Wins:   {result['wins']}")
    print(f" Losses: {result['loss']}")
    print(f" Draws:  {result['draw']}")
    print(f" Win Rate: {result['win_rate']:.2%}")
    print("================================================================")

    print(f"\n================ EVALUATION VS BASELINES =======================")
    baseline_win_rates = evaluate_baselines(candidate_path, num_games=10)
    print("================================================================")

    if should_promote(result):
        print(f"Result exceeds threshold ({config.PROMOTION_THRESHOLD:.2%}). Model promoted to BEST.")
        model_manager.promote_best_model()
        return True, result['win_rate'], baseline_win_rates
    else:
        print("Model rejected.")
        return False, result['win_rate'], baseline_win_rates

if __name__ == "__main__":
    evaluate_new_model()