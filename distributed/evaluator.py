"""
evaluator.py — Candidate Model Evaluation

Promotion: BOTH conditions must pass
PROMOTION_BOT_SCORE_RATIO = 0.80  # candidate_overall_bot_score >= this × best_overall_bot_score
MIN_WINRATE_VS_CURRENT    = 0.55  # winrate vs current checkpoint threshold

Best-model update: candidate qualifies when its overall bot score is at least
this fraction of the stored best score
BEST_UPDATE_SCORE_RATIO = 0.90
"""

import os
import sys
import numpy as np
import concurrent.futures
from tqdm import tqdm
import multiprocessing
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import config
import model_manager
from game import DotsAndBoxesGame
from model import NNetWrapper, dotdict
from mcts import MCTS

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation MCTS args — deterministic CPU inference
# ─────────────────────────────────────────────────────────────────────────────
eval_args = dotdict({
    'lr': config.LEARNING_RATE,
    'epochs': config.EPOCHS,
    'batch_size': config.BATCH_SIZE,
    'num_channels': 256,
    'num_res_blocks': 10,
    'l2_reg': 1e-4,
    'n_simulations': config.MCTS_NUM_SIMULATIONS,
    'dynamic_simulations': False,  # Force fixed simulations for evaluation
    'random_simulation': False,    # Force fixed simulations for evaluation
    'c_puct': config.MCTS_C_PUCT,
    'dirichlet_eps': 0.0,        # no noise during evaluation
    'dirichlet_alpha': config.MCTS_DIRICHLET_ALPHA,
    'time_limit': None,
    'device': 'cpu',
})

# ─────────────────────────────────────────────────────────────────────────────
# Per-process worker — plays one game
# ─────────────────────────────────────────────────────────────────────────────

def _worker_play_single(worker_args):
    """
    Isolated worker to execute a single evaluation match.
    Returns (result, avg_depth) where result ∈ {1, -1, 0} from candidate's POV.
    """
    candidate_path, opp_identifier, p1_starts = worker_args
    import copy

    game = DotsAndBoxesGame(size=5, starting_player=1, early_stopping=True)

    # ── Candidate (always uses MCTS) ──────────────────────────────────────
    cand_net = NNetWrapper(game, eval_args)
    cand_state = torch.load(candidate_path, map_location='cpu', weights_only=False)
    cand_net.nnet.load_state_dict(
        cand_state['state_dict'] if 'state_dict' in cand_state else cand_state
    )
    cand_net.nnet.eval()
    mcts_cand = MCTS(cand_net, eval_args)

    def get_eval_action(mcts_instance, g):
        """Random for first 4 moves, then greedy."""
        move_number = np.count_nonzero(g.l)
        if move_number < 4:
            import random
            valid_moves = g.get_valid_moves()
            return random.choice(valid_moves)
        
        pi = mcts_instance.play(g, temp=0.0)
        return int(np.argmax(pi))

    def agent_cand(g):
        return get_eval_action(mcts_cand, g)

    # ── Opponent ──────────────────────────────────────────────────────────
    if opp_identifier == "greedy":
        from bots.greedy import GreedyPlayer
        baseline = GreedyPlayer(name="Greedy")
        def agent_opp(g): return baseline.get_move(copy.deepcopy(g))

    elif opp_identifier == "greedy_chain":
        from bots.greedy_improve import GreedyChainPlayer
        baseline = GreedyChainPlayer(name="GreedyChain")
        def agent_opp(g): return baseline.get_move(copy.deepcopy(g))

    elif opp_identifier == "simple_bot":
        from bots.simple_bot import SimpleBot
        baseline = SimpleBot(name="SimpleBot")
        def agent_opp(g): return baseline.get_move(copy.deepcopy(g))

    elif opp_identifier == "simple_bot_v2":
        from bots.simple_bot_v2 import SimpleBotV2
        baseline = SimpleBotV2(name="SimpleBotV2")
        def agent_opp(g): return baseline.get_move(copy.deepcopy(g))

    elif opp_identifier == "ucla_bot_v3":
        from bots.ucla_bot import UCLABot_v3
        baseline = UCLABot_v3(name="UCLABot_v3")
        def agent_opp(g): return baseline.get_move(copy.deepcopy(g))

    else:
        # opp_identifier is a model checkpoint path
        if os.path.exists(opp_identifier):
            opp_net = NNetWrapper(game, eval_args)
            opp_state = torch.load(opp_identifier, map_location='cpu', weights_only=False)
            opp_net.nnet.load_state_dict(
                opp_state['state_dict'] if 'state_dict' in opp_state else opp_state
            )
            opp_net.nnet.eval()
            mcts_opp = MCTS(opp_net, eval_args)
            def agent_opp(g): return get_eval_action(mcts_opp, g)
        else:
            import random
            def agent_opp(g):
                valid = g.get_valid_moves()
                return random.choice(valid) if valid else None

    # ── Play the game ─────────────────────────────────────────────────────
    players = {1: agent_cand, -1: agent_opp} if p1_starts else {1: agent_opp, -1: agent_cand}
    depths = []

    while game.is_running():
        cur_player = game.current_player
        move_number = np.count_nonzero(game.l)

        if move_number < 4:
            import random
            valid_moves = game.get_valid_moves()
            action = random.choice(valid_moves)
        else:
            action = players[cur_player](game)
            # Track candidate MCTS depth
            if (p1_starts and cur_player == 1) or (not p1_starts and cur_player == -1):
                if mcts_cand.max_depth_reached >= 0:
                    depths.append(mcts_cand.max_depth_reached)

        game.execute_move(action)

    result = game.result if p1_starts else -game.result
    avg_depth = sum(depths) / len(depths) if depths else 0
    return result, avg_depth


# ─────────────────────────────────────────────────────────────────────────────
# Arena evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_arena(candidate_path: str, opp_identifier: str, num_games: int) -> dict:
    """
    Run num_games between candidate and opp_identifier.
    Returns {"wins", "losses", "draws", "win_rate", "avg_depth"}.
    """
    half = num_games // 2
    worker_args_list = [
        (candidate_path, opp_identifier, idx < half)
        for idx in range(num_games)
    ]

    mp_context = multiprocessing.get_context('spawn')
    num_workers = max(1, min(config.MAX_WORKERS, multiprocessing.cpu_count() - 1))

    wins, losses, draws = 0, 0, 0
    total_depth = 0.0
    completed = 0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, mp_context=mp_context
    ) as executor:
        futures = [executor.submit(_worker_play_single, arg) for arg in worker_args_list]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=num_games,
            desc=f"Arena [{opp_identifier if len(opp_identifier) < 30 else 'model'}]",
        ):
            try:
                res, depth = future.result()
                total_depth += depth
                completed += 1
                if res == 1:
                    wins += 1
                elif res == -1:
                    losses += 1
                else:
                    draws += 1
            except Exception as e:
                print(f"Match failed: {e}")

    decisive = wins + losses
    win_rate = wins / decisive if decisive > 0 else 0.5
    avg_depth = total_depth / completed if completed > 0 else 0.0
    return {"wins": wins, "losses": losses, "draws": draws,
            "win_rate": win_rate, "avg_depth": avg_depth}


# ─────────────────────────────────────────────────────────────────────────────
# Public evaluation API
# ─────────────────────────────────────────────────────────────────────────────

# External bots used for the absolute-strength benchmark
EVAL_BOTS = ["greedy", "simple_bot_v2", "ucla_bot_v3"]


def evaluate_new_model(iteration=None):
    """
    Full candidate evaluation with dual-condition promotion.

    Promotion requires BOTH:
      1. overall_bot_score >= config.PROMOTION_BOT_SCORE_RATIO × best_overall_bot_score
         (candidate scores at least 80% as well as best model on the bot benchmark)
      2. winrate_vs_current >= config.MIN_WINRATE_VS_CURRENT
         (candidate shows local improvement vs the self checkpoint)

    Best-model update (independent):
      candidate_overall_bot_score >= config.BEST_UPDATE_SCORE_RATIO × stored_best_score

    Returns:
        promoted           (bool)   — candidate became current/self
        updated_best       (bool)   — best.pth.tar was updated
        bot_win_rates      (dict)   — {bot_name: win_rate}
        winrate_vs_current (float)
        avg_depth          (float)
    """
    candidate_path = model_manager.get_candidate_path()
    current_path   = model_manager.get_current_model_path()

    if not os.path.exists(candidate_path):
        print(f"[Eval] No candidate found at {candidate_path}.")
        return False, False, {}, 0.0, 0.0

    # ── 1. Screen against external bots (10 games each) ──────────────────
    print("\n" + "="*60)
    print(f"EVALUATION: {config.EVAL_GAMES_VS_BOTS} games × {len(EVAL_BOTS)} bots  +  "
          f"{config.EVAL_GAMES_VS_CURRENT} games vs current")
    print("="*60)

    bot_win_rates = {}

    for bot_name in EVAL_BOTS:
        print(f"\n  ── vs {bot_name} ({config.EVAL_GAMES_VS_BOTS} games) ──")
        result = _run_arena(candidate_path, bot_name, config.EVAL_GAMES_VS_BOTS)
        wr = result["win_rate"]
        bot_win_rates[bot_name] = wr
        print(f"  {bot_name}: {result['wins']}W / {result['losses']}L / {result['draws']}D  "
              f"WR={wr:.2%}")

    overall_bot_score = sum(bot_win_rates.values()) / len(bot_win_rates) if bot_win_rates else 0.0

    # Retrieve stored best score for the promotion gate
    stored_best_score = model_manager.get_best_overall_score()
    # If no best score has been recorded yet (first ever run), treat it as 0
    # so any positive score passes.
    bot_score_threshold = config.PROMOTION_BOT_SCORE_RATIO * stored_best_score

    print(f"\n  Overall bot score: {overall_bot_score:.4f}  |  "
          f"Required: {config.PROMOTION_BOT_SCORE_RATIO:.0%} × best {stored_best_score:.4f} "
          f"= {bot_score_threshold:.4f}")

    # ── 2. Screen vs current checkpoint (50 games) ───────────────────────
    print(f"\n  ── vs current/self ({config.EVAL_GAMES_VS_CURRENT} games) ──")
    if os.path.exists(current_path):
        vs_current_result = _run_arena(candidate_path, current_path, config.EVAL_GAMES_VS_CURRENT)
        winrate_vs_current = vs_current_result["win_rate"]
        avg_depth = vs_current_result["avg_depth"]
        print(f"  vs current: {vs_current_result['wins']}W / {vs_current_result['losses']}L / "
              f"{vs_current_result['draws']}D  WR={winrate_vs_current:.2%}")
    else:
        # No current model yet (first ever promotion) — auto-pass this check
        print("  No current checkpoint found — skipping vs-current check (first promotion).")
        winrate_vs_current = 1.0
        avg_depth = 0.0

    print(f"  Required vs-current WR: {config.MIN_WINRATE_VS_CURRENT:.2%}")

    # ── 3. Promotion decision ─────────────────────────────────────────────
    bot_condition     = overall_bot_score >= bot_score_threshold
    current_condition = winrate_vs_current >= config.MIN_WINRATE_VS_CURRENT
    promoted = bot_condition and current_condition

    print("\n" + "─"*60)
    print(f"  Bot score condition [{overall_bot_score:.4f} >= {config.PROMOTION_BOT_SCORE_RATIO:.0%} × {stored_best_score:.4f} = {bot_score_threshold:.4f}]: "
          f"{'✓ PASS' if bot_condition else '✗ FAIL'}")
    print(f"  Current condition   [{winrate_vs_current:.2%} >= {config.MIN_WINRATE_VS_CURRENT:.2%}]: "
          f"{'✓ PASS' if current_condition else '✗ FAIL'}")
    print(f"  Promotion decision: {'>>> PROMOTED <<<' if promoted else 'REJECTED'}")
    print("─"*60)

    if promoted:
        model_manager.promote_to_current()
        print("[Eval] Candidate promoted → checkpoint_current (self)")

    # ── 4. Best-model update (independent) ───────────────────────────────
    best_threshold = config.BEST_UPDATE_SCORE_RATIO * stored_best_score
    stored_best_min_bot_wr = model_manager.get_best_min_bot_winrate()
    candidate_min_wr = min(bot_win_rates.values()) if bot_win_rates else 0.0

    updated_best = (overall_bot_score >= best_threshold) 

    print(f"\n  Best update check: candidate score {overall_bot_score:.4f} >= "
          f"{config.BEST_UPDATE_SCORE_RATIO:.0%} × stored best {stored_best_score:.4f} "
          f"= {best_threshold:.4f}")
    
    if updated_best:
        model_manager.update_best()
        
        if overall_bot_score > stored_best_score:
            model_manager.set_best_overall_score(overall_bot_score)
            model_manager.set_best_min_bot_winrate(candidate_min_wr)
            
        print(f"[Eval] best.pth.tar updated. New best score: {overall_bot_score:.4f}, min bot WR: {candidate_min_wr:.4f}")

    print("="*60 + "\n")
    return promoted, updated_best, bot_win_rates, winrate_vs_current, avg_depth


# ─────────────────────────────────────────────────────────────────────────────
# Legacy helpers (kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(candidate_model_path, best_model_path, num_games):
    """Legacy: evaluate candidate vs a specific model path."""
    result = _run_arena(candidate_model_path, best_model_path, num_games)
    return result


def evaluate_baselines(candidate_model_path, num_games=10):
    """Legacy: evaluate candidate vs all standard bots."""
    rates = {}
    for bot in EVAL_BOTS:
        r = _run_arena(candidate_model_path, bot, num_games)
        rates[bot] = r["win_rate"]
    return rates


def should_promote(result):
    """Legacy: single-threshold promotion check."""
    return result.get("win_rate", 0.0) >= config.PROMOTION_THRESHOLD


if __name__ == "__main__":
    evaluate_new_model()