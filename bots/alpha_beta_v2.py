"""
Alpha-Beta v2 with full optimizations:
- No deepcopy: uses execute_move / undo_move on a single shared game state
- Zobrist transposition table: exact/lower/upper bound caching with best-move ordering
- Iterative Deepening with strict time limit
- Optimized Move Ordering (TT -> Greedy -> Safe -> Sacrifice)
- Leaf evaluation using GreedyChain-style heuristic:
    * +10000 per captured box (score differential)
    * -1000 per box opponent can immediately capture
    * -100 per chain (box with >= 2 filled edges)
    * +50 bonus for fully safe positions
"""
import random
import time
from typing import Tuple
from math import inf
import sys
import os

from agent_interface import BaseAgent
from game import DotsAndBoxesGame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lookup_board import ZobristHash


# ---------------------------------------------------------------------------
# Greedy-Improve style heuristic helpers (stateless, take game state)
# ---------------------------------------------------------------------------

def _count_captured_boxes(s, player) -> int:
    """Count boxes already owned by `player`."""
    return int((s.b == player).sum())


def _creates_box(s, move) -> bool:
    """True if playing `move` would immediately complete a box."""
    for box in s.get_boxes_of_line(move):
        lines = s.get_lines_of_box(box)
        if sum(1 for ln in lines if s.l[ln] != 0) == 3:
            return True
    return False


def _opponent_immediate_gain(s) -> int:
    """Count how many boxes the current player (the opponent) can take right now."""
    gain = 0
    for m in s.get_valid_moves():
        if _creates_box(s, m):
            gain += 1
    return gain


def _count_chains(s) -> int:
    """Count empty boxes with 2 or 3 filled edges (dangerous chains)."""
    count = 0
    for r in range(s.SIZE):
        for c in range(s.SIZE):
            if s.b[r][c] == 0:
                lines = s.get_lines_of_box((r, c))
                filled = sum(1 for ln in lines if s.l[ln] != 0)
                if filled >= 2:
                    count += 1
    return count


def greedy_evaluate(s, player) -> float:
    """
    Evaluate `s` from `player`'s perspective using the GreedyChain heuristic.

    Score components (from player's view AFTER the search move was made;
    s.current_player is the side who is now to move, i.e. the *opponent*
    of whoever just moved):
      +10000 * box_score_diff   — net boxes owned
      -1000  * opponent_gain    — boxes opponent can immediately take
      -100   * chain_count      — dangerous chain count
      +50                       — if opponent has zero captures available
    """
    my_boxes  = _count_captured_boxes(s, player)
    opp_boxes = _count_captured_boxes(s, -player)
    score = (my_boxes - opp_boxes) * 10000

    # After executing the move, s.current_player is the opponent
    opp_gain = _opponent_immediate_gain(s)
    score -= opp_gain * 1000

    chains = _count_chains(s)
    score -= chains * 100

    if opp_gain == 0:
        score += 50

    return score


# ---------------------------------------------------------------------------
# Alpha-Beta Player
# ---------------------------------------------------------------------------

class AlphaBetaPlayer(BaseAgent):

    def __init__(self, name: str = "Alpha-Beta v2 Bot", time_limit: float = 2.0, endgame_threshold: int = 8):
        super().__init__(name)
        self.time_limit = time_limit
        self.endgame_threshold = endgame_threshold
        self.zobrist = None

    def get_move(self, s: DotsAndBoxesGame) -> int:
        valid_moves = s.get_valid_moves()

        if len(valid_moves) > self.endgame_threshold:
            # 1. Greedy Capture: immediately complete a box
            for r in range(s.SIZE):
                for c in range(s.SIZE):
                    if s.b[r][c] == 0:
                        lines = s.get_lines_of_box((r, c))
                        if sum(1 for ln in lines if s.l[ln] != 0) == 3:
                            return [ln for ln in lines if s.l[ln] == 0][0]

            # 2. Safe Move: doesn't give opponent a 3-edge box
            random.shuffle(valid_moves)
            for move in valid_moves:
                safe = True
                for box in s.get_boxes_of_line(move):
                    lines = s.get_lines_of_box(box)
                    if sum(1 for ln in lines if s.l[ln] != 0) == 2:
                        safe = False
                        break
                if safe:
                    return move

            # 3. Fallback
            return valid_moves[0]

        else:
            return self._iterative_deepening(s)

    def _iterative_deepening(self, s_node: DotsAndBoxesGame) -> int:
        valid_moves = s_node.get_valid_moves()
        max_depth = len(valid_moves)

        if getattr(self, 'zobrist', None) is None or getattr(self.zobrist, 'num_lines', 0) != s_node.N_LINES:
            self.zobrist = ZobristHash(s_node.N_LINES)

        if len(self.zobrist.table) > 1000000:
            self.zobrist.clear()

        initial_hash = self.zobrist.compute_initial_hash(s_node.l)

        end_time = time.time() + self.time_limit
        best_move_overall = valid_moves[0]

        for current_depth in range(1, max_depth + 1):
            try:
                move, _ = AlphaBetaPlayer.alpha_beta_search(
                    s_node       = s_node,
                    a_latest     = None,
                    depth        = current_depth,
                    alpha        = -inf,
                    beta         = inf,
                    maximize     = True,
                    zobrist      = self.zobrist,
                    current_hash = initial_hash,
                    end_time     = end_time,
                    root_player  = s_node.current_player,
                )
                if move is not None:
                    best_move_overall = move
            except TimeoutError:
                break

        return best_move_overall

    @staticmethod
    def alpha_beta_search(s_node: DotsAndBoxesGame,
                          a_latest: int,
                          depth: int,
                          alpha: float,
                          beta: float,
                          maximize: bool,
                          zobrist: ZobristHash,
                          current_hash: int,
                          end_time: float,
                          root_player: int) -> Tuple[int, float]:

        if time.time() > end_time:
            raise TimeoutError()

        valid_moves = s_node.get_valid_moves()

        # --- Base case ---
        if not valid_moves or depth == 0 or not s_node.is_running():
            if not s_node.is_running():
                # Terminal: exact score
                my  = _count_captured_boxes(s_node, root_player)
                opp = _count_captured_boxes(s_node, -root_player)
                return a_latest, (my - opp) * 100000
            # Leaf: greedy heuristic from root player's perspective
            score = greedy_evaluate(s_node, root_player)
            return a_latest, score

        # --- TT Lookup ---
        original_alpha = alpha
        original_beta  = beta
        tt_best_move   = None

        tt_key = (current_hash, s_node.b.tobytes(), int(s_node.current_player))

        tt_entry = zobrist.lookup(tt_key)
        if tt_entry is not None:
            tt_best_move = tt_entry.get('move')
            if tt_entry['depth'] >= depth:
                flag = tt_entry['type']
                if flag == 'exact':
                    return tt_entry['move'], tt_entry['value']
                if flag == 'lower' and tt_entry['value'] > alpha:
                    alpha = tt_entry['value']
                if flag == 'upper' and tt_entry['value'] < beta:
                    beta = tt_entry['value']
                if alpha >= beta:
                    return tt_entry['move'], tt_entry['value']

        # --- Optimized Move Ordering ---
        ordered_moves = []
        if tt_best_move is not None and tt_best_move in valid_moves:
            ordered_moves.append(tt_best_move)
            valid_moves.remove(tt_best_move)

        greedy_moves     = []
        safe_moves       = []
        sacrifice_moves  = []

        for move in valid_moves:
            is_greedy = False
            for box in s_node.get_boxes_of_line(move):
                lines = s_node.get_lines_of_box(box)
                if sum(1 for ln in lines if s_node.l[ln] != 0) == 3:
                    is_greedy = True
                    break

            if is_greedy:
                greedy_moves.append(move)
                continue

            # Check if move gives opponent an immediate capture
            s_node.execute_move(move)
            try:
                creates_danger = any(
                    _creates_box(s_node, m)
                    for m in s_node.get_valid_moves()
                )
            finally:
                s_node.undo_move()

            if creates_danger:
                sacrifice_moves.append(move)
            else:
                safe_moves.append(move)

        def heuristic_score(m):
            s_node.execute_move(m)
            try:
                return greedy_evaluate(s_node, root_player)
            finally:
                s_node.undo_move()

        safe_moves.sort(key=heuristic_score, reverse=True)
        sacrifice_moves.sort(key=heuristic_score, reverse=True)

        ordered_moves.extend(greedy_moves)
        ordered_moves.extend(safe_moves)
        ordered_moves.extend(sacrifice_moves)
        valid_moves = ordered_moves

        # --- Search ---
        a_best = None
        v_best = -inf if maximize else inf

        for a in valid_moves:
            player_before   = s_node.current_player
            s_node.execute_move(a)
            player_switched = (s_node.current_player != player_before)

            child_hash = zobrist.update_hash(current_hash, a)

            try:
                _, v_child = AlphaBetaPlayer.alpha_beta_search(
                    s_node       = s_node,
                    a_latest     = a,
                    depth        = depth - 1 if player_switched else depth,
                    alpha        = alpha,
                    beta         = beta,
                    maximize     = (not player_switched) if maximize else player_switched,
                    zobrist      = zobrist,
                    current_hash = child_hash,
                    end_time     = end_time,
                    root_player  = root_player,
                )
            finally:
                s_node.undo_move()

            if maximize:
                if v_child > v_best:
                    a_best = a
                    v_best = v_child
                if v_best >= beta:
                    break
                alpha = max(alpha, v_best)
            else:
                if v_child < v_best:
                    a_best = a
                    v_best = v_child
                if v_best <= alpha:
                    break
                beta = min(beta, v_best)

        # --- TT Store ---
        if   v_best <= original_alpha: tt_type = 'upper'
        elif v_best >= original_beta:  tt_type = 'lower'
        else:                          tt_type = 'exact'

        zobrist.store(tt_key, {
            'move': a_best, 'value': v_best,
            'depth': depth, 'type': tt_type,
        })

        return a_best, v_best
