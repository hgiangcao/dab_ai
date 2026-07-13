"""
Alpha-Beta with full optimizations:
- No deepcopy: uses execute_move / undo_move on a single shared game state
- Zobrist transposition table: exact/lower/upper bound caching with best-move ordering
- Iterative Deepening with strict time limit
- Optimized Move Ordering (TT -> Greedy -> Remaining)
"""
import random
import time
from typing import Tuple
from math import inf
import numpy as np
import sys
import os

from agent_interface import BaseAgent
from game import DotsAndBoxesGame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lookup_board import ZobristHash


class AlphaBetaPlayer(BaseAgent):

    def __init__(self, name: str = "Alpha-Beta Bot", time_limit: float = 2.0, endgame_threshold: int = 12):
        super().__init__(name)
        self.time_limit = time_limit
        self.endgame_threshold = endgame_threshold

    def get_move(self, s: DotsAndBoxesGame) -> int:
        valid_moves = s.get_valid_moves()
        
        if len(valid_moves) > self.endgame_threshold:
            # 1. Greedy Capture: return an immediate 3-line box capture
            for r in range(s.SIZE):
                for c in range(s.SIZE):
                    if s.b[r][c] == 0:
                        lines = s.get_lines_of_box((r, c))
                        if sum(1 for ln in lines if s.l[ln] != 0) == 3:
                            return [ln for ln in lines if s.l[ln] == 0][0]
            
            # 2. Random Safe Move: move that doesn't create a 3-line box
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
        
        zobrist = ZobristHash(s_node.N_LINES)
        initial_hash = zobrist.compute_initial_hash(s_node.l)
        
        end_time = time.time() + self.time_limit
        best_move_overall = valid_moves[0]
        
        for current_depth in range(1, max_depth + 1):
            tt_backup = zobrist.table.copy()
            try:
                move, _ = AlphaBetaPlayer.alpha_beta_search(
                    s_node       = s_node,
                    a_latest     = None,
                    depth        = current_depth,
                    alpha        = -inf,
                    beta         = inf,
                    maximize     = True,
                    zobrist      = zobrist,
                    current_hash = initial_hash,
                    end_time     = end_time
                )
                best_move_overall = move
            except TimeoutError:
                zobrist.table = tt_backup
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
                          end_time: float) -> Tuple[int, float]:

        if time.time() > end_time:
            raise TimeoutError()

        valid_moves = s_node.get_valid_moves()

        # --- Base case ---
        if not valid_moves or depth == 0 or not s_node.is_running():
            player         = s_node.current_player if maximize else -s_node.current_player
            player_boxes   = int((s_node.b == player).sum())
            opponent_boxes = int((s_node.b == -player).sum())

            if not s_node.is_running():
                if player_boxes > opponent_boxes: return a_latest,  10000
                if player_boxes < opponent_boxes: return a_latest, -10000
                return a_latest, 0

            return a_latest, player_boxes - opponent_boxes

        # --- TT Lookup ---
        original_alpha = alpha
        original_beta  = beta
        tt_best_move   = None

        tt_entry = zobrist.lookup(current_hash)
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

        greedy_moves = []
        remaining_moves = []
        
        for move in valid_moves:
            is_greedy = False
            for box in s_node.get_boxes_of_line(move):
                lines = s_node.get_lines_of_box(box)
                if sum(1 for ln in lines if s_node.l[ln] != 0) == 3:
                    is_greedy = True
                    break
            if is_greedy:
                greedy_moves.append(move)
            else:
                remaining_moves.append(move)

        random.shuffle(remaining_moves)
        ordered_moves.extend(greedy_moves)
        ordered_moves.extend(remaining_moves)
        valid_moves = ordered_moves

        # --- Search (no deepcopy — uses execute_move / undo_move) ---
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
                    end_time     = end_time
                )
            finally:
                s_node.undo_move()   # restore state

            if maximize:
                if v_child > v_best:
                    a_best = a
                    v_best = v_child
                if v_best > beta:
                    break
                alpha = max(alpha, v_best)
            else:
                if v_child < v_best:
                    a_best = a
                    v_best = v_child
                if v_best < alpha:
                    break
                beta = min(beta, v_best)

        # --- TT Store ---
        if   v_best <= original_alpha: tt_type = 'upper'
        elif v_best >= original_beta:  tt_type = 'lower'
        else:                          tt_type = 'exact'

        zobrist.store(current_hash, {
            'move': a_best, 'value': v_best,
            'depth': depth, 'type': tt_type,
        })

        return a_best, v_best