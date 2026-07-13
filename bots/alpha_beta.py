"""
Alpha-Beta with full optimizations:
- No deepcopy: uses execute_move / undo_move on a single shared game state
- Zobrist transposition table: exact/lower/upper bound caching with best-move ordering
"""
import random
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

    def __init__(self, name: str = "Alpha-Beta Bot", depth: int = 2):
        super().__init__(name)
        self.depth = depth

    def get_move(self, s: DotsAndBoxesGame) -> int:

        # For large boards, play randomly for the first 4 lines to save compute
        if s.SIZE >= 3:
            l = s.l
            if np.count_nonzero(l) < 4:
                # Capture any 3-line box immediately
                for box in np.ndindex(s.b.shape):
                    lines = s.get_lines_of_box(box)
                    if len([ln for ln in lines if l[ln] != 0]) == 3:
                        return [ln for ln in lines if l[ln] == 0][0]

                # Random move that doesn't create a 3-line box
                valid_moves = s.get_valid_moves()
                random.shuffle(valid_moves)
                while valid_moves:
                    move = valid_moves.pop(0)
                    ok   = True
                    for box in s.get_boxes_of_line(move):
                        lines = s.get_lines_of_box(box)
                        if len([ln for ln in lines if l[ln] != 0]) == 2:
                            ok = False
                            break
                    if ok:
                        return move

        # Fresh Zobrist TT per move search (prevents stale entries)
        zobrist      = ZobristHash(s.N_LINES)
        initial_hash = zobrist.compute_initial_hash(s.l)

        move, _ = AlphaBetaPlayer.alpha_beta_search(
            s_node       = s,
            a_latest     = None,
            depth        = self.depth,
            alpha        = -inf,
            beta         = inf,
            maximize     = True,
            zobrist      = zobrist,
            current_hash = initial_hash,
        )
        return move

    @staticmethod
    def alpha_beta_search(s_node: DotsAndBoxesGame,
                          a_latest: int,
                          depth: int,
                          alpha: float,
                          beta: float,
                          maximize: bool,
                          zobrist: ZobristHash,
                          current_hash: int) -> Tuple[int, float]:

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
        if tt_entry is not None and tt_entry['depth'] >= depth:
            flag = tt_entry['type']
            if flag == 'exact':
                return tt_entry['move'], tt_entry['value']
            if flag == 'lower' and tt_entry['value'] > alpha:
                alpha = tt_entry['value']
            if flag == 'upper' and tt_entry['value'] < beta:
                beta = tt_entry['value']
            if alpha >= beta:
                return tt_entry['move'], tt_entry['value']
            tt_best_move = tt_entry.get('move')

        # --- Move ordering: TT best move first, rest shuffled ---
        random.shuffle(valid_moves)
        if tt_best_move is not None and tt_best_move in valid_moves:
            valid_moves.remove(tt_best_move)
            valid_moves.insert(0, tt_best_move)

        # --- Search (no deepcopy — uses execute_move / undo_move) ---
        a_best = None
        v_best = -inf if maximize else inf

        for a in valid_moves:
            player_before   = s_node.current_player
            s_node.execute_move(a)
            player_switched = (s_node.current_player != player_before)

            child_hash = zobrist.update_hash(current_hash, a)

            _, v_child = AlphaBetaPlayer.alpha_beta_search(
                s_node       = s_node,
                a_latest     = a,
                depth        = depth - 1,
                alpha        = alpha,
                beta         = beta,
                maximize     = (not player_switched) if maximize else player_switched,
                zobrist      = zobrist,
                current_hash = child_hash,
            )

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