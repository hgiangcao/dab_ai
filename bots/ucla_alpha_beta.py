import sys
import os
import time
from math import inf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_interface import BaseAgent
from bots.ucla_bot_heuristic import UCLAHeuristicEvaluator
from lookup_board import ZobristHash

class UCLAAlphaBeta(BaseAgent):
    def __init__(self, name="UCLA AlphaBeta"):
        super().__init__(name)
        self.evaluator = UCLAHeuristicEvaluator()
        self.zobrist = None

    def get_move(self, game):
        if self.zobrist is None:
            self.zobrist = ZobristHash(game.N_LINES)

        depth = self.choose_depth(game)
        
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return -1

        current_hash = self.zobrist.compute_initial_hash(game.l)
        root_player = game.current_player
        
        # We will maximize for root_player
        move, score = self.alphabeta(
            game=game,
            a_latest=None,
            depth=depth,
            alpha=-inf,
            beta=inf,
            maximize=True,
            current_hash=current_hash,
            root_player=root_player
        )
        
        if move is None:
            move = valid_moves[0]
            
        return move

    def choose_depth(self, game):
        remaining = len(game.get_valid_moves())
        total = game.N_LINES
        ratio = remaining / total

        chains = self.evaluator.find_chains(game)

        if remaining <= 15:
            print("depth: 50")  
            return 50 # Solve endgame
        if chains:
            print("depth: 4")
            return 4 # Tactical chain phase
        if ratio > 0.7:
            print("depth: 3")
            return 3  # Opening
        if ratio > 0.35:
            print("depth: 6")
            return 6  # Middle game
        print("depth: 8")
        return 8      # Default late middle

    def alphabeta(self, game, a_latest, depth, alpha, beta, maximize, current_hash, root_player):
        my = int((game.b == root_player).sum())
        opp = int((game.b == -root_player).sum())
        if my * 2 > game.N_BOXES or opp * 2 > game.N_BOXES:
            return a_latest, (my - opp) * 100000

        valid_moves = game.get_valid_moves()

        if not valid_moves or depth == 0 or not game.is_running():
            if not game.is_running():
                return a_latest, (my - opp) * 100000

            # Quiescence search
            if depth == 0 and self.evaluator.find_chains(game):
                return self.quiescence(game, a_latest, alpha, beta, maximize, root_player)
                
            score = self.evaluator.evaluate(game, player=root_player)
            return a_latest, score

        # TT Lookup
        tt_key = (current_hash, game.b.tobytes(), int(game.current_player))
        tt_entry = self.zobrist.lookup(tt_key)
        
        tt_best_move = None
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

        # Move ordering
        ordered_moves = self.evaluator.order_moves(game, valid_moves)
        if tt_best_move is not None and tt_best_move in ordered_moves:
            ordered_moves.remove(tt_best_move)
            ordered_moves.insert(0, tt_best_move)

        best_move = None
        best_val = -inf if maximize else inf
        original_alpha = alpha
        original_beta = beta

        for a in ordered_moves:
            player_before = game.current_player
            game.execute_move(a)
            player_switched = (game.current_player != player_before)
            child_hash = self.zobrist.update_hash(current_hash, a)

            _, child_val = self.alphabeta(
                game,
                a,
                depth if not player_switched else depth - 1, # KEEP depth if player doesn't switch
                alpha,
                beta,
                (not player_switched) if maximize else player_switched,
                child_hash,
                root_player
            )
            
            game.undo_move()

            if maximize:
                if child_val > best_val:
                    best_val = child_val
                    best_move = a
                alpha = max(alpha, best_val)
            else:
                if child_val < best_val:
                    best_val = child_val
                    best_move = a
                beta = min(beta, best_val)

            if alpha >= beta:
                break

        # TT Store
        if best_val <= original_alpha:
            flag = 'upper'
        elif best_val >= original_beta:
            flag = 'lower'
        else:
            flag = 'exact'
            
        self.zobrist.store(tt_key, {'value': best_val, 'depth': depth, 'type': flag, 'move': best_move})

        return best_move, best_val

    def quiescence(self, game, a_latest, alpha, beta, maximize, root_player):
        my = int((game.b == root_player).sum())
        opp = int((game.b == -root_player).sum())
        if my * 2 > game.N_BOXES or opp * 2 > game.N_BOXES:
            return a_latest, (my - opp) * 100000

        valid_moves = game.get_valid_moves()
        
        # Only search forced moves: moves that capture a box
        capture_moves = [m for m in valid_moves if self._creates_box(game, m)]
        
        if not capture_moves or not game.is_running():
            if not game.is_running():
                return a_latest, (my - opp) * 100000
            score = self.evaluator.evaluate(game, player=root_player)
            return a_latest, score

        best_move = None
        best_val = -inf if maximize else inf

        for a in capture_moves:
            player_before = game.current_player
            game.execute_move(a)
            player_switched = (game.current_player != player_before)

            _, child_val = self.quiescence(
                game,
                a,
                alpha,
                beta,
                (not player_switched) if maximize else player_switched,
                root_player
            )
            
            game.undo_move()

            if maximize:
                if child_val > best_val:
                    best_val = child_val
                    best_move = a
                alpha = max(alpha, best_val)
            else:
                if child_val < best_val:
                    best_val = child_val
                    best_move = a
                beta = min(beta, best_val)

            if alpha >= beta:
                break

        return best_move, best_val

    def _creates_box(self, game, move):
        for box in game.get_boxes_of_line(move):
            lines = game.get_lines_of_box(box)
            if sum(1 for ln in lines if game.l[ln] != 0) == 3:
                return True
        return False
