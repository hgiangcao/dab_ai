import random
from agent_interface import BaseAgent

class GreedyChainPlayer(BaseAgent):
    """
    Improved greedy Dots and Boxes player that considers chain penalties.

    Speed improvements:
    * box_fill_count[] used for counting captured boxes, chains, and evaluating danger.
    * line_to_boxes used instead of get_boxes_of_line().
    * Eliminated Numpy flatten operations and repetitive list creations.
    """

    def __init__(self, name="Greedy Chain Bot"):
        super().__init__(name)

    def get_move(self, s):
        moves = s.get_valid_moves()
        if not moves:
            return -1

        best_score = -10**9
        best_moves = []

        fill = s.box_fill_count
        ltb = s.line_to_boxes

        for move in moves:
            score = self.evaluate_move(s, move, fill, ltb)
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves)

    def evaluate_move(self, s, move, fill, ltb):
        score = 0
        before = self.count_boxes(fill)
        player_before = s.current_player

        s.execute_move(move)
        try:
            after = self.count_boxes(fill)
            captured = after - before

            # 1. Reward capture
            score += captured * 10000

            # 2. Simulate full opponent capture chain
            opponent_gain = 0
            if s.current_player != player_before:
                opponent_gain = self._count_giveaway(s, fill, ltb)
            score -= opponent_gain * 1000

            # 3. Chain penalty
            chain_count = self.count_chains(fill)
            score -= chain_count * 100

            # 4. Safe position
            if opponent_gain == 0:
                score += 50

            # 5. Very safe position (no 2-edge boxes created)
            has_2_edge_box = False
            for bidx in ltb[move]:
                if fill[bidx] == 2:
                    has_2_edge_box = True
                    break
            if opponent_gain == 0 and not has_2_edge_box:
                score += 200

        finally:
            s.undo_move()

        return score

    def count_boxes(self, fill):
        return sum(1 for f in fill if f == 4)

    def count_chains(self, fill):
        """
        Count boxes with 2 or 3 edges.
        More 2 or 3-edge boxes means dangerous chains.
        """
        return sum(1 for f in fill if f == 2 or f == 3)

    def _count_giveaway(self, s, fill, ltb):
        moves_made = []
        try:
            while True:
                cap = -1
                for m in s.get_valid_moves():
                    for bidx in ltb[m]:
                        if fill[bidx] == 3:
                            cap = m
                            break
                    if cap != -1:
                        break
                
                if cap == -1:
                    break
                s.execute_move(cap)
                moves_made.append(cap)
        finally:
            for m in reversed(moves_made):
                s.undo_move()
        return len(moves_made)