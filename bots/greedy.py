import random
from agent_interface import BaseAgent


class GreedyPlayer(BaseAgent):
    """
    Improved greedy Dots and Boxes player:
    Priority:
    1. Capture boxes
    2. Prefer "very safe" moves (adjacent boxes have < 2 lines after move)
    3. Normal safe moves (does not allow immediate capture)
    4. Minimize opponent's capture chain giveaway
    """

    def __init__(self, name="Greedy Bot"):
        super().__init__(name)

    def get_move(self, s):
        valid_moves = s.get_valid_moves()
        if not valid_moves:
            return -1

        # 1. Take immediate boxes
        capture_moves = [m for m in valid_moves if self._creates_box(s, m)]
        if capture_moves:
            return random.choice(capture_moves)

        # 2. Safe moves
        safe_moves = [m for m in valid_moves if not self._creates_danger(s, m)]
        if safe_moves:
            # Within safe moves, prioritize "very safe" moves
            # (which keep adjacent boxes at < 2 edges after the move)
            very_safe_moves = []
            for move in safe_moves:
                s.execute_move(move)
                is_very_safe = True
                try:
                    for box in s.get_boxes_of_line(move):
                        filled = sum(1 for ln in s.get_lines_of_box(box) if s.l[ln] != 0)
                        if filled >= 2:
                            is_very_safe = False
                            break
                finally:
                    s.undo_move()
                if is_very_safe:
                    very_safe_moves.append(move)

            if very_safe_moves:
                return random.choice(very_safe_moves)
            return random.choice(safe_moves)

        # 3. Minimize giveaway
        giveaways = {}
        for m in valid_moves:
            giveaways[m] = self._count_giveaway(s, m)

        min_giveaway = min(giveaways.values())
        best_moves = [m for m in valid_moves if giveaways[m] == min_giveaway]
        return random.choice(best_moves)

    def _creates_box(self, s, move):
        for box in s.get_boxes_of_line(move):
            lines = s.get_lines_of_box(box)
            filled = sum(1 for ln in lines if s.l[ln] != 0)
            if filled == 3:
                return True
        return False

    def _creates_danger(self, s, move):
        """
        After this move opponent can immediately take a box
        """
        danger = False
        s.execute_move(move)
        try:
            for m in s.get_valid_moves():
                if self._creates_box(s, m):
                    danger = True
                    break
        finally:
            s.undo_move()
        return danger

    def _count_giveaway(self, s, move):
        s.execute_move(move)
        moves_made = []
        try:
            while True:
                caps = [m for m in s.get_valid_moves() if self._creates_box(s, m)]
                if not caps:
                    break
                m = caps[0]
                s.execute_move(m)
                moves_made.append(m)
        finally:
            for m in reversed(moves_made):
                s.undo_move()
            s.undo_move()
        return len(moves_made)