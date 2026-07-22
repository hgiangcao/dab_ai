import random
from agent_interface import BaseAgent

class SimpleBot(BaseAgent):
    """
    Simple Bot:
    1. Capture all capturable boxes.
    2. If no boxes can be captured, pick a random valid move that minimizes the number of
       boxes given away to the opponent.
    """
    def __init__(self, name="SimpleBot"):
        super().__init__(name)

    def get_move(self, s):
        valid_moves = s.get_valid_moves()
        if not valid_moves:
            return -1

        # 1. Capture all capturable boxes
        capture_moves = [m for m in valid_moves if self._is_capture(s, m)]
        if capture_moves:
            return random.choice(capture_moves)

        # 2. Pick a random valid move that gives away the minimum number of boxes
        giveaways = {}
        for m in valid_moves:
            giveaways[m] = self._count_giveaway(s, m)

        min_giveaway = min(giveaways.values())
        best_moves = [m for m in valid_moves if giveaways[m] == min_giveaway]

        return random.choice(best_moves)

    def _is_capture(self, s, move):
        # A capture happens when drawing the line completes a box (meaning 3 lines were already drawn).
        return any(sum(1 for ln in s.get_lines_of_box(b) if s.l[ln] != 0) == 3 for b in s.get_boxes_of_line(move))

    def _count_giveaway(self, s, move):
        s.execute_move(move)
        moves_made = []
        try:
            while True:
                caps = [m for m in s.get_valid_moves() if self._is_capture(s, m)]
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
