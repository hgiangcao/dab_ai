import random
from agent_interface import BaseAgent


class GreedyPlayer(BaseAgent):
    """
    Simple greedy Dots and Boxes player:
    Priority:
    1. Capture boxes
    2. Avoid giving boxes
    3. Random safe move
    """

    def __init__(self, name="Greedy Bot"):
        super().__init__(name)

    def get_move(self, s):

        valid_moves = s.get_valid_moves()

        # 1. Take immediate boxes
        capture_moves = []

        for move in valid_moves:
            if self._creates_box(s, move):
                capture_moves.append(move)

        if capture_moves:
            return random.choice(capture_moves)


        # 2. Safe moves
        safe_moves = []

        for move in valid_moves:
            if not self._creates_danger(s, move):
                safe_moves.append(move)

        if safe_moves:
            return random.choice(safe_moves)


        # 3. Forced move
        return random.choice(valid_moves)


    def _creates_box(self, s, move):

        for box in s.get_boxes_of_line(move):
            lines = s.get_lines_of_box(box)

            filled = sum(
                1 for ln in lines
                if s.l[ln] != 0
            )

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