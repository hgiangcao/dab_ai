import random
from agent_interface import BaseAgent

# Precompute bitboard masks for a 5x5 grid (60 edges, 25 boxes)
BOX_MASKS = []
LINE_TO_BOXES = [[] for _ in range(60)]

for r in range(5):
    for c in range(5):
        top = r * 5 + c
        bottom = (r + 1) * 5 + c
        left = 30 + r * 6 + c
        right = 30 + r * 6 + c + 1
        
        mask = (1 << top) | (1 << bottom) | (1 << left) | (1 << right)
        BOX_MASKS.append(mask)
        
        LINE_TO_BOXES[top].append(mask)
        LINE_TO_BOXES[bottom].append(mask)
        LINE_TO_BOXES[left].append(mask)
        LINE_TO_BOXES[right].append(mask)

class UCLABot_v6(BaseAgent):
    def __init__(self, name="UCLABot_v6"):
        super().__init__(name)

    def get_move(self, s) -> int:
        valid_moves = s.get_valid_moves()
        if not valid_moves:
            return -1

        state_bb = 0
        for i, val in enumerate(s.l):
            if val != 0:
                state_bb |= (1 << i)

        for move in valid_moves:
            if self._is_capture(state_bb, move):
                return move

        safe_moves = [m for m in valid_moves if self._is_safe(state_bb, m)]
        
        if safe_moves:
            return random.choice(safe_moves)

        return random.choice(valid_moves)

    def _is_capture(self, state_bb, move):
        new_state = state_bb | (1 << move)
        for box_mask in LINE_TO_BOXES[move]:
            if (new_state & box_mask) == box_mask:
                return True
        return False

    def _is_safe(self, state_bb, move):
        new_state = state_bb | (1 << move)
        for box_mask in LINE_TO_BOXES[move]:
            overlap = new_state & box_mask
            if overlap.bit_count() == 3:
                return False
        return True