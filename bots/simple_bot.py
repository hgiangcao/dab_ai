import random
from agent_interface import BaseAgent

# Precompute bitboard masks for a 5x5 grid (60 edges, 25 boxes)
BOX_MASKS = []
LINE_TO_BOXES = [[] for _ in range(60)]

for r in range(5):
    for c in range(5):
        # Calculate edge indices for 5x5 grid
        top = r * 5 + c
        bottom = (r + 1) * 5 + c
        left = 30 + r * 6 + c
        right = 30 + r * 6 + c + 1
        
        # Create a 64-bit integer mask for the box
        mask = (1 << top) | (1 << bottom) | (1 << left) | (1 << right)
        BOX_MASKS.append(mask)
        
        # Map each line to the boxes it belongs to
        LINE_TO_BOXES[top].append(mask)
        LINE_TO_BOXES[bottom].append(mask)
        LINE_TO_BOXES[left].append(mask)
        LINE_TO_BOXES[right].append(mask)

class SimpleBot(BaseAgent):
    """
    Bitboard-Optimized Simple Bot:
    1. Capture all capturable boxes using O(1) bitwise operations.
    2. Fast rollout giveaway minimization by passing 64-bit integers by value.
    """
    def __init__(self, name="SimpleBot"):
        super().__init__(name)

    def get_move(self, s):
        valid_moves = s.get_valid_moves()
        if not valid_moves:
            return -1

        # Convert object state to a single 64-bit integer
        state_bb = 0
        for i, val in enumerate(s.l):
            if val != 0:
                state_bb |= (1 << i)

        # 1. Capture all capturable boxes
        capture_moves = [m for m in valid_moves if self._is_capture(state_bb, m)]
        if capture_moves:
            return random.choice(capture_moves)

        # 2. Pick a random valid move that gives away the minimum number of boxes
        giveaways = {}
        for m in valid_moves:
            giveaways[m] = self._count_giveaway(state_bb, m, valid_moves)

        min_giveaway = min(giveaways.values())
        best_moves = [m for m in valid_moves if giveaways[m] == min_giveaway]

        return random.choice(best_moves)

    def _is_capture(self, state_bb, move):
        # O(1) bitwise check: Simulate the move, check if any adjacent box mask is fully set
        new_state = state_bb | (1 << move)
        for box_mask in LINE_TO_BOXES[move]:
            if (new_state & box_mask) == box_mask:
                return True
        return False

    def _count_giveaway(self, state_bb, move, valid_moves):
        # Pass state by value (integer modification), eliminating undo_move() entirely
        current_state = state_bb | (1 << move)
        available_moves = [m for m in valid_moves if m != move]
        boxes_given_away = 0
        
        while True:
            # Find all current captures via bitwise logic
            caps = [m for m in available_moves if self._is_capture(current_state, m)]
            if not caps:
                break
            
            # Execute the first available capture on the bitboard
            m = caps[0]
            current_state |= (1 << m)
            available_moves.remove(m)
            boxes_given_away += 1
            
        return boxes_given_away