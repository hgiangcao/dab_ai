import random

class ZobristHash:
    def __init__(self, num_lines):
        # Assign a random 64-bit integer to every possible line on the board
        self.line_hashes = [random.getrandbits(64) for _ in range(num_lines)]
        self.table = {} 

    def compute_initial_hash(self, board_lines):
        """Compute from scratch (only needed once at the start of the search)."""
        h = 0
        for idx, is_drawn in enumerate(board_lines):
            if is_drawn:
                h ^= self.line_hashes[idx]
        return h

    def update_hash(self, current_hash, move_idx):
        """O(1) incremental update. Call this when a line is drawn or undone."""
        return current_hash ^ self.line_hashes[move_idx]

    def store(self, h, data):
        self.table[h] = data

    def lookup(self, h):
        return self.table.get(h, None)