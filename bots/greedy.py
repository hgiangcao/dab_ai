import random
from agent_interface import BaseAgent


class GreedyPlayer(BaseAgent):
    """
    Improved greedy Dots and Boxes player:
    Priority:
    1. Capture boxes (box_fill_count == 3  →  placing this edge completes it)
    2. Very-safe moves (no adjacent box reaches fill == 2 after the move)
    3. Normal safe moves (no adjacent box reaches fill == 3 after the move)
    4. Minimise opponent's capture-chain giveaway

    Speed improvements over the original
    -------------------------------------
    * box_fill_count[] used everywhere instead of sum(1 for ln in lines if l[ln] != 0)
    * line_to_boxes / box_to_lines read from precomputed game attributes instead
      of calling get_boxes_of_line() / get_lines_of_box() (eliminates per-call
      division and branch work).
    * get_valid_moves() called only once per get_move() (was called 3+ times).
    * _creates_danger() inlined into the safe-move scan so execute/undo is done
      once per candidate instead of twice.
    * _creates_box() uses box_fill_count directly — no inner loop.
    * _count_giveaway() uses box_fill_count instead of _creates_box() rebuilds.
    * Local variable caching of fill, ltb, btl inside hot paths.
    """

    def __init__(self, name="Greedy Bot"):
        super().__init__(name)

    # ------------------------------------------------------------------
    # Main decision
    # ------------------------------------------------------------------

    def get_move(self, s):
        valid_moves = s.get_valid_moves()
        if not valid_moves:
            return -1

        fill = s.box_fill_count          # incremental fill counts — O(1) per box
        ltb  = s.line_to_boxes           # line_to_boxes[edge] -> tuple of box indices
        btl  = s.box_to_lines            # box_to_lines[box]   -> tuple of edge indices
        l    = s.l                       # line occupancy array

        # ----------------------------------------------------------------
        # Priority 1: capture any 3-edge box immediately
        # ----------------------------------------------------------------
        capture_moves = []
        for m in valid_moves:
            for bidx in ltb[m]:
                if fill[bidx] == 3:
                    capture_moves.append(m)
                    break
        if capture_moves:
            return random.choice(capture_moves)

        # ----------------------------------------------------------------
        # Priority 2 & 3: classify remaining moves in one pass
        # An edge is "safe"      if placing it brings NO adjacent box to fill==3
        # An edge is "very safe" if placing it brings NO adjacent box to fill>=2
        # ----------------------------------------------------------------
        very_safe_moves = []
        safe_moves = []

        for m in valid_moves:
            boxes = ltb[m]
            safe      = True
            very_safe = True
            for bidx in boxes:
                f = fill[bidx]
                if f == 2:               # placing m would make fill==3 → danger
                    safe = very_safe = False
                    break
                if f == 1:               # placing m would make fill==2 → not very safe
                    very_safe = False
            if safe:
                if very_safe:
                    very_safe_moves.append(m)
                else:
                    safe_moves.append(m)

        if very_safe_moves:
            return random.choice(very_safe_moves)
        if safe_moves:
            return random.choice(safe_moves)

        # ----------------------------------------------------------------
        # Priority 4: minimise giveaway chain
        # ----------------------------------------------------------------
        best_giveaway = 10**9
        best_moves = []
        for m in valid_moves:
            g = self._count_giveaway(s, m, fill, ltb)
            if g < best_giveaway:
                best_giveaway = g
                best_moves = [m]
            elif g == best_giveaway:
                best_moves.append(m)
        return random.choice(best_moves)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_giveaway(self, s, move, fill, ltb):
        """
        Simulate playing `move`, then greedily let opponent capture all
        available 3-edge boxes. Return the number of boxes captured.
        Uses fill[] directly instead of rebuilding counts via get_lines_of_box.
        """
        s.execute_move(move)
        moves_made = []
        try:
            while True:
                # find a capturable edge using the incremental fill counts
                cap = -1
                for m in s.get_valid_moves():
                    for bidx in ltb[m]:
                        if s.box_fill_count[bidx] == 3:
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
            s.undo_move()
        return len(moves_made)