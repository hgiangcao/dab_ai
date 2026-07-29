from agent_interface import BaseAgent
import random# Sentinel value for "no second box" (border edges touch only 1 box)
_NONE = -1


def _build_edge_arrays(game_state):
    """
    Build two flat int lists EDGE_BOX1 and EDGE_BOX2 from the game's
    line_to_boxes mapping.  Border edges that touch only one box get
    EDGE_BOX2[e] = _NONE so the inner loop can skip them with a single
    integer compare instead of iterating a variable-length tuple.
    """
    ltb = game_state.line_to_boxes
    n = len(ltb)
    b1 = [_NONE] * n
    b2 = [_NONE] * n
    for e, boxes in enumerate(ltb):
        b1[e] = boxes[0]
        if len(boxes) == 2:
            b2[e] = boxes[1]
    return b1, b2


class FillBot(BaseAgent):
    """
    Greedy heuristic bot designed for backward training / self-play fill.

    Priority 1: Complete a box if possible (score a point).
    Priority 2: Play a safe edge (does not give the opponent a capturable box).
    Priority 3: Play the first available edge (forced sacrifice).

    Micro-optimisations applied
    ---------------------------
    * Unused imports removed (numpy, sys, os).
    * Inner loop unrolled: two explicit box checks replace tuple iteration.
    * `max_fill` replaced by a single `safe` bool (one less assign per iter).
    * `get_valid_moves()` replaced by direct iteration over `game_state.l` to
      avoid allocating a new list every call.
    * `fill`, `edge_box1`, `edge_box2`, `_NONE_` all cached as locals so CPython
      resolves them without attribute / global lookups inside the hot loop.
    * Randomness is retained for safe and sacrifice edges to ensure state diversity.
    """

    def __init__(self, name: str = "FillBot"):
        super().__init__(name)
        # Lazily built per board size; keyed by N_LINES
        self._edge_box1 = None
        self._edge_box2 = None
        self._n_lines   = -1

    def get_move(self, game_state) -> int:
        # ---- ensure unrolled edge arrays are built for this board ----
        n_lines = game_state.N_LINES
        if n_lines != self._n_lines:
            self._edge_box1, self._edge_box2 = _build_edge_arrays(game_state)
            self._n_lines = n_lines

        # ---- cache everything into locals (5–15 % speedup in CPython) ----
        l         = game_state.l          # line occupancy array
        fill      = game_state.box_fill_count
        edge_box1 = self._edge_box1
        edge_box2 = self._edge_box2
        NONE      = _NONE

        safe_edges = []
        sac_edges = []

        for edge in range(n_lines):
            if l[edge] != 0:
                continue                  # edge already taken

            sac_edges.append(edge)

            # ---- unrolled 2-box check (avoids inner loop + tuple overhead) ----
            safe = True

            b1 = edge_box1[edge]
            if b1 != NONE:
                f = fill[b1]
                if f == 3:
                    return edge           # PRIORITY 1: immediate capture
                if f >= 2:
                    safe = False

            b2 = edge_box2[edge]
            if b2 != NONE:
                f = fill[b2]
                if f == 3:
                    return edge           # PRIORITY 1: immediate capture
                if f >= 2:
                    safe = False

            # ---- PRIORITY 2: collect safe edges ----
            if safe:
                safe_edges.append(edge)

        if safe_edges:
            return random.choice(safe_edges)

        # PRIORITY 3: forced sacrifice
        return random.choice(sac_edges) if sac_edges else None
