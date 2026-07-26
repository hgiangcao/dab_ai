
# ============================================================
# MCTS ENGINE POWERED BY UCLABot_v3 ROLLOUT POLICY
# ============================================================

import math as _math
import time as _time
import random as _random

try:
    import numpy as _np
    from mcts_heuristic import MCTS, Node
except ImportError:
    import numpy as _np
    from bots.mcts_heuristic import MCTS, Node

try:
    from agent_interface import BaseAgent
except ImportError:
    from bots.agent_interface import BaseAgent

try:
    from ucla_bot import UCLABot_v3
except ImportError:
    from bots.ucla_bot import UCLABot_v3

class UCLAMCTSEngine(MCTS):
    """
    Extends the base MCTS engine with:
    - UCLABot_v3 as the rollout policy (replaces light heuristic rollout).
    - UCLA-aware expansion ordering:  capture > safe-edge > safe-center > dangerous.
    - PUCT bias derived from the UCLA move score so the tree leans toward
      UCLA-preferred moves from the first simulation.
    """

    # ------------------------------------------------------------------
    # Rollout: play out with UCLABot_v3 until terminal
    # ------------------------------------------------------------------

    def _rollout(self, s, rollout_player):
        moves_made = 0
        agent = UCLABot_v3()          # fresh agent per rollout (clean queue)

        while s.is_running():
            valid = s.get_valid_moves()
            if not valid:
                break

            move = agent.get_move(s)

            # Safety guard: if UCLABot_v3 returns something off-board
            if move is None or move not in valid:
                move = valid[0]

            s.execute_move(move)
            moves_made += 1

        p1 = int(_np.sum(s.b == 1))
        p2 = int(_np.sum(s.b == -1))
        my  = p1 if rollout_player == 1 else p2
        opp = p2 if rollout_player == 1 else p1
        val = float(my - opp) / max(s.SIZE * s.SIZE, 1)

        return val, moves_made

    # ------------------------------------------------------------------
    # Expansion ordering: capture → safe-edge → safe-center → dangerous
    # ------------------------------------------------------------------

    def _order_expansion_moves(self, s, valid_moves):
        if not valid_moves:
            return []

        captures, safe_edge, safe_center, dangerous = [], [], [], []

        for move in valid_moves:
            is_capture = False
            is_safe    = True

            for box in s.get_boxes_of_line(move):
                lines = s.get_lines_of_box(box)
                drawn = sum(1 for ln in lines if s.l[ln] != 0)
                if drawn == 3:
                    is_capture = True
                if drawn == 2:
                    is_safe = False

            if is_capture:
                captures.append(move)
            elif is_safe:
                if len(s.get_boxes_of_line(move)) == 1:
                    safe_edge.append(move)
                else:
                    safe_center.append(move)
            else:
                dangerous.append(move)

        _random.shuffle(captures)
        _random.shuffle(safe_edge)
        _random.shuffle(safe_center)
        _random.shuffle(dangerous)

        return captures + safe_edge + safe_center + dangerous

    # ------------------------------------------------------------------
    # PUCT prior: higher bias for UCLA-preferred moves
    # ------------------------------------------------------------------

    def _evaluate_bias(self, s, move):
        for box in s.get_boxes_of_line(move):
            lines = s.get_lines_of_box(box)
            drawn = sum(1 for ln in lines if s.l[ln] != 0)
            if drawn == 3:
                return 1.0          # guaranteed capture

        is_safe = True
        for box in s.get_boxes_of_line(move):
            lines = s.get_lines_of_box(box)
            drawn = sum(1 for ln in lines if s.l[ln] != 0)
            if drawn == 2:
                is_safe = False
                break

        if is_safe:
            # Edge box (touches only 1 box) is slightly safer than center
            return 0.4 if len(s.get_boxes_of_line(move)) == 1 else 0.3

        return -0.3                 # dangerous (gifts 3-line box to opponent)


# ============================================================
# UCLAMCTSBot: Hybrid agent
# ============================================================

class UCLAMCTSBot(BaseAgent):
    """
    Hybrid Dots-and-Boxes agent combining UCLABot_v3 and MCTS.

    Strategy
    --------
    1. **Forced bypass** — when 3-sided boxes are present, the capture
       sequence is deterministic and UCLA-optimal.  We call UCLABot_v3
       directly; MCTS cannot improve on it and would waste the time budget.

    2. **MCTS** — for the opening and midgame, where strategic planning
       (chain parity, sacrifice decisions) matters:
       - Expansion ordered by UCLA priority (capture > safe > dangerous).
       - Each simulation rollout is played by a fresh UCLABot_v3 instance,
         giving expert-quality playout signals instead of random moves.
       - UCB includes a per-move prior bias derived from the UCLA score.

    Parameters
    ----------
    time_limit      : seconds per turn after the first (default 0.09 s).
    first_turn_time : seconds for the very first turn (default 0.95 s).
    c_puct          : UCB exploration constant (default 1.4).
    """

    def __init__(
        self,
        name: str = "UCLAMCTSBot",
        time_limit: float = 0.09,
        first_turn_time: float = 0.95,
        c_puct: float = 1.4,
    ):
        super().__init__(name)
        self.time_limit       = time_limit
        self.first_turn_time  = first_turn_time
        self.c_puct           = c_puct
        self._first_turn      = True
        self._ucla            = UCLABot_v3()   # persistent; preserves move_queue
        # Diagnostics (last search)
        self.last_simuls = 0
        self.last_depth  = 0
        self.last_time   = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_move(self, game) -> int:
        # ── 1. Drain queued chain moves from the previous UCLA call ──────
        if self._ucla.move_queue:
            return self._ucla.move_queue.pop(0)

        # ── 2. Forced bypass: captures available → UCLA is optimal ───────
        if self._has_captures(game):
            self.last_simuls = 0
            self.last_depth  = 0
            self.last_time   = 0.0
            return self._ucla.get_move(game)

        # ── 3. Strategic position → run MCTS ─────────────────────────────
        tl = self.first_turn_time if self._first_turn else self.time_limit
        self._first_turn = False

        engine = UCLAMCTSEngine(time_limit=tl, c_puct=self.c_puct)
        best_move, depth, simuls, t = engine.search(game)

        self.last_simuls = simuls
        self.last_depth  = depth
        self.last_time   = t

        if best_move is not None:
            return best_move

        # Final fallback (should be unreachable in normal play)
        valid = game.get_valid_moves()
        return valid[0] if valid else -1

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _has_captures(self, game) -> bool:
        """Return True if any box already has exactly 3 sides drawn."""
        for r in range(game.SIZE):
            for c in range(game.SIZE):
                if game.b[r][c] == 0:
                    lines = game.get_lines_of_box((r, c))
                    if sum(1 for ln in lines if game.l[ln] != 0) == 3:
                        return True
        return False