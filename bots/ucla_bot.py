import random
from agent_interface import BaseAgent

class UCLABot(BaseAgent):
    def __init__(self, name="UCLA JS Bot"):
        super().__init__(name)

    def get_move(self, s) -> int:
        valid_moves = s.get_valid_moves()
        
        # 1. Identify all currently capturable boxes (3 lines drawn)
        captures = []
        for r in range(s.SIZE):
            for c in range(s.SIZE):
                if s.b[r][c] == 0:
                    lines = s.get_lines_of_box((r, c))
                    free = [ln for ln in lines if s.l[ln] == 0]
                    if len(free) == 1:
                        captures.append(free[0])

        # 2. Chain processing & Sacrifice (The Double-Cross)
        if captures:
            chain_moves, remaining_safe = self._trace_chain(s)
            # If taking this chain forces us to open a new one (no safe moves left),
            # and the chain is long enough, sacrifice the last 2 boxes to maintain control.
            if len(chain_moves) >= 3 and remaining_safe == 0:
                return chain_moves[-1]  # The declining move
            return captures[0]          # Otherwise, take the box

        # 3. Play safe moves (moves that do not create a 3-line box)
        safe_moves = []
        for m in valid_moves:
            is_safe = True
            for box in s.get_boxes_of_line(m):
                lines = s.get_lines_of_box(box)
                if sum(1 for ln in lines if s.l[ln] != 0) == 2:
                    is_safe = False
                    break
            if is_safe:
                safe_moves.append(m)
        
        if safe_moves:
            return random.choice(safe_moves)

        # 4. Fallback (Singletons, Doubletons, or Any)
        # Mimics the JS logic by picking the move that gives away the fewest boxes
        best_move = valid_moves[0]
        min_giveaway = float('inf')
        
        for m in valid_moves:
            giveaway = self._count_giveaway(s, m)
            if giveaway < min_giveaway:
                min_giveaway = giveaway
                best_move = m
                
        return best_move

    def _trace_chain(self, s):
        """Simulates capturing a chain to count its length and check board state."""
        moves = 0
        chain = []
        mover = s.current_player
        
        while True:
            caps = []
            for r in range(s.SIZE):
                for c in range(s.SIZE):
                    if s.b[r][c] == 0:
                        lines = s.get_lines_of_box((r, c))
                        free = [ln for ln in lines if s.l[ln] == 0]
                        if len(free) == 1:
                            caps.append(free[0])
            if not caps:
                break
            
            move = caps[0]
            chain.append(move)
            s.execute_move(move)
            moves += 1
            
            # Stop if the turn passes to the opponent
            if s.current_player != mover:
                break
                
        # Count remaining safe moves on the board after the chain is resolved
        safe_count = 0
        if s.is_running():
            for m in s.get_valid_moves():
                is_safe = True
                for box in s.get_boxes_of_line(m):
                    if sum(1 for ln in s.get_lines_of_box(box) if s.l[ln] != 0) == 2:
                        is_safe = False
                        break
                if is_safe: 
                    safe_count += 1
                
        # Undo all simulated moves to restore board state
        for _ in range(moves):
            s.undo_move()
            
        return chain, safe_count

    def _count_giveaway(self, s, move):
        """Simulates a move and counts how many boxes the opponent immediately gets."""
        s.execute_move(move)
        chain, _ = self._trace_chain(s)
        giveaway = len(chain)
        s.undo_move()
        return giveaway

import random
from agent_interface import BaseAgent

class UCLABot_v2(BaseAgent):
    def __init__(self, name="UCLABot_v2"):
        super().__init__(name)

    def get_move(self, s) -> int:
        valid = s.get_valid_moves()
        
        # 1. Take captures
        caps = [m for m in valid if self._is_capture(s, m)]
        if caps:
            chain, safe_after = self._simulate_chain(s)
            if safe_after == 0 and len(chain) == 2:
                return self._get_decline_move(s, caps[0])
            return caps[0]

        # 2. Play safe moves
        safe = [m for m in valid if not self._creates_capture(s, m)]
        if safe:
            return random.choice(safe)

        # 3. Fallback (minimize giveaway)
        return min(valid, key=lambda m: self._count_giveaway(s, m))

    def _is_capture(self, s, move):
        # FIXED: A capture happens when a box already has 3 lines drawn.
        return any(sum(1 for ln in s.get_lines_of_box(b) if s.l[ln] != 0) == 3 for b in s.get_boxes_of_line(move))

    def _creates_capture(self, s, move):
        s.execute_move(move)
        res = any(sum(1 for ln in s.get_lines_of_box((r,c)) if s.l[ln] != 0) == 3 for r in range(s.SIZE) for c in range(s.SIZE) if s.b[r][c] == 0)
        s.undo_move()
        return res

    def _simulate_chain(self, s):
        moves = []
        mover = s.current_player
        while True:
            caps = [m for m in s.get_valid_moves() if self._is_capture(s, m)]
            if not caps: break
            moves.append(caps[0])
            s.execute_move(caps[0])
            if s.current_player != mover: break
            
        safe_left = sum(1 for m in s.get_valid_moves() if not self._creates_capture(s, m)) if s.is_running() else 0
        for _ in moves: s.undo_move()
        return moves, safe_left

    def _get_decline_move(self, s, capture_move):
        for box in s.get_boxes_of_line(capture_move):
            if sum(1 for ln in s.get_lines_of_box(box) if s.l[ln] != 0) == 2:
                return next(ln for ln in s.get_lines_of_box(box) if s.l[ln] == 0 and ln != capture_move)
        return capture_move

    def _count_giveaway(self, s, move):
        s.execute_move(move)
        chain, _ = self._simulate_chain(s)
        s.undo_move()
        return len(chain)

class UCLABot_v3(BaseAgent):
    def __init__(self, name="UCLABot_v3"):
        super().__init__(name)
        self.move_queue = []

    def get_move(self, game) -> int:
        if self.move_queue:
            return self.move_queue.pop(0)
            
        self.size = game.SIZE
        self.m = self.size
        self.n = self.size
        self.N_LINES = game.N_LINES
        
        self.hedge = [[0]*self.n for _ in range(self.m + 1)]
        self.vedge = [[0]*(self.n + 1) for _ in range(self.m)]
        self.box = [[0]*self.n for _ in range(self.m)]
        
        for r in range(self.m + 1):
            for c in range(self.n):
                line_idx = r * self.n + c
                if game.l[line_idx] != 0:
                    self.hedge[r][c] = 1
                    if r > 0: self.box[r-1][c] += 1
                    if r < self.m: self.box[r][c] += 1
                    
        for c in range(self.n + 1):
            for r in range(self.m):
                line_idx = int(self.N_LINES / 2) + c * self.size + r
                if game.l[line_idx] != 0:
                    self.vedge[r][c] = 1
                    if c > 0: self.box[r][c-1] += 1
                    if c < self.n: self.box[r][c] += 1
                    
        self.player = 0
        self.zz = 0
        self.x = -1
        self.y = -1
        self.u = -1
        self.v = -1
        self.count = 0
        self.loop = False
        
        self.makemove()
        
        if self.move_queue:
            return self.move_queue.pop(0)
        else:
            valid = game.get_valid_moves()
            return valid[0]

    def sethedge(self, x, y):
        self.hedge[x][y] = 1
        if x > 0: self.box[x-1][y] += 1
        if x < self.m: self.box[x][y] += 1
        self.move_queue.append(x * self.n + y)
        self.checkh(x, y)
        self.player = 1 - self.player

    def setvedge(self, x, y):
        self.vedge[x][y] = 1
        if y > 0: self.box[x][y-1] += 1
        if y < self.n: self.box[x][y] += 1
        self.move_queue.append(int(self.N_LINES / 2) + y * self.size + x)
        self.checkv(x, y)
        self.player = 1 - self.player

    def checkh(self, x, y):
        hit = 0
        if x > 0 and self.box[x-1][y] == 4: hit = 1
        if x < self.m and self.box[x][y] == 4: hit = 1
        if hit > 0: self.player = 1 - self.player

    def checkv(self, x, y):
        hit = 0
        if y > 0 and self.box[x][y-1] == 4: hit = 1
        if y < self.n and self.box[x][y] == 4: hit = 1
        if hit > 0: self.player = 1 - self.player

    def takeedge(self, zz, x, y):
        if zz > 1: self.setvedge(x, y)
        else: self.sethedge(x, y)

    def makemove(self):
        self.takesafe3s()
        if self.sides3():
            if self.sides01():
                self.takeall3s()
                self.takeedge(self.zz, self.x, self.y)
            else:
                self.sac(self.u, self.v)
        elif self.sides01(): self.takeedge(self.zz, self.x, self.y)
        elif self.singleton(): self.takeedge(self.zz, self.x, self.y)
        elif self.doubleton(): self.takeedge(self.zz, self.x, self.y)
        else: self.makeanymove()

    def takesafe3s(self):
        for i in range(self.m):
            for j in range(self.n):
                if self.box[i][j] == 3:
                    if self.vedge[i][j] < 1:
                        if j == 0 or self.box[i][j-1] != 2: self.setvedge(i, j)
                    elif self.hedge[i][j] < 1:
                        if i == 0 or self.box[i-1][j] != 2: self.sethedge(i, j)
                    elif self.vedge[i][j+1] < 1:
                        if j == self.n - 1 or self.box[i][j+1] != 2: self.setvedge(i, j+1)
                    else:
                        if i == self.m - 1 or self.box[i+1][j] != 2: self.sethedge(i+1, j)

    def sides3(self):
        for i in range(self.m):
            for j in range(self.n):
                if self.box[i][j] == 3:
                    self.u = i
                    self.v = j
                    return True
        return False

    def takeall3s(self):
        while self.sides3():
            self.takebox(self.u, self.v)

    def sides01(self):
        if random.random() < 0.5: self.zz = 1
        else: self.zz = 2
        i = int(self.m * random.random())
        j = int(self.n * random.random())
        if self.zz == 1:
            if self.randhedge(i, j): return True
            else:
                self.zz = 2
                if self.randvedge(i, j): return True
        else:
            if self.randvedge(i, j): return True
            else:
                self.zz = 1
                if self.randhedge(i, j): return True
        return False

    def safehedge(self, i, j):
        if self.hedge[i][j] < 1:
            if i == 0:
                if self.box[i][j] < 2: return True
            elif i == self.m:
                if self.box[i-1][j] < 2: return True
            elif self.box[i][j] < 2 and self.box[i-1][j] < 2: return True
        return False

    def safevedge(self, i, j):
        if self.vedge[i][j] < 1:
            if j == 0:
                if self.box[i][j] < 2: return True
            elif j == self.n:
                if self.box[i][j-1] < 2: return True
            elif self.box[i][j] < 2 and self.box[i][j-1] < 2: return True
        return False

    def randhedge(self, i, j):
        x = i
        y = j
        while True:
            if self.safehedge(x, y):
                self.x = x
                self.y = y
                return True
            else:
                y += 1
                if y == self.n:
                    y = 0
                    x += 1
                    if x > self.m: x = 0
            if x == i and y == j: break
        return False

    def randvedge(self, i, j):
        x = i
        y = j
        while True:
            if self.safevedge(x, y):
                self.x = x
                self.y = y
                return True
            else:
                y += 1
                if y > self.n:
                    y = 0
                    x += 1
                    if x == self.m: x = 0
            if x == i and y == j: break
        return False

    def singleton(self):
        for i in range(self.m):
            for j in range(self.n):
                if self.box[i][j] == 2:
                    numb = 0
                    if self.hedge[i][j] < 1:
                        if i < 1 or self.box[i-1][j] < 2: numb += 1
                    self.zz = 2
                    if self.vedge[i][j] < 1:
                        if j < 1 or self.box[i][j-1] < 2: numb += 1
                        if numb > 1:
                            self.x, self.y = i, j
                            return True
                    if self.vedge[i][j+1] < 1:
                        if j + 1 == self.n or self.box[i][j+1] < 2: numb += 1
                        if numb > 1:
                            self.x, self.y = i, j + 1
                            return True
                    self.zz = 1
                    if self.hedge[i+1][j] < 1:
                        if i + 1 == self.m or self.box[i+1][j] < 2: numb += 1
                        if numb > 1:
                            self.x, self.y = i + 1, j
                            return True
        return False

    def doubleton(self):
        self.zz = 2
        for i in range(self.m):
            for j in range(self.n - 1):
                if self.box[i][j] == 2 and self.box[i][j+1] == 2 and self.vedge[i][j+1] < 1:
                    if self.ldub(i, j) and self.rdub(i, j+1):
                        self.x, self.y = i, j + 1
                        return True
        self.zz = 1
        for j in range(self.n):
            for i in range(self.m - 1):
                if self.box[i][j] == 2 and self.box[i+1][j] == 2 and self.hedge[i+1][j] < 1:
                    if self.udub(i, j) and self.ddub(i+1, j):
                        self.x, self.y = i + 1, j
                        return True
        return False

    def ldub(self, i, j):
        if self.vedge[i][j] < 1:
            if j < 1 or self.box[i][j-1] < 2: return True
        elif self.hedge[i][j] < 1:
            if i < 1 or self.box[i-1][j] < 2: return True
        elif i == self.m - 1 or self.box[i+1][j] < 2: return True
        return False

    def rdub(self, i, j):
        if self.vedge[i][j+1] < 1:
            if j + 1 == self.n or self.box[i][j+1] < 2: return True
        elif self.hedge[i][j] < 1:
            if i < 1 or self.box[i-1][j] < 2: return True
        elif i + 1 == self.m or self.box[i+1][j] < 2: return True
        return False

    def udub(self, i, j):
        if self.hedge[i][j] < 1:
            if i < 1 or self.box[i-1][j] < 2: return True
        elif self.vedge[i][j] < 1:
            if j < 1 or self.box[i][j-1] < 2: return True
        elif j == self.n - 1 or self.box[i][j+1] < 2: return True
        return False

    def ddub(self, i, j):
        if self.hedge[i+1][j] < 1:
            if i == self.m - 1 or self.box[i+1][j] < 2: return True
        elif self.vedge[i][j] < 1:
            if j < 1 or self.box[i][j-1] < 2: return True
        elif j == self.n - 1 or self.box[i][j+1] < 2: return True
        return False

    def sac(self, i, j):
        self.count = 0
        self.loop = False
        self.incount(0, i, j)
        if not self.loop: self.takeallbut(i, j)
        boxes_taken = sum(1 for r in range(self.m) for c in range(self.n) if self.box[r][c] == 4)
        if self.count + boxes_taken == self.m * self.n:
            self.takeall3s()
        else:
            if self.loop:
                self.count -= 2
            self.outcount(0, i, j)

    def incount(self, k, i, j):
        self.count += 1
        if k != 1 and self.vedge[i][j] < 1:
            if j > 0:
                if self.box[i][j-1] > 2:
                    self.count += 1
                    self.loop = True
                elif self.box[i][j-1] > 1: self.incount(3, i, j-1)
        elif k != 2 and self.hedge[i][j] < 1:
            if i > 0:
                if self.box[i-1][j] > 2:
                    self.count += 1
                    self.loop = True
                elif self.box[i-1][j] > 1: self.incount(4, i-1, j)
        elif k != 3 and self.vedge[i][j+1] < 1:
            if j < self.n - 1:
                if self.box[i][j+1] > 2:
                    self.count += 1
                    self.loop = True
                elif self.box[i][j+1] > 1: self.incount(1, i, j+1)
        elif k != 4 and self.hedge[i+1][j] < 1:
            if i < self.m - 1:
                if self.box[i+1][j] > 2:
                    self.count += 1
                    self.loop = True
                elif self.box[i+1][j] > 1: self.incount(2, i+1, j)

    def takeallbut(self, x, y):
        while self.sides3not(x, y):
            self.takebox(self.u, self.v)

    def sides3not(self, x, y):
        for i in range(self.m):
            for j in range(self.n):
                if self.box[i][j] == 3:
                    if i != x or j != y:
                        self.u, self.v = i, j
                        return True
        return False

    def takebox(self, i, j):
        if self.hedge[i][j] < 1: self.sethedge(i, j)
        elif self.vedge[i][j] < 1: self.setvedge(i, j)
        elif self.hedge[i+1][j] < 1: self.sethedge(i+1, j)
        else: self.setvedge(i, j+1)

    def outcount(self, k, i, j):
        if self.count > 0:
            if k != 1 and self.vedge[i][j] < 1:
                if self.count != 2: self.setvedge(i, j)
                self.count -= 1
                self.outcount(3, i, j-1)
            elif k != 2 and self.hedge[i][j] < 1:
                if self.count != 2: self.sethedge(i, j)
                self.count -= 1
                self.outcount(4, i-1, j)
            elif k != 3 and self.vedge[i][j+1] < 1:
                if self.count != 2: self.setvedge(i, j+1)
                self.count -= 1
                self.outcount(1, i, j+1)
            elif k != 4 and self.hedge[i+1][j] < 1:
                if self.count != 2: self.sethedge(i+1, j)
                self.count -= 1
                self.outcount(2, i+1, j)

    def makeanymove(self):
        x = -1
        y = -1
        found = False
        for i in range(self.m + 1):
            for j in range(self.n):
                if self.hedge[i][j] < 1:
                    x, y = i, j
                    found = True
                    break
            if found: break
            
        if not found:
            for i in range(self.m):
                for j in range(self.n + 1):
                    if self.vedge[i][j] < 1:
                        x, y = i, j
                        found = True
                        break
                if found: break
            if found:
                self.setvedge(x, y)
        else:
            self.sethedge(x, y)
            
        if not found:
            return
            
        if self.player == 0:
            self.makemove()


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