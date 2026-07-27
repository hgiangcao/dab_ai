import random
from agent_interface import BaseAgent
"""
Improved version of UCLABot_v3.

Kept from v3 (already solid, well-tested logic):
  - safe-move detection (safehedge/safevedge)
  - forced-capture handling (takesafe3s / takeall3s / takebox)
  - double-cross sacrifice logic for chains and loops (sac / incount / outcount)
  - singleton / doubleton special-case safe moves

New in v4:
  1. Chain/loop decomposition (`get_regions`) - boxes with exactly 2 sides
     taken are grouped into connected "chain" or "loop" regions, the same
     structure Berlekamp's dots-and-boxes theory reasons about.
  2. Shortest-chain-first opening. When v3 is forced to open a new region
     (`makeanymove`), it used to just scan for the first free edge in
     row-major order. v4 instead finds the SHORTEST available chain (or
     loop, if no chain exists) and opens it from a true end. This is the
     single most reliable low-level endgame improvement: it minimizes what
     you hand the opponent when you have no safe move left.
  3. Long Chain Rule parity steering. v4 tracks whether it moved first in
     the game (`is_first_player`) and, whenever there is more than one safe
     move available, simulates each candidate and prefers the one that
     leaves the number of long chains (length >= 3) at the parity that
     favors this bot having "control" in the endgame (Berlekamp's long
     chain rule). This is a heuristic best-effort implementation - it does
     not model branching/joint structures exactly (full dots-and-boxes
     theory is significantly more involved), but it steers the midgame in
     the right direction far more often than picking safe moves at random.

The public interface (`get_move`) is unchanged, so this is a drop-in
replacement for UCLABot_v3.
"""


class UCLABot_v5(BaseAgent):
    def __init__(self, name="UCLABot_v5"):
        super().__init__(name)
        self.move_queue = []
        # Long Chain Rule bookkeeping - set on the first call to get_move().
        self.is_first_player = None

    # ------------------------------------------------------------------
    # Top-level entry point (unchanged interface from v3)
    # ------------------------------------------------------------------
    def get_move(self, game) -> int:
        import numpy as np

        total_lines = int(np.count_nonzero(game.l))

        if total_lines == 0:
            # Fresh game - this bot is being asked to make the very first
            # move, so it is the first player.
            self.move_queue.clear()
            self.is_first_player = True
        elif self.is_first_player is None:
            # First time we're seeing this game and lines already exist -
            # someone else moved before us, so we're the second player.
            self.is_first_player = False

        while self.move_queue:
            mv = self.move_queue[0]
            if game.l[mv] != 0:
                self.move_queue.clear()
                break
            return self.move_queue.pop(0)

        self.size = game.SIZE
        self.m = self.size
        self.n = self.size
        self.N_LINES = game.N_LINES

        self.hedge = [[0] * self.n for _ in range(self.m + 1)]
        self.vedge = [[0] * (self.n + 1) for _ in range(self.m)]
        self.box = [[0] * self.n for _ in range(self.m)]

        for r in range(self.m + 1):
            for c in range(self.n):
                line_idx = r * self.n + c
                if game.l[line_idx] != 0:
                    self.hedge[r][c] = 1
                    if r > 0:
                        self.box[r - 1][c] += 1
                    if r < self.m:
                        self.box[r][c] += 1

        for c in range(self.n + 1):
            for r in range(self.m):
                line_idx = int(self.N_LINES / 2) + c * self.size + r
                if game.l[line_idx] != 0:
                    self.vedge[r][c] = 1
                    if c > 0:
                        self.box[r][c - 1] += 1
                    if c < self.n:
                        self.box[r][c] += 1

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

    # ------------------------------------------------------------------
    # Low-level edge/box bookkeeping (unchanged from v3)
    # ------------------------------------------------------------------
    def sethedge(self, x, y):
        self.hedge[x][y] = 1
        if x > 0:
            self.box[x - 1][y] += 1
        if x < self.m:
            self.box[x][y] += 1
        self.move_queue.append(x * self.n + y)
        self.checkh(x, y)
        self.player = 1 - self.player

    def setvedge(self, x, y):
        self.vedge[x][y] = 1
        if y > 0:
            self.box[x][y - 1] += 1
        if y < self.n:
            self.box[x][y] += 1
        self.move_queue.append(int(self.N_LINES / 2) + y * self.size + x)
        self.checkv(x, y)
        self.player = 1 - self.player

    def checkh(self, x, y):
        hit = 0
        if x > 0 and self.box[x - 1][y] == 4:
            hit = 1
        if x < self.m and self.box[x][y] == 4:
            hit = 1
        if hit > 0:
            self.player = 1 - self.player

    def checkv(self, x, y):
        hit = 0
        if y > 0 and self.box[x][y - 1] == 4:
            hit = 1
        if y < self.n and self.box[x][y] == 4:
            hit = 1
        if hit > 0:
            self.player = 1 - self.player

    def takeedge(self, zz, x, y):
        if zz > 1:
            self.setvedge(x, y)
        else:
            self.sethedge(x, y)

    # ------------------------------------------------------------------
    # Top-level decision logic
    # ------------------------------------------------------------------
    def makemove(self):
        self.takesafe3s()
        if self.sides3():
            best = self.get_best_safe_move()
            if best is not None:
                self.takeall3s()
                zz, x, y = best
                self.takeedge(zz, x, y)
            else:
                self.sac(self.u, self.v)
        else:
            best = self.get_best_safe_move()
            if best is not None:
                zz, x, y = best
                self.takeedge(zz, x, y)
            elif self.singleton():
                self.takeedge(self.zz, self.x, self.y)
            elif self.doubleton():
                self.takeedge(self.zz, self.x, self.y)
            else:
                self.makeanymove()

    def takesafe3s(self):
        for i in range(self.m):
            for j in range(self.n):
                if self.box[i][j] == 3:
                    if self.vedge[i][j] < 1:
                        if j == 0 or self.box[i][j - 1] != 2:
                            self.setvedge(i, j)
                    elif self.hedge[i][j] < 1:
                        if i == 0 or self.box[i - 1][j] != 2:
                            self.sethedge(i, j)
                    elif self.vedge[i][j + 1] < 1:
                        if j == self.n - 1 or self.box[i][j + 1] != 2:
                            self.setvedge(i, j + 1)
                    else:
                        if i == self.m - 1 or self.box[i + 1][j] != 2:
                            self.sethedge(i + 1, j)

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

    # ------------------------------------------------------------------
    # Safe move detection (unchanged primitives, new enumeration on top)
    # ------------------------------------------------------------------
    def safehedge(self, i, j):
        if self.hedge[i][j] < 1:
            if i == 0:
                if self.box[i][j] < 2:
                    return True
            elif i == self.m:
                if self.box[i - 1][j] < 2:
                    return True
            elif self.box[i][j] < 2 and self.box[i - 1][j] < 2:
                return True
        return False

    def safevedge(self, i, j):
        if self.vedge[i][j] < 1:
            if j == 0:
                if self.box[i][j] < 2:
                    return True
            elif j == self.n:
                if self.box[i][j - 1] < 2:
                    return True
            elif self.box[i][j] < 2 and self.box[i][j - 1] < 2:
                return True
        return False

    def get_all_safe_moves(self):
        """Enumerate every currently-safe edge as (zz, x, y) tuples.

        zz follows the v3 convention used by takeedge(): zz == 1 means a
        horizontal edge (hedge), zz == 2 means a vertical edge (vedge).
        """
        moves = []
        for i in range(self.m + 1):
            for j in range(self.n):
                if self.safehedge(i, j):
                    moves.append((1, i, j))
        for i in range(self.m):
            for j in range(self.n + 1):
                if self.safevedge(i, j):
                    moves.append((2, i, j))
        return moves

    def get_best_safe_move(self):
        """Return the safe move to play, applying Long Chain Rule parity
        steering when more than one option exists. Returns None if no safe
        move is currently available."""
        moves = self.get_all_safe_moves()
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
        return self.choose_parity_move(moves)

    # ------------------------------------------------------------------
    # Long Chain Rule parity steering
    # ------------------------------------------------------------------
    def choose_parity_move(self, moves):
        # Heuristic target: if we moved first, we want to hand the
        # opponent an EVEN number of long chains to open (parity 0);
        # if we moved second, we want an ODD number (parity 1). This is a
        # simplified approximation of Berlekamp's long chain rule and does
        # not account for loops or joint double-cross value precisely.
        desired_parity = 0 if self.is_first_player else 1

        best_move = moves[0]
        best_score = None
        for (zz, x, y) in moves:
            long_chain_count = self._simulate_long_chain_count(zz, x, y)
            parity = long_chain_count % 2
            score = 0 if parity == desired_parity else 1
            if best_score is None or score < best_score:
                best_score = score
                best_move = (zz, x, y)
                if best_score == 0:
                    break
        return best_move

    def _simulate_long_chain_count(self, zz, x, y):
        hedge = [row[:] for row in self.hedge]
        vedge = [row[:] for row in self.vedge]
        box = [row[:] for row in self.box]

        if zz > 1:
            vedge[x][y] = 1
            if y > 0:
                box[x][y - 1] += 1
            if y < self.n:
                box[x][y] += 1
        else:
            hedge[x][y] = 1
            if x > 0:
                box[x - 1][y] += 1
            if x < self.m:
                box[x][y] += 1

        regions = self.get_regions(hedge, vedge, box)
        long_chains = [r for r in regions if not r["is_loop"] and r["length"] >= 3]
        return len(long_chains)

    # ------------------------------------------------------------------
    # Chain / loop decomposition
    # ------------------------------------------------------------------
    def get_regions(self, hedge=None, vedge=None, box=None):
        """Group boxes with exactly 2 sides taken into connected chain/loop
        regions. Two such boxes are connected if the edge between them is
        still free. Operates on self's state by default, but accepts
        explicit hedge/vedge/box grids so it can be reused for simulation
        of hypothetical moves."""
        if hedge is None:
            hedge = self.hedge
        if vedge is None:
            vedge = self.vedge
        if box is None:
            box = self.box

        visited = set()
        regions = []
        for i in range(self.m):
            for j in range(self.n):
                if box[i][j] == 2 and (i, j) not in visited:
                    comp = []
                    stack = [(i, j)]
                    visited.add((i, j))
                    while stack:
                        ci, cj = stack.pop()
                        comp.append((ci, cj))
                        neighbors = []
                        if cj + 1 < self.n and vedge[ci][cj + 1] == 0 and box[ci][cj + 1] == 2:
                            neighbors.append((ci, cj + 1))
                        if cj - 1 >= 0 and vedge[ci][cj] == 0 and box[ci][cj - 1] == 2:
                            neighbors.append((ci, cj - 1))
                        if ci + 1 < self.m and hedge[ci + 1][cj] == 0 and box[ci + 1][cj] == 2:
                            neighbors.append((ci + 1, cj))
                        if ci - 1 >= 0 and hedge[ci][cj] == 0 and box[ci - 1][cj] == 2:
                            neighbors.append((ci - 1, cj))
                        for nb in neighbors:
                            if nb not in visited:
                                visited.add(nb)
                                stack.append(nb)

                    is_loop, degrees = self._classify_region(comp, hedge, vedge, box)
                    regions.append({
                        "boxes": comp,
                        "length": len(comp),
                        "is_loop": is_loop,
                        "degrees": degrees,
                    })
        return regions

    def _classify_region(self, comp, hedge, vedge, box):
        comp_set = set(comp)
        degrees = {}
        for (ci, cj) in comp:
            d = 0
            if cj + 1 < self.n and vedge[ci][cj + 1] == 0 and (ci, cj + 1) in comp_set:
                d += 1
            if cj - 1 >= 0 and vedge[ci][cj] == 0 and (ci, cj - 1) in comp_set:
                d += 1
            if ci + 1 < self.m and hedge[ci + 1][cj] == 0 and (ci + 1, cj) in comp_set:
                d += 1
            if ci - 1 >= 0 and hedge[ci][cj] == 0 and (ci - 1, cj) in comp_set:
                d += 1
            degrees[(ci, cj)] = d
        # A loop is a cycle: every box in it has both of its free edges
        # pointing to other boxes still inside the same region.
        is_loop = len(comp) >= 4 and all(d == 2 for d in degrees.values())
        return is_loop, degrees

    # ------------------------------------------------------------------
    # Shortest-chain-first opening
    # ------------------------------------------------------------------
    def get_shortest_region_opening(self):
        regions = self.get_regions()
        chains = [r for r in regions if not r["is_loop"]]
        loops = [r for r in regions if r["is_loop"]]

        if chains:
            chains.sort(key=lambda r: r["length"])
            target = chains[0]
        elif loops:
            loops.sort(key=lambda r: r["length"])
            target = loops[0]
        else:
            return None

        return self._find_opening_edge(target)

    def _find_opening_edge(self, region):
        comp = region["boxes"]
        comp_set = set(comp)
        degrees = region["degrees"]

        # Prefer opening from a true end of a chain (degree < 2). Loops
        # have no endpoints, so every box is a candidate.
        endpoints = [b for b in comp if degrees[b] < 2]
        candidates = endpoints if endpoints else comp

        for (ci, cj) in candidates:
            if self.hedge[ci][cj] == 0 and not (ci - 1 >= 0 and (ci - 1, cj) in comp_set):
                return (1, ci, cj)
            if self.hedge[ci + 1][cj] == 0 and not (ci + 1 < self.m and (ci + 1, cj) in comp_set):
                return (1, ci + 1, cj)
            if self.vedge[ci][cj] == 0 and not (cj - 1 >= 0 and (ci, cj - 1) in comp_set):
                return (2, ci, cj)
            if self.vedge[ci][cj + 1] == 0 and not (cj + 1 < self.n and (ci, cj + 1) in comp_set):
                return (2, ci, cj + 1)

        # Fallback: any free edge belonging to the region.
        for (ci, cj) in comp:
            if self.hedge[ci][cj] == 0:
                return (1, ci, cj)
            if self.hedge[ci + 1][cj] == 0:
                return (1, ci + 1, cj)
            if self.vedge[ci][cj] == 0:
                return (2, ci, cj)
            if self.vedge[ci][cj + 1] == 0:
                return (2, ci, cj + 1)
        return None

    def makeanymove(self):
        opening = self.get_shortest_region_opening()
        if opening is not None:
            zz, x, y = opening
            self.takeedge(zz, x, y)
        else:
            # No 2-sided boxes exist yet (very early game) - fall back to
            # v3's original first-free-edge scan.
            found = False
            x = y = -1
            for i in range(self.m + 1):
                for j in range(self.n):
                    if self.hedge[i][j] < 1:
                        x, y = i, j
                        found = True
                        break
                if found:
                    break

            if found:
                self.sethedge(x, y)
            else:
                for i in range(self.m):
                    for j in range(self.n + 1):
                        if self.vedge[i][j] < 1:
                            x, y = i, j
                            found = True
                            break
                    if found:
                        break
                if found:
                    self.setvedge(x, y)
                else:
                    return

        if self.player == 0:
            self.makemove()

    # ------------------------------------------------------------------
    # Singleton / doubleton safe-move special cases (unchanged from v3)
    # ------------------------------------------------------------------
    def singleton(self):
        for i in range(self.m):
            for j in range(self.n):
                if self.box[i][j] == 2:
                    numb = 0
                    if self.hedge[i][j] < 1:
                        if i < 1 or self.box[i - 1][j] < 2:
                            numb += 1
                    self.zz = 2
                    if self.vedge[i][j] < 1:
                        if j < 1 or self.box[i][j - 1] < 2:
                            numb += 1
                        if numb > 1:
                            self.x, self.y = i, j
                            return True
                    if self.vedge[i][j + 1] < 1:
                        if j + 1 == self.n or self.box[i][j + 1] < 2:
                            numb += 1
                        if numb > 1:
                            self.x, self.y = i, j + 1
                            return True
                    self.zz = 1
                    if self.hedge[i + 1][j] < 1:
                        if i + 1 == self.m or self.box[i + 1][j] < 2:
                            numb += 1
                        if numb > 1:
                            self.x, self.y = i + 1, j
                            return True
        return False

    def doubleton(self):
        self.zz = 2
        for i in range(self.m):
            for j in range(self.n - 1):
                if self.box[i][j] == 2 and self.box[i][j + 1] == 2 and self.vedge[i][j + 1] < 1:
                    if self.ldub(i, j) and self.rdub(i, j + 1):
                        self.x, self.y = i, j + 1
                        return True
        self.zz = 1
        for j in range(self.n):
            for i in range(self.m - 1):
                if self.box[i][j] == 2 and self.box[i + 1][j] == 2 and self.hedge[i + 1][j] < 1:
                    if self.udub(i, j) and self.ddub(i + 1, j):
                        self.x, self.y = i + 1, j
                        return True
        return False

    def ldub(self, i, j):
        if self.vedge[i][j] < 1:
            if j < 1 or self.box[i][j - 1] < 2:
                return True
        elif self.hedge[i][j] < 1:
            if i < 1 or self.box[i - 1][j] < 2:
                return True
        elif i == self.m - 1 or self.box[i + 1][j] < 2:
            return True
        return False

    def rdub(self, i, j):
        if self.vedge[i][j + 1] < 1:
            if j + 1 == self.n or self.box[i][j + 1] < 2:
                return True
        elif self.hedge[i][j] < 1:
            if i < 1 or self.box[i - 1][j] < 2:
                return True
        elif i + 1 == self.m or self.box[i + 1][j] < 2:
            return True
        return False

    def udub(self, i, j):
        if self.hedge[i][j] < 1:
            if i < 1 or self.box[i - 1][j] < 2:
                return True
        elif self.vedge[i][j] < 1:
            if j < 1 or self.box[i][j - 1] < 2:
                return True
        elif j == self.n - 1 or self.box[i][j + 1] < 2:
            return True
        return False

    def ddub(self, i, j):
        if self.hedge[i + 1][j] < 1:
            if i == self.m - 1 or self.box[i + 1][j] < 2:
                return True
        elif self.vedge[i][j] < 1:
            if j < 1 or self.box[i][j - 1] < 2:
                return True
        elif j == self.n - 1 or self.box[i][j + 1] < 2:
            return True
        return False

    # ------------------------------------------------------------------
    # Double-cross sacrifice logic for a forced chain/loop (unchanged from v3)
    # ------------------------------------------------------------------
    def sac(self, i, j):
        self.count = 0
        self.loop = False
        self.incount(0, i, j)
        if not self.loop:
            self.takeallbut(i, j)
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
                if self.box[i][j - 1] > 2:
                    self.count += 1
                    self.loop = True
                elif self.box[i][j - 1] > 1:
                    self.incount(3, i, j - 1)
        elif k != 2 and self.hedge[i][j] < 1:
            if i > 0:
                if self.box[i - 1][j] > 2:
                    self.count += 1
                    self.loop = True
                elif self.box[i - 1][j] > 1:
                    self.incount(4, i - 1, j)
        elif k != 3 and self.vedge[i][j + 1] < 1:
            if j < self.n - 1:
                if self.box[i][j + 1] > 2:
                    self.count += 1
                    self.loop = True
                elif self.box[i][j + 1] > 1:
                    self.incount(1, i, j + 1)
        elif k != 4 and self.hedge[i + 1][j] < 1:
            if i < self.m - 1:
                if self.box[i + 1][j] > 2:
                    self.count += 1
                    self.loop = True
                elif self.box[i + 1][j] > 1:
                    self.incount(2, i + 1, j)

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
        if self.hedge[i][j] < 1:
            self.sethedge(i, j)
        elif self.vedge[i][j] < 1:
            self.setvedge(i, j)
        elif self.hedge[i + 1][j] < 1:
            self.sethedge(i + 1, j)
        else:
            self.setvedge(i, j + 1)

    def outcount(self, k, i, j):
        if self.count > 0:
            if k != 1 and self.vedge[i][j] < 1:
                if self.count != 2:
                    self.setvedge(i, j)
                self.count -= 1
                self.outcount(3, i, j - 1)
            elif k != 2 and self.hedge[i][j] < 1:
                if self.count != 2:
                    self.sethedge(i, j)
                self.count -= 1
                self.outcount(4, i - 1, j)
            elif k != 3 and self.vedge[i][j + 1] < 1:
                if self.count != 2:
                    self.setvedge(i, j + 1)
                self.count -= 1
                self.outcount(1, i, j + 1)
            elif k != 4 and self.hedge[i + 1][j] < 1:
                if self.count != 2:
                    self.sethedge(i + 1, j)
                self.count -= 1
                self.outcount(2, i + 1, j)