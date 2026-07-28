import random
from agent_interface import BaseAgent


# ---------------------------------------------------------------------------
# Static board topology — built once per board size, cached globally
# ---------------------------------------------------------------------------

_TABLE_CACHE: dict = {}


def _build_tables(size):
    """
    Precompute adjacency tables for a Dots-and-Boxes board of given size.

    Line indexing (same as DotsAndBoxesGame):
      horizontal: index = row*size + col   (row in 0..size, col in 0..size-1)
      vertical:   index = half + col*size + row  (col in 0..size, row in 0..size-1)
      where half = N_LINES // 2 = size*(size+1)

    Returns
    -------
    adj_boxes : list[tuple[int]]  — adj_boxes[edge] = 1 or 2 box-indices
    adj_edges : list[tuple[int]]  — adj_edges[box]  = 4 edge-indices
    n_lines   : int
    n_boxes   : int
    """
    n_lines = 2 * size * (size + 1)
    half    = n_lines // 2
    n_boxes = size * size

    adj_boxes = []
    for line in range(n_lines):
        if line < half:                      # horizontal
            i = line // size
            j = line  % size
            if i == 0:
                boxes = (i * size + j,)
            elif i == size:
                boxes = ((i - 1) * size + j,)
            else:
                boxes = ((i - 1) * size + j, i * size + j)
        else:                                # vertical
            vl = line - half
            j  = vl // size
            i  = vl  % size
            if j == 0:
                boxes = (i * size + j,)
            elif j == size:
                boxes = (i * size + j - 1,)
            else:
                boxes = (i * size + j - 1, i * size + j)
        adj_boxes.append(boxes)

    adj_edges = []
    for box_idx in range(n_boxes):
        i = box_idx // size
        j = box_idx  % size
        top    = i * size + j
        bottom = (i + 1) * size + j
        left   = half + j * size + i
        right  = half + (j + 1) * size + i
        adj_edges.append((top, bottom, left, right))

    return adj_boxes, adj_edges, n_lines, n_boxes


def _get_tables(size):
    if size not in _TABLE_CACHE:
        _TABLE_CACHE[size] = _build_tables(size)
    return _TABLE_CACHE[size]


class UCLABot_v6(BaseAgent):
    """
    UCLABot_v3 strategy with true algorithmic optimisations.

    Key improvements over the previous bitboard version
    ---------------------------------------------------
    1. box_fill_count[] is read DIRECTLY from the game object — the game
       already maintains it incrementally via execute_move(). No per-call
       rebuild.

    2. three_edge_set — Python set of flat box-indices with fill == 3.
       sides3() / takeall3s() are now O(1) / O(k) instead of scanning
       all 25 boxes every time.

    3. safe_edges — Python set of free edge-indices whose placement does NOT
       raise any adjacent box to fill == 3 (would gift a capture to the
       opponent). sides01() picks uniformly at random from this set in O(1).

    4. Both sets are updated INCREMENTALLY inside _place_edge(): only the
       1-2 boxes adjacent to the placed edge are touched — no full-board scan.

    5. Flat bool list edge_taken[] replaces hset()/vset() method-call overhead.

    6. Precomputed adj_boxes[] / adj_edges[] replace all (i,j) coordinate math
       in the hot path.

    Strategy is identical to UCLABot_v3 (takesafe3s → sides3/sides01 →
    singleton → doubleton → sac chain logic → makeanymove).
    """

    def __init__(self, name="UCLABot_v6", size=5):
        super().__init__(name)
        self._size   = size
        self._tables = _get_tables(size)
        self.move_queue: list = []
        # working state rebuilt each get_move() call
        self._bc:    list = []
        self._et:    list = []
        self._three: set  = set()
        self._safe:  set  = set()
        # strategy temporaries
        self._move   = -1
        self._count  = 0
        self._loop   = False
        self._player = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def get_move(self, game) -> int:
        # Drain pre-planned queue (moves decided in a previous call)
        while self.move_queue:
            mv = self.move_queue[0]
            if game.l[mv] != 0:          # queue is stale — someone else moved
                self.move_queue.clear()
                break
            return self.move_queue.pop(0)

        size = game.SIZE
        if size != self._size:
            self._size   = size
            self._tables = _get_tables(size)

        adj_boxes, adj_edges, n_lines, n_boxes = self._tables

        # ---- Build working state in O(N_LINES + N_BOXES) ----
        # box_fill_count: the game already maintains this incrementally
        bc = list(game.box_fill_count)
        # edge_taken: flat bool array
        et = [v != 0 for v in game.l]

        # three_edge_set
        three = {b for b in range(n_boxes) if bc[b] == 3}

        # safe_edges: free edges that don't create a 3-edge box
        safe: set = set()
        for edge in range(n_lines):
            if et[edge]:
                continue
            for bidx in adj_boxes[edge]:
                if bc[bidx] == 2:
                    break
            else:
                safe.add(edge)

        self._bc     = bc
        self._et     = et
        self._three  = three
        self._safe   = safe
        self._player = 0

        self._makemove()

        if self.move_queue:
            return self.move_queue.pop(0)

        valid = game.get_valid_moves()
        return valid[0] if valid else -1

    # ------------------------------------------------------------------
    # Incremental edge placement — O(adjacency) ≈ O(1)
    # ------------------------------------------------------------------

    def _place_edge(self, edge: int):
        """Mark edge as taken; update bc[], three, safe incrementally."""
        adj_boxes, adj_edges, n_lines, n_boxes = self._tables
        bc    = self._bc
        et    = self._et
        three = self._three
        safe  = self._safe

        et[edge] = True
        safe.discard(edge)
        self.move_queue.append(edge)

        scored = False
        for bidx in adj_boxes[edge]:
            bc[bidx] += 1
            cnt = bc[bidx]
            if cnt == 3:
                three.add(bidx)
                # All free edges of this box are now unsafe (placing them
                # completes the box, gifting it to the opponent)
                for e2 in adj_edges[bidx]:
                    safe.discard(e2)
            elif cnt == 4:
                three.discard(bidx)
                scored = True
                # Re-evaluate free edges around this now-complete box —
                # they may have become safe again
                for e2 in adj_edges[bidx]:
                    if not et[e2]:
                        for bidx2 in adj_boxes[e2]:
                            if bc[bidx2] == 2:
                                break
                        else:
                            safe.add(e2)

        if not scored:
            self._player ^= 1

    # ------------------------------------------------------------------
    # Coordinate helpers (inline-friendly)
    # ------------------------------------------------------------------

    def _rc(self, bidx: int):
        return divmod(bidx, self._size)

    def _hedge(self, i: int, j: int) -> int:
        return i * self._size + j

    def _vedge(self, i: int, j: int) -> int:
        return self._tables[2] // 2 + j * self._size + i

    def _ht(self, i: int, j: int) -> bool:
        return self._et[i * self._size + j]

    def _vt(self, i: int, j: int) -> bool:
        return self._et[self._tables[2] // 2 + j * self._size + i]

    def _bc_at(self, i: int, j: int) -> int:
        return self._bc[i * self._size + j]

    # ------------------------------------------------------------------
    # Strategy — identical control flow to UCLABot_v3
    # ------------------------------------------------------------------

    def _makemove(self):
        self._takesafe3s()
        if self._three:
            bidx = next(iter(self._three))
            i, j = self._rc(bidx)
            if self._sides01():
                self._takeall3s()
                self._place_edge(self._move)
            else:
                self._sac(i, j)
        elif self._sides01():
            self._place_edge(self._move)
        elif self._singleton():
            self._place_edge(self._move)
        elif self._doubleton():
            self._place_edge(self._move)
        else:
            self._makeanymove()

    # --- takesafe3s ---

    def _takesafe3s(self):
        """Capture all 3-edge boxes that are safe to take without opening a chain."""
        changed = True
        while changed:
            changed = False
            for bidx in list(self._three):
                i, j = self._rc(bidx)
                edge = self._safe_capture_edge(i, j)
                if edge is not None:
                    self._place_edge(edge)
                    changed = True

    def _safe_capture_edge(self, i: int, j: int):
        """Return the free edge of 3-edge box (i,j) that doesn't open a 2-chain."""
        s = self._size
        if not self._ht(i, j):
            if i == 0 or self._bc_at(i - 1, j) != 2:
                return self._hedge(i, j)
        if not self._ht(i + 1, j):
            if i == s - 1 or self._bc_at(i + 1, j) != 2:
                return self._hedge(i + 1, j)
        if not self._vt(i, j):
            if j == 0 or self._bc_at(i, j - 1) != 2:
                return self._vedge(i, j)
        if not self._vt(i, j + 1):
            if j == s - 1 or self._bc_at(i, j + 1) != 2:
                return self._vedge(i, j + 1)
        return None

    # --- takeall3s ---

    def _takeall3s(self):
        while self._three:
            bidx = next(iter(self._three))
            i, j = self._rc(bidx)
            self._takebox(i, j)

    def _takebox(self, i: int, j: int):
        if not self._ht(i, j):         self._place_edge(self._hedge(i, j))
        elif not self._vt(i, j):       self._place_edge(self._vedge(i, j))
        elif not self._ht(i + 1, j):   self._place_edge(self._hedge(i + 1, j))
        else:                           self._place_edge(self._vedge(i, j + 1))

    # --- sides01: O(1) — just pick from safe_edges set ---

    def _sides01(self) -> bool:
        if not self._safe:
            return False
        self._move = random.choice(list(self._safe))
        return True

    # --- singleton ---

    def _singleton(self) -> bool:
        """Find a 2-edge box with at least 2 free safe exits."""
        s = self._size
        for bidx in range(s * s):
            if self._bc[bidx] != 2:
                continue
            i, j = self._rc(bidx)
            opts = []
            if not self._ht(i, j) and (i == 0 or self._bc_at(i - 1, j) < 2):
                opts.append(self._hedge(i, j))
            if not self._ht(i + 1, j) and (i == s - 1 or self._bc_at(i + 1, j) < 2):
                opts.append(self._hedge(i + 1, j))
            if not self._vt(i, j) and (j == 0 or self._bc_at(i, j - 1) < 2):
                opts.append(self._vedge(i, j))
            if not self._vt(i, j + 1) and (j == s - 1 or self._bc_at(i, j + 1) < 2):
                opts.append(self._vedge(i, j + 1))
            if len(opts) >= 2:
                self._move = random.choice(opts)
                return True
        return False

    # --- doubleton ---

    def _doubleton(self) -> bool:
        """Find two adjacent 2-edge boxes sharing a free edge, both with safe exits."""
        s = self._size
        # horizontal pairs
        for i in range(s):
            for j in range(s - 1):
                if (self._bc_at(i, j) == 2 and self._bc_at(i, j + 1) == 2
                        and not self._vt(i, j + 1)):
                    if self._ldub(i, j) and self._rdub(i, j + 1):
                        self._move = self._vedge(i, j + 1)
                        return True
        # vertical pairs
        for j in range(s):
            for i in range(s - 1):
                if (self._bc_at(i, j) == 2 and self._bc_at(i + 1, j) == 2
                        and not self._ht(i + 1, j)):
                    if self._udub(i, j) and self._ddub(i + 1, j):
                        self._move = self._hedge(i + 1, j)
                        return True
        return False

    def _ldub(self, i, j):
        s = self._size
        if not self._vt(i, j):
            if j < 1 or self._bc_at(i, j - 1) < 2: return True
        elif not self._ht(i, j):
            if i < 1 or self._bc_at(i - 1, j) < 2: return True
        elif i == s - 1 or self._bc_at(i + 1, j) < 2: return True
        return False

    def _rdub(self, i, j):
        s = self._size
        if not self._vt(i, j + 1):
            if j + 1 == s or self._bc_at(i, j + 1) < 2: return True
        elif not self._ht(i, j):
            if i < 1 or self._bc_at(i - 1, j) < 2: return True
        elif i + 1 == s or self._bc_at(i + 1, j) < 2: return True
        return False

    def _udub(self, i, j):
        s = self._size
        if not self._ht(i, j):
            if i < 1 or self._bc_at(i - 1, j) < 2: return True
        elif not self._vt(i, j):
            if j < 1 or self._bc_at(i, j - 1) < 2: return True
        elif j == s - 1 or self._bc_at(i, j + 1) < 2: return True
        return False

    def _ddub(self, i, j):
        s = self._size
        if not self._ht(i + 1, j):
            if i == s - 1 or self._bc_at(i + 1, j) < 2: return True
        elif not self._vt(i, j):
            if j < 1 or self._bc_at(i, j - 1) < 2: return True
        elif j == s - 1 or self._bc_at(i, j + 1) < 2: return True
        return False

    # --- sacrifice / chain logic ---

    def _sac(self, i: int, j: int):
        s = self._size
        self._count = 0
        self._loop  = False
        self._incount(0, i, j)
        if not self._loop:
            self._takeallbut(i, j)
        boxes_taken = sum(1 for bc in self._bc if bc == 4)
        if self._count + boxes_taken == s * s:
            self._takeall3s()
        else:
            if self._loop:
                self._count -= 2
            self._outcount(0, i, j)

    def _incount(self, k: int, i: int, j: int):
        s = self._size
        self._count += 1
        if k != 1 and not self._vt(i, j):
            if j > 0:
                bc = self._bc_at(i, j - 1)
                if bc > 2:
                    self._count += 1; self._loop = True
                elif bc > 1:
                    self._incount(3, i, j - 1)
        elif k != 2 and not self._ht(i, j):
            if i > 0:
                bc = self._bc_at(i - 1, j)
                if bc > 2:
                    self._count += 1; self._loop = True
                elif bc > 1:
                    self._incount(4, i - 1, j)
        elif k != 3 and not self._vt(i, j + 1):
            if j < s - 1:
                bc = self._bc_at(i, j + 1)
                if bc > 2:
                    self._count += 1; self._loop = True
                elif bc > 1:
                    self._incount(1, i, j + 1)
        elif k != 4 and not self._ht(i + 1, j):
            if i < s - 1:
                bc = self._bc_at(i + 1, j)
                if bc > 2:
                    self._count += 1; self._loop = True
                elif bc > 1:
                    self._incount(2, i + 1, j)

    def _takeallbut(self, xi: int, xj: int):
        """Take all 3-edge boxes except (xi, xj)."""
        while True:
            found = None
            for bidx in self._three:
                i, j = self._rc(bidx)
                if i != xi or j != xj:
                    found = (i, j)
                    break
            if found is None:
                break
            self._takebox(found[0], found[1])

    def _outcount(self, k: int, i: int, j: int):
        if self._count <= 0:
            return
        if k != 1 and not self._vt(i, j):
            if self._count != 2:
                self._place_edge(self._vedge(i, j))
            self._count -= 1
            self._outcount(3, i, j - 1)
        elif k != 2 and not self._ht(i, j):
            if self._count != 2:
                self._place_edge(self._hedge(i, j))
            self._count -= 1
            self._outcount(4, i - 1, j)
        elif k != 3 and not self._vt(i, j + 1):
            if self._count != 2:
                self._place_edge(self._vedge(i, j + 1))
            self._count -= 1
            self._outcount(1, i, j + 1)
        elif k != 4 and not self._ht(i + 1, j):
            if self._count != 2:
                self._place_edge(self._hedge(i + 1, j))
            self._count -= 1
            self._outcount(2, i + 1, j)

    # --- fallback: any free edge ---

    def _makeanymove(self):
        _, _, n_lines, _ = self._tables
        for edge in range(n_lines):
            if not self._et[edge]:
                self._place_edge(edge)
                if self._player == 0:
                    self._makemove()
                return
