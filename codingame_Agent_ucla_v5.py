import sys
import random
import time
import numpy as np
from typing import List, Tuple


class _EndgameSolverTimeout(Exception):
    """Raised internally to bail out of the exact endgame solver's
    recursion when the per-move time budget runs out, so callers can fall
    back to the old heuristics instead of missing the deadline."""
    pass


class BaseAgent:
    def __init__(self, name=""):
        self.name = name


class DotsAndBoxesGame:
    def __init__(self, size: int, starting_player: int = 1):
        self.SIZE = size
        self.current_player = starting_player
        self.result = None
        self.N_LINES = 2 * size * (size + 1)
        self.l = np.zeros((self.N_LINES,), dtype=np.float32)
        self.N_BOXES = size * size
        self.b = np.zeros((size, size))
        self._history: list = []

    def draw_line(self, line: int):
        self.l[line] = self.current_player

    def switch_current_player(self):
        self.current_player *= -1

    def capture_box(self, row: int, col: int):
        self.b[row][col] = self.current_player

    def execute_move(self, line: int):
        player_before = self.current_player
        result_before = self.result
        self.draw_line(line)
        boxes_captured = []
        for box in self.get_boxes_of_line(line):
            lines = self.get_lines_of_box(box)
            if np.count_nonzero(self.l[lines]) == 4:
                self.capture_box(row=box[0], col=box[1])
                boxes_captured.append(box)
        if not boxes_captured:
            self.switch_current_player()
        else:
            self.check_finished()
        self._history.append((line, boxes_captured, player_before, result_before))

    def undo_move(self):
        line, boxes_captured, player_before, result_before = self._history.pop()
        self.current_player = player_before
        self.result = result_before
        self.l[line] = 0.0
        for box in boxes_captured:
            self.b[box[0]][box[1]] = 0.0

    def check_finished(self):
        if self.result is not None:
            return
        if np.count_nonzero(self.l == 0) == 0:
            p1 = int(np.sum(self.b == 1))
            p2 = int(np.sum(self.b == -1))
            self.result = 1 if p1 > p2 else (-1 if p2 > p1 else 0)

    def is_running(self) -> bool:
        return self.result is None

    def get_valid_moves(self) -> List[int]:
        return np.where(self.l == 0)[0].tolist()

    def get_boxes_of_line(self, line: int) -> List[Tuple[int, int]]:
        if line < int(self.N_LINES / 2):
            i = line // self.SIZE
            j = line % self.SIZE
            if i == 0:           return [(i, j)]
            elif i == self.SIZE: return [(i - 1, j)]
            else:                return [(i - 1, j), (i, j)]
        else:
            line = line - int(self.N_LINES / 2)
            j = line // self.SIZE
            i = line % self.SIZE
            if j == 0:           return [(i, j)]
            elif j == self.SIZE: return [(i, j - 1)]
            else:                return [(i, j - 1), (i, j)]

    def get_lines_of_box(self, box: Tuple[int, int]) -> List[int]:
        i, j = box
        return [
            i * self.SIZE + j,
            (i + 1) * self.SIZE + j,
            int(self.N_LINES / 2) + j * self.SIZE + i,
            int(self.N_LINES / 2) + (j + 1) * self.SIZE + i,
        ]


class UCLABot_v4(BaseAgent):
    def __init__(self, name: str = "UCLABot_v4"):
        super().__init__(name)
        self.move_queue: list = []

    def get_move(self, game) -> int:
        import numpy as np
        if int(np.count_nonzero(game.l)) == 0:
            self.move_queue.clear()

        while self.move_queue:
            mv = self.move_queue[0]
            if game.l[mv] != 0:
                self.move_queue.clear()
                break
            return self.move_queue.pop(0)
        t_start = time.time()
        # First move of the whole game (no lines drawn yet) gets a much larger
        # budget (~1s, minus a safety margin for I/O/setup). Every subsequent
        # move must stay comfortably under the 0.1s per-move limit.
        drawn = int(np.count_nonzero(game.l))
        if drawn == 0:
            budget = 0.85
        else:
            budget = 0.075
        self.deadline = t_start + budget
        self.size    = game.SIZE
        self.m       = self.size
        self.n       = self.size
        self.N_LINES = game.N_LINES
        self.hedge = [[0] * self.n for _ in range(self.m + 1)]
        self.vedge = [[0] * (self.n + 1) for _ in range(self.m)]
        self.box   = [[0] * self.n for _ in range(self.m)]
        for r in range(self.m + 1):
            for c in range(self.n):
                if game.l[r * self.n + c] != 0:
                    self.hedge[r][c] = 1
                    if r > 0:      self.box[r - 1][c] += 1
                    if r < self.m: self.box[r][c]     += 1
        for c in range(self.n + 1):
            for r in range(self.m):
                if game.l[int(self.N_LINES / 2) + c * self.size + r] != 0:
                    self.vedge[r][c] = 1
                    if c > 0:      self.box[r][c - 1] += 1
                    if c < self.n: self.box[r][c]      += 1
        self.player = 0
        self.zz = 0
        self.x = self.y = self.u = self.v = -1
        self.count = 0
        self.loop  = False
        self.makemove()
        if self.move_queue:
            return self.move_queue.pop(0)
        valid = game.get_valid_moves()
        return valid[0] if valid else -1

    def makemove(self):
        self.takesafe3s()
        if self.sides3():
            if self.best_safe_move():
                self.takeall3s()
                self.takeedge(self.zz, self.x, self.y)
            else:
                picked = self._pick_sac_chain()
                if picked is not None:
                    self.u, self.v = picked
                self.sac(self.u, self.v)
        elif self.best_safe_move():
            self.takeedge(self.zz, self.x, self.y)
        elif self.best_singleton():
            self.takeedge(self.zz, self.x, self.y)
        elif self.best_doubleton():
            self.takeedge(self.zz, self.x, self.y)
        else:
            self.makeanymove()

    def best_safe_move(self) -> bool:
        best_score = None
        best_list  = []

        our_safe = []
        for i in range(self.m + 1):
            for j in range(self.n):
                if self.safehedge(i, j):
                    our_safe.append((1, i, j))

        for i in range(self.m):
            for j in range(self.n + 1):
                if self.safevedge(i, j):
                    our_safe.append((2, i, j))

        if not our_safe:
            return False

        random.shuffle(our_safe)
        for cand in our_safe:
            if time.time() > self.deadline:
                break
            sc = self._minimax_1_ply_score(cand)
            sc += self._positional_bonus(cand)
            if best_score is None or sc > best_score:
                best_score = sc
                best_list = [cand]
            elif sc == best_score:
                best_list.append(cand)

        if not best_list:
            best_list = our_safe
        self.zz, self.x, self.y = random.choice(best_list)
        return True

    def _positional_bonus(self, move) -> float:
        # Small tiebreaker: prefer edges nearer the board centre and moves
        # that keep more future safe moves open (mobility), so the bot does
        # not commit to lopsided regions or collapse mobility unnecessarily
        # when several candidates score equally on the chain heuristic.
        zz, x, y = move
        cx, cy = (self.m - 1) / 2.0, (self.n - 1) / 2.0
        if zz == 1:
            dist = abs(x - 0.5 - cx * 0) + abs((x if x < self.m else x - 1) - cx) + abs(y - cy)
        else:
            dist = abs((y if y < self.n else y - 1) - cy) + abs(x - cx)
        max_dist = self.m + self.n
        return (max_dist - dist) * 0.05

    def _minimax_1_ply_score(self, move) -> float:
        h = [row[:] for row in self.hedge]
        v = [row[:] for row in self.vedge]
        b = [row[:] for row in self.box]
        self._apply_edge_copy(move[0], move[1], move[2], h, v, b)

        opp_safe = []
        for i in range(self.m + 1):
            for j in range(self.n):
                if h[i][j] == 0 and self._is_safehedge(i, j, b): opp_safe.append((1, i, j))
        for i in range(self.m):
            for j in range(self.n + 1):
                if v[i][j] == 0 and self._is_safevedge(i, j, b): opp_safe.append((2, i, j))

        if not opp_safe:
            return self._eval_chain_config(h, v, b) + 0.02 * self._mobility(h, v, b)

        random.shuffle(opp_safe)
        # Sample as many opponent replies as the remaining time budget
        # comfortably allows instead of a fixed cap of 3 -- an exhaustive
        # (or near-exhaustive) reply check is a much better approximation
        # of true minimax, and small boards/endgames usually have few
        # simultaneous safe replies anyway.
        remaining = self.deadline - time.time()
        if remaining <= 0:
            k = 1
        elif remaining > 0.02:
            k = len(opp_safe)
        else:
            k = min(len(opp_safe), 3)

        min_score = None
        for opp_m in opp_safe[:k]:
            if time.time() > self.deadline:
                break
            h2 = [row[:] for row in h]
            v2 = [row[:] for row in v]
            b2 = [row[:] for row in b]
            self._apply_edge_copy(opp_m[0], opp_m[1], opp_m[2], h2, v2, b2)
            self._simulate_safe_phase(h2, v2, b2)
            sc = self._eval_chain_config(h2, v2, b2) + 0.02 * self._mobility(h2, v2, b2)
            if min_score is None or sc < min_score:
                min_score = sc
        return min_score if min_score is not None else self._eval_chain_config(h, v, b)

    def _mobility(self, h: list, v: list, b: list) -> int:
        # Number of currently-safe moves left in a position: a rough proxy
        # for "how far we still are from being forced into the chain
        # phase". Preferring moves that keep this higher delays chain
        # formation, in the spirit of a connectivity heuristic.
        n = 0
        for i in range(self.m + 1):
            for j in range(self.n):
                if h[i][j] == 0 and self._is_safehedge(i, j, b): n += 1
        for i in range(self.m):
            for j in range(self.n + 1):
                if v[i][j] == 0 and self._is_safevedge(i, j, b): n += 1
        return n

    def _score_safe_move(self, zz: int, x: int, y: int) -> float:
        h = [row[:] for row in self.hedge]
        v = [row[:] for row in self.vedge]
        b = [row[:] for row in self.box]
        self._apply_edge_copy(zz, x, y, h, v, b)
        self._simulate_safe_phase(h, v, b)
        return self._eval_chain_config(h, v, b)

    def _simulate_safe_phase(self, h: list, v: list, b: list, max_moves: int = 50):
        safe_moves = set()
        for i in range(self.m + 1):
            for j in range(self.n):
                if h[i][j] == 0 and self._is_safehedge(i, j, b): safe_moves.add((1, i, j))
        for i in range(self.m):
            for j in range(self.n + 1):
                if v[i][j] == 0 and self._is_safevedge(i, j, b): safe_moves.add((2, i, j))
        
        for _ in range(max_moves):
            if not safe_moves:
                break
            best_danger = 99
            best_move = None
            to_remove = []
            for move in safe_moves:
                zz, x, y = move
                if zz == 1 and not self._is_safehedge(x, y, b):
                    to_remove.append(move); continue
                if zz == 2 and not self._is_safevedge(x, y, b):
                    to_remove.append(move); continue
                
                d = self._danger_count(zz, x, y, b)
                if d < best_danger:
                    best_danger = d
                    best_move = move
                    if d == 0: break
            
            for rm in to_remove:
                safe_moves.remove(rm)
                
            if best_move is None:
                break
            
            self._apply_edge_copy(best_move[0], best_move[1], best_move[2], h, v, b)
            safe_moves.remove(best_move)

    def _danger_count(self, zz: int, x: int, y: int, b: list) -> int:
        count = 0
        if zz == 1:
            if x > 0      and b[x - 1][y] == 1: count += 1
            if x < self.m and b[x][y]     == 1: count += 1
        else:
            if y > 0      and b[x][y - 1] == 1: count += 1
            if y < self.n and b[x][y]     == 1: count += 1
        return count

    def _apply_edge_copy(self, zz: int, x: int, y: int, h: list, v: list, b: list):
        if zz == 1:
            h[x][y] = 1
            if x > 0:      b[x - 1][y] += 1
            if x < self.m: b[x][y]     += 1
        else:
            v[x][y] = 1
            if y > 0:      b[x][y - 1] += 1
            if y < self.n: b[x][y]      += 1

    def _eval_chain_config(self, h: list, v: list, b: list) -> float:
        danger_2 = capture_3 = 0
        for i in range(self.m):
            for j in range(self.n):
                bv = b[i][j]
                if bv == 2: danger_2  += 1
                elif bv == 3: capture_3 += 1
        chains, loop_count, complex_lens = self._find_chains(h, v, b)
        long_chains        = sum(1 for c in chains if c >= 3)
        short_chains       = sum(1 for c in chains if c == 2)
        complex_count      = len(complex_lens)
        total_chain_boxes  = sum(chains) + sum(complex_lens)
        parity       = (long_chains + loop_count + complex_count) % 2
        parity_bonus = 30 if parity == 1 else -30
        return float(
              parity_bonus
            - danger_2        * 10
            - capture_3       *  5
            - long_chains     *  4
            - short_chains    *  2
            - complex_count   *  5
            - total_chain_boxes *  1
        )

    # ------------------------------------------------------------------
    # Exact abstract endgame solver (Berlekamp chain rule).
    #
    # Once the board has settled into a fixed set of chains/loops with no
    # safe moves left, optimal play depends only on the MULTISET of
    # chain/loop lengths, not on the actual board -- so instead of the old
    # "always double-cross unless it's the last chain" heuristic, we solve
    # that abstract game exactly with a small memoized recursion.
    #
    # f(S) = the optimal net-box margin for whoever is about to MOVE in
    # abstract state S (a multiset of (length, is_loop) pairs), i.e. the
    # player who has no safe move left and must sacrifice ("open") one of
    # the remaining components. The opener picks whichever component
    # maximises their own outcome; the receiver of that sacrifice then
    # picks, for that specific component, whichever of "take all" /
    # "double-cross" (leave the last 2 boxes, or 4 for a loop) minimises
    # the opener's resulting value -- which is equivalent to maximising
    # their own, since the game is zero-sum. This same recursion answers
    # two different real questions: which component we should open when
    # we're the one out of safe moves (see _smart_sacrifice), and whether
    # we should take-all or double-cross a chain that's already been
    # opened for us (see _should_take_all_chain).
    # ------------------------------------------------------------------

    _EG_MAX_COMPONENTS = 12  # cap to keep the recursion inside the move budget

    def _endgame_value(self, multiset) -> float:
        if not multiset:
            return 0.0
        if multiset in self._eg_memo:
            return self._eg_memo[multiset]
        self._eg_calls += 1
        if self._eg_calls % 200 == 0 and time.time() > self.deadline:
            raise _EndgameSolverTimeout()
        best = None
        for item in set(multiset):
            rest = list(multiset)
            rest.remove(item)
            rest = tuple(sorted(rest))
            v = self._endgame_option_value(item[0], item[1], rest)
            if best is None or v > best:
                best = v
        self._eg_memo[multiset] = best
        return best

    def _endgame_option_value(self, length: int, is_loop: bool, rest) -> float:
        """Value to the ORIGINAL OPENER of sacrificing a component of this
        length/type, given the receiver responds optimally and `rest` is
        the remaining multiset after this component is removed."""
        k = 4 if is_loop else 2
        f_rest = self._endgame_value(rest)
        v_take_all = -length - f_rest
        if length >= k:
            v_double_cross = (2 * k - length) + f_rest
            return min(v_take_all, v_double_cross)
        return v_take_all

    def _endgame_best_open(self, multiset):
        """Which (length, is_loop) component should the opener sacrifice
        first? Returns that key, or None if multiset is empty."""
        if not multiset:
            return None
        best_item = best_val = None
        for item in set(multiset):
            rest = list(multiset)
            rest.remove(item)
            rest = tuple(sorted(rest))
            v = self._endgame_option_value(item[0], item[1], rest)
            if best_val is None or v > best_val:
                best_val = v; best_item = item
        return best_item

    def _endgame_receiver_should_double_cross(self, length: int, is_loop: bool, rest) -> bool:
        """Given a component that is already opened (a 3-sided box exists)
        of this length/type, and `rest` the other currently-open
        components, should the receiver (us) double-cross it (leave the
        last 2 boxes, 4 for a loop) instead of taking it all?"""
        k = 4 if is_loop else 2
        if length < k:
            return False
        f_rest = self._endgame_value(rest)
        v_take_all = -length - f_rest
        v_double_cross = (2 * k - length) + f_rest
        return v_double_cross < v_take_all

    def _components_multiset(self, comps):
        return tuple(sorted((c['length'], c['kind'] == 'loop') for c in comps
                             if c['kind'] in ('chain', 'loop')))

    def _should_take_all_chain(self, i: int, j: int) -> bool:
        """Should we take the whole currently-open chain/loop entered at
        (i, j) rather than double-crossing it? Falls back to the old
        always-double-cross behaviour if the position contains a
        branching complex (the clean theory doesn't cover those) or has
        too many components for the solver to be worth the time."""
        comps = self._detect_components(self.hedge, self.vedge, self.box)
        if any(c['kind'] == 'complex' for c in comps) or len(comps) > self._EG_MAX_COMPONENTS:
            return False
        this_comp = next((c for c in comps if (i, j) in c['boxes']), None)
        rest = self._components_multiset([c for c in comps if c is not this_comp])
        length = this_comp['length'] if this_comp is not None else self.count
        self._eg_memo = {}
        self._eg_calls = 0
        try:
            double_cross = self._endgame_receiver_should_double_cross(length, self.loop, rest)
        except _EndgameSolverTimeout:
            return False
        return not double_cross

    def _box_edges(self, i: int, j: int):
        return [(1, i, j), (1, i + 1, j), (2, i, j), (2, i, j + 1)]

    def _edge_drawn(self, zz: int, x: int, y: int) -> bool:
        return (self.hedge[x][y] if zz == 1 else self.vedge[x][y]) >= 1

    def _edge_other_box(self, zz: int, x: int, y: int, i: int, j: int):
        """Given edge (zz, x, y) is one of the 4 edges of box (i, j),
        return the box on the other side of it, or None at the board
        boundary."""
        if zz == 1:
            return (i - 1, j) if x == i and i > 0 else ((i + 1, j) if x != i and i < self.m - 1 else None)
        return (i, j - 1) if y == j and j > 0 else ((i, j + 1) if y != j and j < self.n - 1 else None)

    def _opening_edge(self, comp):
        """Pick the edge to draw to sacrifice `comp`: for a chain this
        must be an outer edge of one of its endpoint boxes (opening a
        chain from the middle splits it into two chains instead of
        offering the whole thing), for a loop any undrawn edge works."""
        boxes   = comp['boxes']
        box_set = set(boxes)
        candidates = comp['ends'] if comp['kind'] == 'chain' and comp['ends'] else boxes
        for (i, j) in candidates:
            for zz, x, y in self._box_edges(i, j):
                if self._edge_drawn(zz, x, y):
                    continue
                other = self._edge_other_box(zz, x, y, i, j)
                if other is not None and other in box_set:
                    continue  # internal edge -- would split the chain, not open it
                return (zz, x, y)
        for (i, j) in boxes:
            for zz, x, y in self._box_edges(i, j):
                if not self._edge_drawn(zz, x, y):
                    return (zz, x, y)
        return None

    def _smart_sacrifice(self) -> bool:
        """When we're out of safe/singleton/doubleton moves and must open
        a fresh chain for the opponent, use the exact solver to pick
        which component to sacrifice (falling back to shortest-chain-
        first if the position isn't cleanly solvable in budget), then
        open it from a proper end rather than lexicographically-first
        free edge."""
        comps = self._detect_components(self.hedge, self.vedge, self.box)
        openable = [c for c in comps if c['kind'] in ('chain', 'loop')]
        if not openable:
            return False
        target_key = None
        if not any(c['kind'] == 'complex' for c in comps) and len(openable) <= self._EG_MAX_COMPONENTS:
            multiset = self._components_multiset(openable)
            self._eg_memo = {}
            self._eg_calls = 0
            try:
                target_key = self._endgame_best_open(multiset)
            except _EndgameSolverTimeout:
                target_key = None
        if target_key is not None:
            t_len, t_loop = target_key
            candidates = [c for c in openable
                          if c['length'] == t_len and (c['kind'] == 'loop') == t_loop]
        else:
            chains_only = [c for c in openable if c['kind'] == 'chain']
            pool = chains_only if chains_only else openable
            min_len = min(c['length'] for c in pool)
            candidates = [c for c in pool if c['length'] == min_len]
        if not candidates:
            return False
        comp = random.choice(candidates)
        move = self._opening_edge(comp)
        if move is None:
            return False
        self.takeedge(*move)
        return True

    def _box_neighbors(self, i: int, j: int):
        """Yield (ni, nj, edge_is_drawn) for the (at most 4) box neighbours
        of box (i, j), where edge_is_drawn reflects the actual line that
        separates the two boxes -- NOT just their box values. This is what
        makes chain detection correct: two boxes both having <=2 open
        sides are only chain-connected if the specific edge between them
        is still undrawn."""
        if i > 0:          yield (i - 1, j, self.hedge_ref[i][j]     != 0)
        if i < self.m - 1: yield (i + 1, j, self.hedge_ref[i + 1][j] != 0)
        if j > 0:          yield (i, j - 1, self.vedge_ref[i][j]     != 0)
        if j < self.n - 1: yield (i, j + 1, self.vedge_ref[i][j + 1] != 0)

    def _detect_components(self, h: list, v: list, b: list):
        # A box is part of the emerging chain/loop skeleton once it has at
        # most 2 undrawn sides left (box value >= 2) -- not only once it is
        # fully capturable (value == 3, as the old version required). Two
        # such boxes are chain-connected iff the specific edge between them
        # is still undrawn, using the real h/v edge state passed in.
        #
        # Exact three-way classification by edges vs. nodes in each
        # component (instead of the old `edge_count >= len(comp)` test,
        # which lumped genuine loops together with branching/joined
        # structures): a simple path (chain) has edges == nodes - 1; a
        # simple cycle (loop) has edges == nodes; anything with MORE edges
        # than nodes is a branching complex (e.g. two chains fused at a
        # junction box) and behaves differently in the endgame than either.
        self.hedge_ref, self.vedge_ref = h, v
        cap     = {(i, j) for i in range(self.m) for j in range(self.n) if 2 <= b[i][j] < 4}
        visited = set()
        comps   = []
        for start in cap:
            if start in visited:
                continue
            comp   = []
            stack  = [start]
            visited.add(start)
            degree = {}
            edge_count = 0
            while stack:
                ci, cj = stack.pop()
                comp.append((ci, cj))
                deg = 0
                for ni, nj, drawn in self._box_neighbors(ci, cj):
                    if drawn or (ni, nj) not in cap:
                        continue
                    deg += 1
                    edge_count += 1
                    if (ni, nj) not in visited:
                        visited.add((ni, nj)); stack.append((ni, nj))
                degree[(ci, cj)] = deg
            edge_count //= 2  # each connecting edge was seen from both sides
            n_nodes = len(comp)
            if edge_count == n_nodes - 1:
                kind = 'chain'
            elif edge_count == n_nodes and n_nodes >= 4:
                kind = 'loop'
            else:
                kind = 'complex'
            entries = [box for box in comp if b[box[0]][box[1]] == 3]
            ends    = [box for box in comp if degree[box] <= 1]
            comps.append({
                'boxes': comp, 'length': n_nodes, 'kind': kind,
                'entries': entries, 'ends': ends,
            })
        return comps

    def _find_chains(self, h: list, v: list, b: list):
        comps        = self._detect_components(h, v, b)
        lengths      = [c['length'] for c in comps if c['kind'] == 'chain']
        loop_count   = sum(1 for c in comps if c['kind'] == 'loop')
        complex_lens = [c['length'] for c in comps if c['kind'] == 'complex']
        return lengths, loop_count, complex_lens

    def _is_safehedge(self, i: int, j: int, b: list) -> bool:
        if i == 0:      return b[i][j] < 2
        if i == self.m: return b[i - 1][j] < 2
        return b[i][j] < 2 and b[i - 1][j] < 2

    def _is_safevedge(self, i: int, j: int, b: list) -> bool:
        if j == 0:      return b[i][j] < 2
        if j == self.n: return b[i][j - 1] < 2
        return b[i][j] < 2 and b[i][j - 1] < 2

    def best_singleton(self) -> bool:
        best_sc = best_args = None
        for i in range(self.m):
            for j in range(self.n):
                if self.box[i][j] != 2:
                    continue
                escape = []
                if self.hedge[i][j]   < 1 and (i < 1       or self.box[i - 1][j] < 2):
                    escape.append((1, i,     j    ))
                if self.vedge[i][j]   < 1 and (j < 1       or self.box[i][j - 1] < 2):
                    escape.append((2, i,     j    ))
                if self.vedge[i][j+1] < 1 and (j+1==self.n or self.box[i][j + 1] < 2):
                    escape.append((2, i,     j + 1))
                if self.hedge[i+1][j] < 1 and (i+1==self.m or self.box[i + 1][j] < 2):
                    escape.append((1, i + 1, j    ))
                if len(escape) < 2:
                    continue
                for cand in escape:
                    if time.time() > self.deadline:
                        break
                    sc = self._score_safe_move(*cand)
                    if best_sc is None or sc > best_sc:
                        best_sc = sc; best_args = cand
        if best_args is None:
            return False
        self.zz, self.x, self.y = best_args
        return True

    def best_doubleton(self) -> bool:
        best_sc = best_args = None
        for i in range(self.m):
            for j in range(self.n - 1):
                if time.time() > self.deadline:
                    break
                if (self.box[i][j] == 2 and self.box[i][j + 1] == 2
                        and self.vedge[i][j + 1] < 1
                        and self.ldub(i, j) and self.rdub(i, j + 1)):
                    sc = self._score_safe_move(2, i, j + 1)
                    if best_sc is None or sc > best_sc:
                        best_sc = sc; best_args = (2, i, j + 1)
        for j in range(self.n):
            for i in range(self.m - 1):
                if time.time() > self.deadline:
                    break
                if (self.box[i][j] == 2 and self.box[i + 1][j] == 2
                        and self.hedge[i + 1][j] < 1
                        and self.udub(i, j) and self.ddub(i + 1, j)):
                    sc = self._score_safe_move(1, i + 1, j)
                    if best_sc is None or sc > best_sc:
                        best_sc = sc; best_args = (1, i + 1, j)
        if best_args is None:
            return False
        self.zz, self.x, self.y = best_args
        return True

    def sethedge(self, x, y):
        self.hedge[x][y] = 1
        if x > 0:      self.box[x - 1][y] += 1
        if x < self.m: self.box[x][y]     += 1
        self.move_queue.append(x * self.n + y)
        self.checkh(x, y)
        self.player = 1 - self.player

    def setvedge(self, x, y):
        self.vedge[x][y] = 1
        if y > 0:      self.box[x][y - 1] += 1
        if y < self.n: self.box[x][y]      += 1
        self.move_queue.append(int(self.N_LINES / 2) + y * self.size + x)
        self.checkv(x, y)
        self.player = 1 - self.player

    def checkh(self, x, y):
        hit = 0
        if x > 0      and self.box[x - 1][y] == 4: hit = 1
        if x < self.m and self.box[x][y]     == 4: hit = 1
        if hit: self.player = 1 - self.player

    def checkv(self, x, y):
        hit = 0
        if y > 0      and self.box[x][y - 1] == 4: hit = 1
        if y < self.n and self.box[x][y]     == 4: hit = 1
        if hit: self.player = 1 - self.player

    def takeedge(self, zz, x, y):
        if zz > 1: self.setvedge(x, y)
        else:      self.sethedge(x, y)

    def takesafe3s(self):
        for i in range(self.m):
            for j in range(self.n):
                if self.box[i][j] == 3:
                    if self.vedge[i][j] < 1:
                        if j == 0 or self.box[i][j - 1] != 2: self.setvedge(i, j)
                    elif self.hedge[i][j] < 1:
                        if i == 0 or self.box[i - 1][j] != 2: self.sethedge(i, j)
                    elif self.vedge[i][j + 1] < 1:
                        if j == self.n - 1 or self.box[i][j + 1] != 2: self.setvedge(i, j + 1)
                    else:
                        if i == self.m - 1 or self.box[i + 1][j] != 2: self.sethedge(i + 1, j)

    def sides3(self):
        for i in range(self.m):
            for j in range(self.n):
                if self.box[i][j] == 3:
                    self.u = i; self.v = j; return True
        return False

    def takeall3s(self):
        while self.sides3():
            self.takebox(self.u, self.v)

    def safehedge(self, i, j):
        if self.hedge[i][j] < 1:
            if i == 0:
                if self.box[i][j] < 2: return True
            elif i == self.m:
                if self.box[i - 1][j] < 2: return True
            elif self.box[i][j] < 2 and self.box[i - 1][j] < 2: return True
        return False

    def safevedge(self, i, j):
        if self.vedge[i][j] < 1:
            if j == 0:
                if self.box[i][j] < 2: return True
            elif j == self.n:
                if self.box[i][j - 1] < 2: return True
            elif self.box[i][j] < 2 and self.box[i][j - 1] < 2: return True
        return False

    def randhedge(self, i, j):
        x, y = i, j
        while True:
            if self.safehedge(x, y):
                self.x, self.y = x, y; return True
            y += 1
            if y == self.n:
                y = 0; x += 1
                if x > self.m: x = 0
            if x == i and y == j: break
        return False

    def randvedge(self, i, j):
        x, y = i, j
        while True:
            if self.safevedge(x, y):
                self.x, self.y = x, y; return True
            y += 1
            if y > self.n:
                y = 0; x += 1
                if x == self.m: x = 0
            if x == i and y == j: break
        return False

    def ldub(self, i, j):
        if self.vedge[i][j] < 1:
            if j < 1 or self.box[i][j - 1] < 2: return True
        elif self.hedge[i][j] < 1:
            if i < 1 or self.box[i - 1][j] < 2: return True
        elif i == self.m - 1 or self.box[i + 1][j] < 2: return True
        return False

    def rdub(self, i, j):
        if self.vedge[i][j + 1] < 1:
            if j + 1 == self.n or self.box[i][j + 1] < 2: return True
        elif self.hedge[i][j] < 1:
            if i < 1 or self.box[i - 1][j] < 2: return True
        elif i + 1 == self.m or self.box[i + 1][j] < 2: return True
        return False

    def udub(self, i, j):
        if self.hedge[i][j] < 1:
            if i < 1 or self.box[i - 1][j] < 2: return True
        elif self.vedge[i][j] < 1:
            if j < 1 or self.box[i][j - 1] < 2: return True
        elif j == self.n - 1 or self.box[i][j + 1] < 2: return True
        return False

    def ddub(self, i, j):
        if self.hedge[i + 1][j] < 1:
            if i == self.m - 1 or self.box[i + 1][j] < 2: return True
        elif self.vedge[i][j] < 1:
            if j < 1 or self.box[i][j - 1] < 2: return True
        elif j == self.n - 1 or self.box[i][j + 1] < 2: return True
        return False

    def _box_neighbors_live(self, i: int, j: int):
        if i > 0:          yield (i - 1, j, self.hedge[i][j]     != 0)
        if i < self.m - 1: yield (i + 1, j, self.hedge[i + 1][j] != 0)
        if j > 0:          yield (i, j - 1, self.vedge[i][j]     != 0)
        if j < self.n - 1: yield (i, j + 1, self.vedge[i][j + 1] != 0)

    def _pick_sac_chain(self):
        """Among all currently-open chains/loops (connected via undrawn
        edges) that contain at least one immediately-capturable (3-sided)
        box, return the entry box belonging to the SHORTEST one. Standard
        double-cross theory says: when you must open a chain, open the
        shortest one first and save longer chains/loops for later, since
        whoever opens the *last* chain hands over control of the endgame."""
        cap = {(i, j) for i in range(self.m) for j in range(self.n) if self.box[i][j] >= 2}
        visited   = set()
        best_box  = None
        best_size = None
        for i in range(self.m):
            for j in range(self.n):
                if self.box[i][j] != 3 or (i, j) in visited:
                    continue
                comp = [(i, j)]
                seen_local = {(i, j)}
                stack = [(i, j)]
                while stack:
                    ci, cj = stack.pop()
                    for ni, nj, drawn in self._box_neighbors_live(ci, cj):
                        if drawn or (ni, nj) not in cap or (ni, nj) in seen_local:
                            continue
                        seen_local.add((ni, nj))
                        comp.append((ni, nj))
                        stack.append((ni, nj))
                visited |= seen_local
                size = len(comp)
                if best_size is None or size < best_size:
                    best_size = size
                    best_box  = (i, j)
        return best_box

    def sac(self, i, j):
        self.count = 0
        self.loop  = False
        self.incount(0, i, j)
        if not self.loop:
            self.takeallbut(i, j)
        boxes_taken = sum(1 for r in range(self.m) for c in range(self.n) if self.box[r][c] == 4)
        if self.count + boxes_taken == self.m * self.n:
            self.takeall3s()
        elif self._should_take_all_chain(i, j):
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
                    self.count += 1; self.loop = True
                elif self.box[i][j - 1] > 1: self.incount(3, i, j - 1)
        elif k != 2 and self.hedge[i][j] < 1:
            if i > 0:
                if self.box[i - 1][j] > 2:
                    self.count += 1; self.loop = True
                elif self.box[i - 1][j] > 1: self.incount(4, i - 1, j)
        elif k != 3 and self.vedge[i][j + 1] < 1:
            if j < self.n - 1:
                if self.box[i][j + 1] > 2:
                    self.count += 1; self.loop = True
                elif self.box[i][j + 1] > 1: self.incount(1, i, j + 1)
        elif k != 4 and self.hedge[i + 1][j] < 1:
            if i < self.m - 1:
                if self.box[i + 1][j] > 2:
                    self.count += 1; self.loop = True
                elif self.box[i + 1][j] > 1: self.incount(2, i + 1, j)

    def takeallbut(self, x, y):
        while self.sides3not(x, y):
            self.takebox(self.u, self.v)

    def sides3not(self, x, y):
        for i in range(self.m):
            for j in range(self.n):
                if self.box[i][j] == 3:
                    if i != x or j != y:
                        self.u, self.v = i, j; return True
        return False

    def takebox(self, i, j):
        if   self.hedge[i][j]     < 1: self.sethedge(i, j)
        elif self.vedge[i][j]     < 1: self.setvedge(i, j)
        elif self.hedge[i + 1][j] < 1: self.sethedge(i + 1, j)
        else:                           self.setvedge(i, j + 1)

    def outcount(self, k, i, j):
        if self.count > 0:
            if k != 1 and self.vedge[i][j] < 1:
                if self.count != 2: self.setvedge(i, j)
                self.count -= 1
                self.outcount(3, i, j - 1)
            elif k != 2 and self.hedge[i][j] < 1:
                if self.count != 2: self.sethedge(i, j)
                self.count -= 1
                self.outcount(4, i - 1, j)
            elif k != 3 and self.vedge[i][j + 1] < 1:
                if self.count != 2: self.setvedge(i, j + 1)
                self.count -= 1
                self.outcount(1, i, j + 1)
            elif k != 4 and self.hedge[i + 1][j] < 1:
                if self.count != 2: self.sethedge(i + 1, j)
                self.count -= 1
                self.outcount(2, i + 1, j)

    def makeanymove(self):
        if self._smart_sacrifice():
            if self.player == 0:
                self.makemove()
            return
        x = y = -1
        found = False
        for i in range(self.m + 1):
            for j in range(self.n):
                if self.hedge[i][j] < 1:
                    x, y = i, j; found = True; break
            if found: break
        if not found:
            for i in range(self.m):
                for j in range(self.n + 1):
                    if self.vedge[i][j] < 1:
                        x, y = i, j; found = True; break
                if found: break
            if found:
                self.setvedge(x, y)
        else:
            self.sethedge(x, y)
        if not found:
            return
        if self.player == 0:
            self.makemove()


def cg_to_ij(box_str, size):
    j = ord(box_str[0]) - ord('A')
    i = size - int(box_str[1:])
    return i, j


def cg_to_line(box_str, side, size, n_lines):
    i, j = cg_to_ij(box_str, size)
    if side == 'T': return i * size + j
    if side == 'B': return (i + 1) * size + j
    if side == 'L': return int(n_lines / 2) + j * size + i
    if side == 'R': return int(n_lines / 2) + (j + 1) * size + i
    raise ValueError(f"Unknown side: {side}")


def line_to_cg(line_index, size, n_lines):
    num_h = int(n_lines / 2)
    if line_index < num_h:
        i = line_index // size
        j = line_index % size
        if i < size:
            box_i, box_j, side = i, j, 'T'
        else:
            box_i, box_j, side = i - 1, j, 'B'
    else:
        v_idx = line_index - num_h
        j = v_idx // size
        i = v_idx % size
        if j < size:
            box_i, box_j, side = i, j, 'L'
        else:
            box_i, box_j, side = i, j - 1, 'R'
    col_char = chr(ord('A') + box_j)
    row_str  = str(size - box_i)
    return f"{col_char}{row_str} {side}"


def run_codingame(agent_type):
    board_size = int(input())
    player_id  = input()
    agent = UCLABot_v4(name="UCLABot_v4")
    while True:
        try:
            line_in = input()
        except EOFError:
            break
        player_score, opponent_score = [int(x) for x in line_in.split()]
        num_boxes = int(input())
        s = DotsAndBoxesGame(size=board_size, starting_player=1)
        playable_lines: List[int] = []
        for _ in range(num_boxes):
            box, sides = input().split()
            for side in sides:
                l_idx = cg_to_line(box, side, board_size, s.N_LINES)
                playable_lines.append(l_idx)
        playable_set = set(playable_lines)
        for l_idx in range(s.N_LINES):
            if l_idx not in playable_set:
                s.l[l_idx] = 1.0
        for r in range(s.SIZE):
            for c in range(s.SIZE):
                lines = s.get_lines_of_box((r, c))
                if sum(1 for ln in lines if s.l[ln] != 0) == 4:
                    s.b[r][c] = 1.0
        diff = player_score - opponent_score
        assigned_diff = 0
        for r in range(s.SIZE):
            for c in range(s.SIZE):
                if s.b[r][c] == 1.0:
                    if assigned_diff < diff:
                        s.b[r][c] = 1.0; assigned_diff += 1
                    elif assigned_diff > diff:
                        s.b[r][c] = -1.0; assigned_diff -= 1
                    else:
                        s.b[r][c] = 1.0; assigned_diff += 1
        if (s.b != 0).all():
            s.result = (1 if player_score > opponent_score
                        else (-1 if opponent_score > player_score else 0))
        import time
        t0 = time.time()
        move = agent.get_move(s)
        dt = time.time() - t0
        print(line_to_cg(move, board_size, s.N_LINES) + f" MSG {agent_type} t={dt:.3f}s")


if __name__ == '__main__':
    run_codingame('ucla_v4')