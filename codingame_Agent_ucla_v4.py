import sys
import random
import numpy as np
from typing import List, Tuple


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
        if self.move_queue:
            return self.move_queue.pop(0)
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
        import time
        t0 = time.time()
        time_limit = getattr(self, 'time_limit', 0.090)
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
            if time.time() - t0 > time_limit:
                break
            sc = self._minimax_1_ply_score(cand)
            if best_score is None or sc > best_score:
                best_score = sc
                best_list = [cand]
            elif sc == best_score:
                best_list.append(cand)

        if not best_list:
            best_list = our_safe
        self.zz, self.x, self.y = random.choice(best_list)
        return True

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
            return self._eval_chain_config(b)
        
        random.shuffle(opp_safe)
        min_score = None
        for opp_m in opp_safe[:3]:
            h2 = [row[:] for row in h]
            v2 = [row[:] for row in v]
            b2 = [row[:] for row in b]
            self._apply_edge_copy(opp_m[0], opp_m[1], opp_m[2], h2, v2, b2)
            self._simulate_safe_phase(h2, v2, b2)
            sc = self._eval_chain_config(b2)
            if min_score is None or sc < min_score:
                min_score = sc
        return min_score

    def _score_safe_move(self, zz: int, x: int, y: int) -> float:
        h = [row[:] for row in self.hedge]
        v = [row[:] for row in self.vedge]
        b = [row[:] for row in self.box]
        self._apply_edge_copy(zz, x, y, h, v, b)
        self._simulate_safe_phase(h, v, b)
        return self._eval_chain_config(b)

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

    def _eval_chain_config(self, b: list) -> float:
        danger_2 = capture_3 = 0
        for i in range(self.m):
            for j in range(self.n):
                bv = b[i][j]
                if bv == 2: danger_2  += 1
                elif bv == 3: capture_3 += 1
        chains, loop_count = self._find_chains(b)
        long_chains        = sum(1 for c in chains if c >= 3)
        short_chains       = sum(1 for c in chains if c == 2)
        total_chain_boxes  = sum(chains)
        parity       = (long_chains + loop_count) % 2
        parity_bonus = 30 if parity == 1 else -30
        return float(
              parity_bonus
            - danger_2        * 10
            - capture_3       *  5
            - long_chains     *  4
            - short_chains    *  2
            - total_chain_boxes *  1
        )

    def _find_chains(self, b: list):
        cap     = {(i, j) for i in range(self.m) for j in range(self.n) if b[i][j] == 3}
        visited = set()
        lengths = []
        loops   = 0
        for start in cap:
            if start in visited:
                continue
            comp  = []
            stack = [start]
            visited.add(start)
            while stack:
                ci, cj = stack.pop()
                comp.append((ci, cj))
                for ni, nj in ((ci-1,cj),(ci+1,cj),(ci,cj-1),(ci,cj+1)):
                    if (ni, nj) in cap and (ni, nj) not in visited:
                        visited.add((ni, nj)); stack.append((ni, nj))
            if len(comp) >= 4 and all(
                sum(1 for ni, nj in ((ci-1,cj),(ci+1,cj),(ci,cj-1),(ci,cj+1))
                    if (ni, nj) in cap) >= 2
                for ci, cj in comp
            ):
                loops += 1
            else:
                lengths.append(len(comp))
        return lengths, loops

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
                if (self.box[i][j] == 2 and self.box[i][j + 1] == 2
                        and self.vedge[i][j + 1] < 1
                        and self.ldub(i, j) and self.rdub(i, j + 1)):
                    sc = self._score_safe_move(2, i, j + 1)
                    if best_sc is None or sc > best_sc:
                        best_sc = sc; best_args = (2, i, j + 1)
        for j in range(self.n):
            for i in range(self.m - 1):
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

    def sac(self, i, j):
        self.count = 0
        self.loop  = False
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
    turn_count = 0
    while True:
        try:
            line_in = input()
        except EOFError:
            break
        agent.time_limit = 0.950 if turn_count == 0 else 0.090
        turn_count += 1
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
