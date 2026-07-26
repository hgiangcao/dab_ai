import sys
import math
import random
import time
import numpy as np
from typing import Tuple, List

# ====================================================================
# BASE AGENT
# ====================================================================

class BaseAgent:
    def __init__(self, name=""):
        self.name = name

# ====================================================================
# DOTS AND BOXES GAME
# ====================================================================

class DotsAndBoxesGame:
    """
    Implementation of the Dots-and-Boxes game.

    Line indexing (SIZE=2 example):
        +  0 +  1 +
        6    8   10
        +  2 +  3 +
        7    9   11
        +  4 +  5 +

    Horizontal lines: indices 0 .. N_LINES/2 - 1
    Vertical lines:   indices N_LINES/2 .. N_LINES - 1
    """

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
        assert self.l[line] == 0, "line is already drawn"
        self.l[line] = self.current_player

    def switch_current_player(self):
        self.current_player *= -1

    def capture_box(self, row: int, col: int):
        assert self.b[row][col] == 0, "box is already captured"
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
        if not self._history:
            raise RuntimeError("No move to undo")
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
            if i == 0:          return [(i, j)]
            elif i == self.SIZE: return [(i - 1, j)]
            else:               return [(i - 1, j), (i, j)]
        else:
            line = line - int(self.N_LINES / 2)
            j = line // self.SIZE
            i = line % self.SIZE
            if j == 0:          return [(i, j)]
            elif j == self.SIZE: return [(i, j - 1)]
            else:               return [(i, j - 1), (i, j)]

    def get_lines_of_box(self, box: Tuple[int, int]) -> List[int]:
        i, j = box
        line_top    = i * self.SIZE + j
        line_bottom = (i + 1) * self.SIZE + j
        line_left   = int(self.N_LINES / 2) + j * self.SIZE + i
        line_right  = int(self.N_LINES / 2) + (j + 1) * self.SIZE + i
        return [line_top, line_bottom, line_left, line_right]


# ====================================================================
# UCLA BOT V3
# ====================================================================

class UCLABot_v3(BaseAgent):
    """
    Strategy:
    1. Take all safe immediately capturable boxes.
    2. If 3-sided boxes remain:
       - If safe moves exist: take all 3-sided boxes then play safe.
       - Else: sacrifice (double-cross) to keep control.
    3. Play safe move (singleton / doubleton heuristic).
    4. Fall back to any free edge.

    The algorithm may queue multiple moves (chain captures) per call.
    The CodingGame loop consumes exactly one move per turn.
    """

    def __init__(self, name: str = "UCLABot_v3"):
        super().__init__(name)
        self.move_queue: List[int] = []

    def get_move(self, game: DotsAndBoxesGame) -> int:
        if self.move_queue:
            return self.move_queue.pop(0)

        self.size = game.SIZE
        self.m = self.size
        self.n = self.size
        self.N_LINES = game.N_LINES

        self.hedge = [[0] * self.n for _ in range(self.m + 1)]
        self.vedge = [[0] * (self.n + 1) for _ in range(self.m)]
        self.box   = [[0] * self.n for _ in range(self.m)]

        for r in range(self.m + 1):
            for c in range(self.n):
                line_idx = r * self.n + c
                if game.l[line_idx] != 0:
                    self.hedge[r][c] = 1
                    if r > 0:      self.box[r - 1][c] += 1
                    if r < self.m: self.box[r][c]     += 1

        for c in range(self.n + 1):
            for r in range(self.m):
                line_idx = int(self.N_LINES / 2) + c * self.size + r
                if game.l[line_idx] != 0:
                    self.vedge[r][c] = 1
                    if c > 0:      self.box[r][c - 1] += 1
                    if c < self.n: self.box[r][c]      += 1

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
            return valid[0] if valid else -1

    def sethedge(self, x: int, y: int):
        self.hedge[x][y] = 1
        if x > 0:      self.box[x - 1][y] += 1
        if x < self.m: self.box[x][y]     += 1
        self.move_queue.append(x * self.n + y)
        self.checkh(x, y)
        self.player = 1 - self.player

    def setvedge(self, x: int, y: int):
        self.vedge[x][y] = 1
        if y > 0:      self.box[x][y - 1] += 1
        if y < self.n: self.box[x][y]      += 1
        self.move_queue.append(int(self.N_LINES / 2) + y * self.size + x)
        self.checkv(x, y)
        self.player = 1 - self.player

    def checkh(self, x: int, y: int):
        hit = 0
        if x > 0       and self.box[x - 1][y] == 4: hit = 1
        if x < self.m  and self.box[x][y]     == 4: hit = 1
        if hit: self.player = 1 - self.player

    def checkv(self, x: int, y: int):
        hit = 0
        if y > 0       and self.box[x][y - 1] == 4: hit = 1
        if y < self.n  and self.box[x][y]     == 4: hit = 1
        if hit: self.player = 1 - self.player

    def takeedge(self, zz: int, x: int, y: int):
        if zz > 1: self.setvedge(x, y)
        else:      self.sethedge(x, y)

    def makemove(self):
        self.takesafe3s()
        if self.sides3():
            if self.sides01():
                self.takeall3s()
                self.takeedge(self.zz, self.x, self.y)
            else:
                self.sac(self.u, self.v)
        elif self.sides01():   self.takeedge(self.zz, self.x, self.y)
        elif self.singleton(): self.takeedge(self.zz, self.x, self.y)
        elif self.doubleton(): self.takeedge(self.zz, self.x, self.y)
        else:                  self.makeanymove()

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

    def sides3(self) -> bool:
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

    def sides01(self) -> bool:
        self.zz = 1 if random.random() < 0.5 else 2
        i = int(self.m * random.random())
        j = int(self.n * random.random())
        if self.zz == 1:
            if self.randhedge(i, j): return True
            self.zz = 2
            return self.randvedge(i, j)
        else:
            if self.randvedge(i, j): return True
            self.zz = 1
            return self.randhedge(i, j)

    def safehedge(self, i: int, j: int) -> bool:
        if self.hedge[i][j] < 1:
            if i == 0:
                if self.box[i][j] < 2: return True
            elif i == self.m:
                if self.box[i - 1][j] < 2: return True
            elif self.box[i][j] < 2 and self.box[i - 1][j] < 2: return True
        return False

    def safevedge(self, i: int, j: int) -> bool:
        if self.vedge[i][j] < 1:
            if j == 0:
                if self.box[i][j] < 2: return True
            elif j == self.n:
                if self.box[i][j - 1] < 2: return True
            elif self.box[i][j] < 2 and self.box[i][j - 1] < 2: return True
        return False

    def randhedge(self, i: int, j: int) -> bool:
        x, y = i, j
        while True:
            if self.safehedge(x, y):
                self.x, self.y = x, y
                return True
            y += 1
            if y == self.n:
                y = 0
                x += 1
                if x > self.m: x = 0
            if x == i and y == j: break
        return False

    def randvedge(self, i: int, j: int) -> bool:
        x, y = i, j
        while True:
            if self.safevedge(x, y):
                self.x, self.y = x, y
                return True
            y += 1
            if y > self.n:
                y = 0
                x += 1
                if x == self.m: x = 0
            if x == i and y == j: break
        return False

    def singleton(self) -> bool:
        for i in range(self.m):
            for j in range(self.n):
                if self.box[i][j] == 2:
                    numb = 0
                    if self.hedge[i][j] < 1:
                        if i < 1 or self.box[i - 1][j] < 2: numb += 1
                    self.zz = 2
                    if self.vedge[i][j] < 1:
                        if j < 1 or self.box[i][j - 1] < 2: numb += 1
                        if numb > 1:
                            self.x, self.y = i, j
                            return True
                    if self.vedge[i][j + 1] < 1:
                        if j + 1 == self.n or self.box[i][j + 1] < 2: numb += 1
                        if numb > 1:
                            self.x, self.y = i, j + 1
                            return True
                    self.zz = 1
                    if self.hedge[i + 1][j] < 1:
                        if i + 1 == self.m or self.box[i + 1][j] < 2: numb += 1
                        if numb > 1:
                            self.x, self.y = i + 1, j
                            return True
        return False

    def doubleton(self) -> bool:
        self.zz = 2
        for i in range(self.m):
            for j in range(self.n - 1):
                if (self.box[i][j] == 2 and self.box[i][j + 1] == 2
                        and self.vedge[i][j + 1] < 1):
                    if self.ldub(i, j) and self.rdub(i, j + 1):
                        self.x, self.y = i, j + 1
                        return True
        self.zz = 1
        for j in range(self.n):
            for i in range(self.m - 1):
                if (self.box[i][j] == 2 and self.box[i + 1][j] == 2
                        and self.hedge[i + 1][j] < 1):
                    if self.udub(i, j) and self.ddub(i + 1, j):
                        self.x, self.y = i + 1, j
                        return True
        return False

    def ldub(self, i: int, j: int) -> bool:
        if self.vedge[i][j] < 1:
            if j < 1 or self.box[i][j - 1] < 2: return True
        elif self.hedge[i][j] < 1:
            if i < 1 or self.box[i - 1][j] < 2: return True
        elif i == self.m - 1 or self.box[i + 1][j] < 2: return True
        return False

    def rdub(self, i: int, j: int) -> bool:
        if self.vedge[i][j + 1] < 1:
            if j + 1 == self.n or self.box[i][j + 1] < 2: return True
        elif self.hedge[i][j] < 1:
            if i < 1 or self.box[i - 1][j] < 2: return True
        elif i + 1 == self.m or self.box[i + 1][j] < 2: return True
        return False

    def udub(self, i: int, j: int) -> bool:
        if self.hedge[i][j] < 1:
            if i < 1 or self.box[i - 1][j] < 2: return True
        elif self.vedge[i][j] < 1:
            if j < 1 or self.box[i][j - 1] < 2: return True
        elif j == self.n - 1 or self.box[i][j + 1] < 2: return True
        return False

    def ddub(self, i: int, j: int) -> bool:
        if self.hedge[i + 1][j] < 1:
            if i == self.m - 1 or self.box[i + 1][j] < 2: return True
        elif self.vedge[i][j] < 1:
            if j < 1 or self.box[i][j - 1] < 2: return True
        elif j == self.n - 1 or self.box[i][j + 1] < 2: return True
        return False

    def sac(self, i: int, j: int):
        self.count = 0
        self.loop = False
        self.incount(0, i, j)
        if not self.loop:
            self.takeallbut(i, j)
        boxes_taken = sum(
            1 for r in range(self.m) for c in range(self.n)
            if self.box[r][c] == 4
        )
        if self.count + boxes_taken == self.m * self.n:
            self.takeall3s()
        else:
            if self.loop:
                self.count -= 2
            self.outcount(0, i, j)

    def incount(self, k: int, i: int, j: int):
        self.count += 1
        if k != 1 and self.vedge[i][j] < 1:
            if j > 0:
                if self.box[i][j - 1] > 2:
                    self.count += 1
                    self.loop = True
                elif self.box[i][j - 1] > 1: self.incount(3, i, j - 1)
        elif k != 2 and self.hedge[i][j] < 1:
            if i > 0:
                if self.box[i - 1][j] > 2:
                    self.count += 1
                    self.loop = True
                elif self.box[i - 1][j] > 1: self.incount(4, i - 1, j)
        elif k != 3 and self.vedge[i][j + 1] < 1:
            if j < self.n - 1:
                if self.box[i][j + 1] > 2:
                    self.count += 1
                    self.loop = True
                elif self.box[i][j + 1] > 1: self.incount(1, i, j + 1)
        elif k != 4 and self.hedge[i + 1][j] < 1:
            if i < self.m - 1:
                if self.box[i + 1][j] > 2:
                    self.count += 1
                    self.loop = True
                elif self.box[i + 1][j] > 1: self.incount(2, i + 1, j)

    def takeallbut(self, x: int, y: int):
        while self.sides3not(x, y):
            self.takebox(self.u, self.v)

    def sides3not(self, x: int, y: int) -> bool:
        for i in range(self.m):
            for j in range(self.n):
                if self.box[i][j] == 3:
                    if i != x or j != y:
                        self.u, self.v = i, j
                        return True
        return False

    def takebox(self, i: int, j: int):
        if   self.hedge[i][j]     < 1: self.sethedge(i, j)
        elif self.vedge[i][j]     < 1: self.setvedge(i, j)
        elif self.hedge[i + 1][j] < 1: self.sethedge(i + 1, j)
        else:                           self.setvedge(i, j + 1)

    def outcount(self, k: int, i: int, j: int):
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
        for i in range(self.m + 1):
            for j in range(self.n):
                if self.hedge[i][j] < 1:
                    self.sethedge(i, j)
                    return
        for i in range(self.m):
            for j in range(self.n + 1):
                if self.vedge[i][j] < 1:
                    self.setvedge(i, j)
                    return


# ====================================================================
# CODINGAME I/O HELPERS
# ====================================================================

def cg_to_ij(box_str: str, size: int):
    col_char = box_str[0]
    row_char = box_str[1:]
    j = ord(col_char) - ord('A')
    i = size - int(row_char)
    return i, j


def cg_to_line(box_str: str, side: str, size: int, n_lines: int) -> int:
    i, j = cg_to_ij(box_str, size)
    if side == 'T': return i * size + j
    if side == 'B': return (i + 1) * size + j
    if side == 'L': return int(n_lines / 2) + j * size + i
    if side == 'R': return int(n_lines / 2) + (j + 1) * size + i
    raise ValueError(f"Unknown side: {side}")


def line_to_cg(line_index: int, size: int, n_lines: int) -> str:
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


# ====================================================================
# CODINGAME MAIN LOOP
# ====================================================================

def run_codingame(agent_type: str = "ucla_v3"):
    board_size = int(input())
    player_id  = input()   # 'A' or 'B'

    agent = UCLABot_v3(name="UCLABot_v3")

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

        # Reconstruct box ownership
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
                        s.b[r][c] = 1.0
                        assigned_diff += 1
                    elif assigned_diff > diff:
                        s.b[r][c] = -1.0
                        assigned_diff -= 1
                    else:
                        s.b[r][c] = 1.0
                        assigned_diff += 1

        if (s.b != 0).all():
            s.result = (1 if player_score > opponent_score
                        else (-1 if opponent_score > player_score else 0))

        move = agent.get_move(s)

        print(line_to_cg(move, board_size, s.N_LINES) + f" MSG {agent_type}")


if __name__ == '__main__':
    run_codingame('ucla_v3')
