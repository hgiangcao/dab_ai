import sys
import math
import random
import time
import numpy as np
from typing import Tuple, List


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


class Node:
    __slots__ = ('parent', 'move', 'children', 'visits', 'value_sum',
                 'player_to_move', 'untried_moves', 'terminal', 'bias')

    def __init__(self, parent, move, player_to_move, valid_moves, terminal, bias=0.0):
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.player_to_move = player_to_move
        self.terminal = terminal
        self.untried_moves = valid_moves if valid_moves is not None else []
        self.bias = bias


class MCTS:
    def __init__(self, n_simulations: int = 100, time_limit: float = None, c_puct: float = 1.4):
        self.n_simulations = n_simulations
        self.time_limit = time_limit
        self.c_puct = c_puct

    def search(self, game_state):
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None, 0, 0, 0.0

        root = Node(
            parent=None, move=None,
            player_to_move=game_state.current_player,
            valid_moves=self._order_expansion_moves(game_state, valid_moves),
            terminal=not game_state.is_running()
        )

        simulations_done = 0
        max_tree_depth = 0
        start_time = time.time()

        while True:
            if self.time_limit is not None:
                if time.time() - start_time >= self.time_limit:
                    break
            elif simulations_done >= self.n_simulations:
                break

            simulations_done += 1
            node = root
            moves_made = 0

            while not node.untried_moves and node.children:
                node = self._select_child(node)
                game_state.execute_move(node.move)
                moves_made += 1

            if node.untried_moves:
                move = node.untried_moves.pop(0)
                bias = self._evaluate_bias(game_state, move)
                game_state.execute_move(move)
                moves_made += 1
                terminal = not game_state.is_running()
                valid_for_child = (self._order_expansion_moves(game_state, game_state.get_valid_moves())
                                   if not terminal else [])
                child = Node(parent=node, move=move,
                             player_to_move=game_state.current_player,
                             valid_moves=valid_for_child,
                             terminal=terminal, bias=bias)
                node.children.append(child)
                node = child

            max_tree_depth = max(max_tree_depth, moves_made)

            rollout_player = node.player_to_move
            val, rollout_moves = self._rollout(game_state, rollout_player)
            moves_made += rollout_moves

            curr = node
            while curr is not None:
                curr.visits += 1
                curr.value_sum += val
                if curr.parent is not None:
                    if curr.parent.player_to_move != curr.player_to_move:
                        val = -val
                curr = curr.parent

            for _ in range(moves_made):
                game_state.undo_move()

        total_time = time.time() - start_time
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            return best_child.move, max_tree_depth, simulations_done, total_time
        return None, 0, 0, 0.0

    def _select_child(self, node):
        best_score = -float('inf')
        best_child = None
        log_n = math.log(node.visits) if node.visits > 0 else 0
        for child in node.children:
            if child.visits == 0:
                ucb = float('inf')
            else:
                q = child.value_sum / child.visits
                if node.player_to_move != child.player_to_move:
                    q = -q
                ucb = q + self.c_puct * math.sqrt(log_n / child.visits) + (child.bias / (child.visits + 1))
            if ucb > best_score:
                best_score = ucb
                best_child = child
        return best_child

    def _rollout(self, s, rollout_player):
        moves_made = 0
        while s.is_running():
            valid = s.get_valid_moves()
            move = self._find_capture(s, valid)
            if move is None:
                best_score = -float('inf')
                best_moves = []
                for m in valid:
                    score = self._score_rollout_move(s, m)
                    if score > best_score:
                        best_score = score
                        best_moves = [m]
                    elif score == best_score:
                        best_moves.append(m)
                move = random.choice(best_moves)
            s.execute_move(move)
            moves_made += 1
        p1 = int(np.sum(s.b == 1))
        p2 = int(np.sum(s.b == -1))
        my = p1 if rollout_player == 1 else p2
        opp = p2 if rollout_player == 1 else p1
        val = float(my - opp) / (s.SIZE * s.SIZE)
        return val, moves_made

    def _score_rollout_move(self, s, move):
        score = 0
        is_capture = False
        is_safe = True
        for box in s.get_boxes_of_line(move):
            lines = s.get_lines_of_box(box)
            drawn = sum(1 for ln in lines if s.l[ln] != 0)
            if drawn == 3:
                is_capture = True
            if drawn == 2:
                is_safe = False
        if is_capture:
            score += 100
        elif is_safe:
            score += 30 if len(s.get_boxes_of_line(move)) == 1 else 20
        else:
            score -= 50
        return score

    def _evaluate_bias(self, s, move):
        return self._score_rollout_move(s, move) / 100.0

    def _find_capture(self, s, valid_moves):
        for move in valid_moves:
            for box in s.get_boxes_of_line(move):
                lines = s.get_lines_of_box(box)
                if sum(1 for ln in lines if s.l[ln] != 0) == 3:
                    return move
        return None

    def _order_expansion_moves(self, s, valid_moves):
        if not valid_moves:
            return []
        captures, safe_edge, safe_center, dangerous = [], [], [], []
        for move in valid_moves:
            is_capture = False
            is_safe = True
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
        random.shuffle(captures)
        random.shuffle(safe_edge)
        random.shuffle(safe_center)
        random.shuffle(dangerous)
        return captures + safe_edge + safe_center + dangerous


class UCLABot_v3(BaseAgent):
    def __init__(self, name: str = "UCLABot_v3"):
        super().__init__(name)
        self.move_queue: List[int] = []

    def get_move(self, game) -> int:
        import numpy as _np2
        if int(_np2.count_nonzero(game.l)) == 0:
            self.move_queue.clear()
        if self.move_queue:
            mv = self.move_queue[0]
            if game.l[mv] != 0:
                self.move_queue.clear()
            else:
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
        valid = game.get_valid_moves()
        return valid[0] if valid else -1

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
                self.x, self.y = x, y
                return True
            y += 1
            if y == self.n:
                y = 0
                x += 1
                if x > self.m: x = 0
            if x == i and y == j: break
        return False

    def randvedge(self, i, j):
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

    def singleton(self):
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

    def doubleton(self):
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


class UCLAMCTSEngine(MCTS):
    def _rollout(self, s, rollout_player):
        moves_made = 0
        agent = UCLABot_v3()
        while s.is_running():
            valid = s.get_valid_moves()
            if not valid:
                break
            move = agent.get_move(s)
            if move is None or move not in valid:
                move = valid[0]
            s.execute_move(move)
            moves_made += 1
        p1 = int(np.sum(s.b == 1))
        p2 = int(np.sum(s.b == -1))
        my  = p1 if rollout_player == 1 else p2
        opp = p2 if rollout_player == 1 else p1
        val = float(my - opp) / max(s.SIZE * s.SIZE, 1)
        return val, moves_made

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
                if drawn == 3: is_capture = True
                if drawn == 2: is_safe = False
            if is_capture:
                captures.append(move)
            elif is_safe:
                if len(s.get_boxes_of_line(move)) == 1:
                    safe_edge.append(move)
                else:
                    safe_center.append(move)
            else:
                dangerous.append(move)
        random.shuffle(captures)
        random.shuffle(safe_edge)
        random.shuffle(safe_center)
        random.shuffle(dangerous)
        return captures + safe_edge + safe_center + dangerous

    def _evaluate_bias(self, s, move):
        for box in s.get_boxes_of_line(move):
            lines = s.get_lines_of_box(box)
            drawn = sum(1 for ln in lines if s.l[ln] != 0)
            if drawn == 3:
                return 1.0
        is_safe = True
        for box in s.get_boxes_of_line(move):
            lines = s.get_lines_of_box(box)
            drawn = sum(1 for ln in lines if s.l[ln] != 0)
            if drawn == 2:
                is_safe = False
                break
        if is_safe:
            return 0.4 if len(s.get_boxes_of_line(move)) == 1 else 0.3
        return -0.3


class UCLAMCTSBot(BaseAgent):
    def __init__(self, name="UCLAMCTSBot", time_limit=0.09, first_turn_time=0.95, c_puct=1.4):
        super().__init__(name)
        self.time_limit      = time_limit
        self.first_turn_time = first_turn_time
        self.c_puct          = c_puct
        self._first_turn     = True
        self._ucla           = UCLABot_v3()
        self.last_simuls = 0
        self.last_depth  = 0
        self.last_time   = 0.0

    def get_move(self, game) -> int:
        if self._ucla.move_queue:
            return self._ucla.move_queue.pop(0)
        if self._has_captures(game):
            self.last_simuls = 0
            self.last_depth  = 0
            self.last_time   = 0.0
            return self._ucla.get_move(game)
        tl = self.first_turn_time if self._first_turn else self.time_limit
        self._first_turn = False
        engine = UCLAMCTSEngine(time_limit=tl, c_puct=self.c_puct)
        best_move, depth, simuls, t = engine.search(game)
        self.last_simuls = simuls
        self.last_depth  = depth
        self.last_time   = t
        if best_move is not None:
            return best_move
        valid = game.get_valid_moves()
        return valid[0] if valid else -1

    def _has_captures(self, game) -> bool:
        for r in range(game.SIZE):
            for c in range(game.SIZE):
                if game.b[r][c] == 0:
                    lines = game.get_lines_of_box((r, c))
                    if sum(1 for ln in lines if game.l[ln] != 0) == 3:
                        return True
        return False


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
    agent = UCLAMCTSBot(name="UCLAMCTSBot")
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
        print(line_to_cg(move, board_size, s.N_LINES) + f" MSG {agent_type} sims={agent.last_simuls} t={agent.last_time:.3f}s")


if __name__ == '__main__':
    run_codingame('mcts_ucla_v3')
