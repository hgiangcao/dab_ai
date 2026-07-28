import random
from agent_interface import BaseAgent


class UCLABot_v6(BaseAgent):
    """
    Bitboard port of UCLABot_v3's full chain-rule strategy.

    Decision logic is IDENTICAL to v3 (safe-3rd-edge captures, chain/loop
    detection with double-cross sacrifice, singleton/doubleton avoidance,
    randomized safe-move search, fallback any-move). The only thing that
    changed is the state representation:

      - v3 keeps hedge[][] / vedge[][] / box[][] as Python lists, rebuilt
        with nested loops every get_move() call, with box[][] counts
        incremented by hand inside sethedge/setvedge.

      - v6 keeps a single integer `state_bb` (bit i set <=> line i taken).
        "Is edge (i,j) taken?" becomes a single bit test (hset/vset), and
        "how many edges does box (i,j) have?" becomes
        (state_bb & BOX_MASKS[box]).bit_count() - an O(1) popcount, so
        there's no per-move bookkeeping of box counts at all; they're
        always derived directly and cheaply from state_bb.
    """

    def __init__(self, name="UCLABot_v6", size=5):
        super().__init__(name)
        self.move_queue = []
        self.state_bb = 0
        self._setup(size)

    def _setup(self, size):
        self.size = size
        self.m = size
        self.n = size
        self.n_lines = 2 * size * (size + 1)
        self.half_lines = self.n_lines // 2

        # BOX_MASKS[i*n+j] = bitmask of the 4 edge-line-indices bounding box (i, j)
        self.BOX_MASKS = []
        for r in range(self.size):
            for c in range(self.size):
                top = r * self.size + c
                bottom = (r + 1) * self.size + c
                left = self.half_lines + c * self.size + r
                right = self.half_lines + (c + 1) * self.size + r
                mask = (1 << top) | (1 << bottom) | (1 << left) | (1 << right)
                self.BOX_MASKS.append(mask)

    # ---------------------------------------------------------------- #
    # bitboard helpers (replace hedge[][] / vedge[][] / box[][] lookups)
    # ---------------------------------------------------------------- #

    def hidx(self, i, j):
        return i * self.n + j

    def vidx(self, i, j):
        return self.half_lines + j * self.size + i

    def hset(self, i, j):
        """True if horizontal edge (i, j) is already taken."""
        return (self.state_bb >> self.hidx(i, j)) & 1 != 0

    def vset(self, i, j):
        """True if vertical edge (i, j) is already taken."""
        return (self.state_bb >> self.vidx(i, j)) & 1 != 0

    def box_count(self, i, j):
        """Number of edges already taken around box (i, j)."""
        return (self.state_bb & self.BOX_MASKS[i * self.n + j]).bit_count()

    # ---------------------------------------------------------------- #
    # entry point
    # ---------------------------------------------------------------- #

    def get_move(self, game) -> int:
        if not any(game.l):
            self.move_queue.clear()

        while self.move_queue:
            mv = self.move_queue[0]
            if game.l[mv] != 0:
                self.move_queue.clear()
                break
            return self.move_queue.pop(0)

        # rebuild bitboard sizing if the board dimensions changed
        if len(game.l) != self.n_lines:
            self._setup(game.SIZE)

        self.size = game.SIZE
        self.m = self.size
        self.n = self.size
        self.N_LINES = game.N_LINES

        # pack current line state into the bitboard
        self.state_bb = 0
        for i, val in enumerate(game.l):
            if val != 0:
                self.state_bb |= (1 << i)

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

    # ---------------------------------------------------------------- #
    # move application
    # ---------------------------------------------------------------- #

    def sethedge(self, x, y):
        self.state_bb |= (1 << self.hidx(x, y))
        self.move_queue.append(self.hidx(x, y))
        self.checkh(x, y)
        self.player = 1 - self.player

    def setvedge(self, x, y):
        self.state_bb |= (1 << self.vidx(x, y))
        self.move_queue.append(self.vidx(x, y))
        self.checkv(x, y)
        self.player = 1 - self.player

    def checkh(self, x, y):
        hit = 0
        if x > 0 and self.box_count(x - 1, y) == 4: hit = 1
        if x < self.m and self.box_count(x, y) == 4: hit = 1
        if hit > 0: self.player = 1 - self.player

    def checkv(self, x, y):
        hit = 0
        if y > 0 and self.box_count(x, y - 1) == 4: hit = 1
        if y < self.n and self.box_count(x, y) == 4: hit = 1
        if hit > 0: self.player = 1 - self.player

    def takeedge(self, zz, x, y):
        if zz > 1: self.setvedge(x, y)
        else: self.sethedge(x, y)

    # ---------------------------------------------------------------- #
    # strategy (chain rule) - identical control flow to v3
    # ---------------------------------------------------------------- #

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
                if self.box_count(i, j) == 3:
                    if not self.vset(i, j):
                        if j == 0 or self.box_count(i, j - 1) != 2: self.setvedge(i, j)
                    elif not self.hset(i, j):
                        if i == 0 or self.box_count(i - 1, j) != 2: self.sethedge(i, j)
                    elif not self.vset(i, j + 1):
                        if j == self.n - 1 or self.box_count(i, j + 1) != 2: self.setvedge(i, j + 1)
                    else:
                        if i == self.m - 1 or self.box_count(i + 1, j) != 2: self.sethedge(i + 1, j)

    def sides3(self):
        for i in range(self.m):
            for j in range(self.n):
                if self.box_count(i, j) == 3:
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
        if not self.hset(i, j):
            if i == 0:
                if self.box_count(i, j) < 2: return True
            elif i == self.m:
                if self.box_count(i - 1, j) < 2: return True
            elif self.box_count(i, j) < 2 and self.box_count(i - 1, j) < 2: return True
        return False

    def safevedge(self, i, j):
        if not self.vset(i, j):
            if j == 0:
                if self.box_count(i, j) < 2: return True
            elif j == self.n:
                if self.box_count(i, j - 1) < 2: return True
            elif self.box_count(i, j) < 2 and self.box_count(i, j - 1) < 2: return True
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
                if self.box_count(i, j) == 2:
                    numb = 0
                    if not self.hset(i, j):
                        if i < 1 or self.box_count(i - 1, j) < 2: numb += 1
                    self.zz = 2
                    if not self.vset(i, j):
                        if j < 1 or self.box_count(i, j - 1) < 2: numb += 1
                        if numb > 1:
                            self.x, self.y = i, j
                            return True
                    if not self.vset(i, j + 1):
                        if j + 1 == self.n or self.box_count(i, j + 1) < 2: numb += 1
                        if numb > 1:
                            self.x, self.y = i, j + 1
                            return True
                    self.zz = 1
                    if not self.hset(i + 1, j):
                        if i + 1 == self.m or self.box_count(i + 1, j) < 2: numb += 1
                        if numb > 1:
                            self.x, self.y = i + 1, j
                            return True
        return False

    def doubleton(self):
        self.zz = 2
        for i in range(self.m):
            for j in range(self.n - 1):
                if self.box_count(i, j) == 2 and self.box_count(i, j + 1) == 2 and not self.vset(i, j + 1):
                    if self.ldub(i, j) and self.rdub(i, j + 1):
                        self.x, self.y = i, j + 1
                        return True
        self.zz = 1
        for j in range(self.n):
            for i in range(self.m - 1):
                if self.box_count(i, j) == 2 and self.box_count(i + 1, j) == 2 and not self.hset(i + 1, j):
                    if self.udub(i, j) and self.ddub(i + 1, j):
                        self.x, self.y = i + 1, j
                        return True
        return False

    def ldub(self, i, j):
        if not self.vset(i, j):
            if j < 1 or self.box_count(i, j - 1) < 2: return True
        elif not self.hset(i, j):
            if i < 1 or self.box_count(i - 1, j) < 2: return True
        elif i == self.m - 1 or self.box_count(i + 1, j) < 2: return True
        return False

    def rdub(self, i, j):
        if not self.vset(i, j + 1):
            if j + 1 == self.n or self.box_count(i, j + 1) < 2: return True
        elif not self.hset(i, j):
            if i < 1 or self.box_count(i - 1, j) < 2: return True
        elif i + 1 == self.m or self.box_count(i + 1, j) < 2: return True
        return False

    def udub(self, i, j):
        if not self.hset(i, j):
            if i < 1 or self.box_count(i - 1, j) < 2: return True
        elif not self.vset(i, j):
            if j < 1 or self.box_count(i, j - 1) < 2: return True
        elif j == self.n - 1 or self.box_count(i, j + 1) < 2: return True
        return False

    def ddub(self, i, j):
        if not self.hset(i + 1, j):
            if i == self.m - 1 or self.box_count(i + 1, j) < 2: return True
        elif not self.vset(i, j):
            if j < 1 or self.box_count(i, j - 1) < 2: return True
        elif j == self.n - 1 or self.box_count(i, j + 1) < 2: return True
        return False

    def sac(self, i, j):
        self.count = 0
        self.loop = False
        self.incount(0, i, j)
        if not self.loop: self.takeallbut(i, j)
        boxes_taken = sum(1 for r in range(self.m) for c in range(self.n) if self.box_count(r, c) == 4)
        if self.count + boxes_taken == self.m * self.n:
            self.takeall3s()
        else:
            if self.loop:
                self.count -= 2
            self.outcount(0, i, j)

    def incount(self, k, i, j):
        self.count += 1
        if k != 1 and not self.vset(i, j):
            if j > 0:
                if self.box_count(i, j - 1) > 2:
                    self.count += 1
                    self.loop = True
                elif self.box_count(i, j - 1) > 1: self.incount(3, i, j - 1)
        elif k != 2 and not self.hset(i, j):
            if i > 0:
                if self.box_count(i - 1, j) > 2:
                    self.count += 1
                    self.loop = True
                elif self.box_count(i - 1, j) > 1: self.incount(4, i - 1, j)
        elif k != 3 and not self.vset(i, j + 1):
            if j < self.n - 1:
                if self.box_count(i, j + 1) > 2:
                    self.count += 1
                    self.loop = True
                elif self.box_count(i, j + 1) > 1: self.incount(1, i, j + 1)
        elif k != 4 and not self.hset(i + 1, j):
            if i < self.m - 1:
                if self.box_count(i + 1, j) > 2:
                    self.count += 1
                    self.loop = True
                elif self.box_count(i + 1, j) > 1: self.incount(2, i + 1, j)

    def takeallbut(self, x, y):
        while self.sides3not(x, y):
            self.takebox(self.u, self.v)

    def sides3not(self, x, y):
        for i in range(self.m):
            for j in range(self.n):
                if self.box_count(i, j) == 3:
                    if i != x or j != y:
                        self.u, self.v = i, j
                        return True
        return False

    def takebox(self, i, j):
        if not self.hset(i, j): self.sethedge(i, j)
        elif not self.vset(i, j): self.setvedge(i, j)
        elif not self.hset(i + 1, j): self.sethedge(i + 1, j)
        else: self.setvedge(i, j + 1)

    def outcount(self, k, i, j):
        if self.count > 0:
            if k != 1 and not self.vset(i, j):
                if self.count != 2: self.setvedge(i, j)
                self.count -= 1
                self.outcount(3, i, j - 1)
            elif k != 2 and not self.hset(i, j):
                if self.count != 2: self.sethedge(i, j)
                self.count -= 1
                self.outcount(4, i - 1, j)
            elif k != 3 and not self.vset(i, j + 1):
                if self.count != 2: self.setvedge(i, j + 1)
                self.count -= 1
                self.outcount(1, i, j + 1)
            elif k != 4 and not self.hset(i + 1, j):
                if self.count != 2: self.sethedge(i + 1, j)
                self.count -= 1
                self.outcount(2, i + 1, j)

    def makeanymove(self):
        x = -1
        y = -1
        found = False
        for i in range(self.m + 1):
            for j in range(self.n):
                if not self.hset(i, j):
                    x, y = i, j
                    found = True
                    break
            if found: break

        if not found:
            for i in range(self.m):
                for j in range(self.n + 1):
                    if not self.vset(i, j):
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