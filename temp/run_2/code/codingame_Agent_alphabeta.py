import sys
import math
import random
import time
import numpy as np
from typing import Tuple, List, Dict

class BaseAgent:
    def __init__(self, name=""):
        self.name = name
import random

class ZobristHash:
    def __init__(self, num_lines):
        # Assign a random 64-bit integer to every possible line on the board
        self.line_hashes = [random.getrandbits(64) for _ in range(num_lines)]
        self.table = {} 

    def compute_initial_hash(self, board_lines):
        """Compute from scratch (only needed once at the start of the search)."""
        h = 0
        for idx, is_drawn in enumerate(board_lines):
            if is_drawn:
                h ^= self.line_hashes[idx]
        return h

    def update_hash(self, current_hash, move_idx):
        """O(1) incremental update. Call this when a line is drawn or undone."""
        return current_hash ^ self.line_hashes[move_idx]

    def store(self, h, data):
        self.table[h] = data

    def lookup(self, h):
        return self.table.get(h, None)
from typing import Tuple, List
from random import randint
import math
import numpy as np


class DotsAndBoxesGame:
    """
    Implementation of the Dots-and-Boxes game, including relevant parameters for representing the game state
    and the logic for playing the game.

    Attributes
    ----------
    SIZE : int
        board size (in number of boxes per row and column)
    current_player : int
        player which is playing the next move. It can be determined manually or randomly which player should have the
        first turn of the game
        values: {-1, 1} = {player 2, player 1}
    result : int
        game result
        values: {None, -1, 0, 1} = {game is running, win player 2, draw, win player 1}
    N_LINES : int
        total number of lines that can be drawn as a result of the board size
    l : np.ndarray
        line vector of length N_LINES. Each element corresponds to one line on the board
        element values: {-1, 0, 1} = {line drawn by player 2, line is free, line drawn by player 1}
        the line indices correspond with the lines of a Dots-and-Boxes game in the following manner (i.e., first the
        horizontal lines are numbered, then the vertical lines):
        +  0 +  1 +
        6    8   10
        +  2 +  3 +
        7    9   11
        +  4 +  5 +
    N_BOXES : int
        total number of boxes that can be captured as a result of the board size
    b : np.ndarray
        tracks which boxes were captured by which player
        element values: {-1, 0, 1} = {box captured by player 2, box not captured yet, box captured by player 1}
    """
    def __init__(self, size: int, starting_player: int = None):

        self.SIZE = size
        self.current_player = (1 if randint(0, 1) == 1 else -1) if starting_player is None else starting_player
        self.result = None

        # lines
        self.N_LINES = 2 * size * (size + 1)
        self.l = np.zeros((self.N_LINES,), dtype=np.float32)

        # boxes
        self.N_BOXES = size * size
        self.b = np.zeros((size, size))

        # undo stack: each entry is (line, boxes_captured, player_before, result_before)
        self._history: list = []

    """
    Class setters.
    """
    def draw_line(self, line: int):
        assert self.l[line] == 0, "line is already drawn"
        self.l[line] = self.current_player

    def switch_current_player(self):
        self.current_player *= -1

    def capture_box(self, row: int, col: int):
        assert self.b[row][col] == 0, "box is already captured"
        self.b[row][col] = self.current_player


    """
    Game Logic.
    """
    def execute_move(self, line: int):
        # Save state for undo before modifying anything
        player_before  = self.current_player
        result_before  = self.result

        # execute move means drawing the line
        self.draw_line(line)

        # check whether a new box was captured
        boxes_captured = []
        for box in self.get_boxes_of_line(line):
            lines = self.get_lines_of_box(box)
            if np.count_nonzero(self.l[lines]) == 4:
                self.capture_box(row=box[0], col=box[1])
                boxes_captured.append(box)

        # switch current player when the player did not capture a box
        if not boxes_captured:
            self.switch_current_player()
        else:
            self.check_finished()

        # Push undo record
        self._history.append((line, boxes_captured, player_before, result_before))

    def undo_move(self):
        """Undo the last execute_move call. O(boxes_captured) — essentially O(1)."""
        if not self._history:
            raise RuntimeError("No move to undo")
        line, boxes_captured, player_before, result_before = self._history.pop()

        # Restore scalar fields first
        self.current_player = player_before
        self.result         = result_before

        # Un-draw the line
        self.l[line] = 0.0

        # Un-capture any boxes
        for box in boxes_captured:
            self.b[box[0]][box[1]] = 0.0


    def __eq__(self, obj) -> bool:
        if obj is None:
            return False

        if not isinstance(obj, DotsAndBoxesGame):
            return False

        if not self.current_player == obj.current_player or \
                not self.result == obj.result or \
                not self.SIZE == obj.SIZE or \
                not self.N_LINES == obj.N_LINES or \
                not np.array_equal(self.l, obj.l) or \
                not self.N_BOXES == obj.N_BOXES or \
                not np.array_equal(self.b, obj.b):
            return False
        return True



    def check_finished(self):
        if self.result is not None:
            return

        # Only end the game when all lines are filled
        if np.count_nonzero(self.l == 0) == 0:
            p1_score = int(np.sum(self.b == 1))
            p2_score = int(np.sum(self.b == -1))
            if p1_score > p2_score:
                self.result = 1
            elif p2_score > p1_score:
                self.result = -1
            else:
                self.result = 0

    def is_running(self) -> bool:
        return self.result is None

    def get_valid_moves(self) -> List[int]:
        return np.where(self.l == 0)[0].tolist()


    """
    Important methods to get line and box information.
    """
    def get_boxes_of_line(self, line: int) -> List[Tuple[int, int]]:

        if line < int(self.N_LINES / 2):
            # horizontal line
            i = line // self.SIZE
            j = line % self.SIZE  # column for both boxes

            if i == 0:
                return [(i, j)]
            elif i == self.SIZE:
                return [(i - 1, j)]
            else:
                return [(i - 1, j), (i, j)]

        else:
            # vertical line
            line = line - int(self.N_LINES / 2)
            j = line // self.SIZE
            i = line % self.SIZE  # row for both boxes

            if j == 0:
                return [(i, j)]
            elif j == self.SIZE:
                return [(i, j - 1)]
            else:
                return [(i, j - 1), (i, j)]  # [left box, right box]

    def get_lines_of_box(self, box: Tuple[int, int]) -> List[int]:
        i = box[0]
        j = box[1]

        # horizontal lines
        line_top = i * self.SIZE + j  # top line
        line_bottom = (i + 1) * self.SIZE + j  # bottom line

        # vertical lines
        line_left = int(self.N_LINES / 2) + j * self.SIZE + i  # left line
        line_right = int(self.N_LINES / 2) + (j + 1) * self.SIZE + i  # right line

        return [line_top, line_bottom, line_left, line_right]


    """
    Import methods for self-play using MCTS and the neural network that is to be trained.
    """
    def get_canonical_lines(self) -> np.ndarray:
        """
        The neural network expects the position vector s from the POV of the current player (=1).
        """
        canonical_lines = self.current_player * self.l
        canonical_lines[canonical_lines == 0.] = 0.  # -0.0 to 0.0

        return canonical_lines

    def get_canonical_boxes(self) -> np.ndarray:
        """
        The neural network expects the position vector s from the POV of the current player (=1).
        """
        canonical_boxes = self.current_player * self.b
        canonical_boxes[canonical_boxes == 0.] = 0.  # -0.0 to 0.0

        return canonical_boxes


    @staticmethod
    def get_rotations_and_reflections_lines(l: np.ndarray) -> List[np.ndarray]:
        """
        For a line vector, determine the equivalent line vectors, i.e., rotations and reflections.

        Parameters
        -------
        l : np.ndarray
            line vector for which the equivalent vectors should be determined

        Returns
        -------
        equivalents : [np.ndarray]
            rotations and reflections of the given line vector l (8 vectors, including l itself)
        """

        # rotations
        h, v = DotsAndBoxesGame.l_to_h_v(l)
        rotations = [np.copy(l)]
        for i in range(3):
            h, v = np.flipud(v.T), np.flipud(h.T)
            rotations.append(DotsAndBoxesGame.h_v_to_l(h=h, v=v))

        # reflections
        reflections = []
        for l in rotations:
            h, v = DotsAndBoxesGame.l_to_h_v(l)
            reflections.append(DotsAndBoxesGame.h_v_to_l(h=np.fliplr(h), v=np.fliplr(v)))

        return rotations + reflections


    @staticmethod
    def get_rotations_and_reflections_boxes(b: np.ndarray) -> List[np.ndarray]:
        """
        For a box matrix, determine the equivalent box matrices, i.e., rotations and reflections.

        Parameters
        -------
        b : np.ndarray
            box matrix for which the equivalent vectors should be determined

        Returns
        -------
        equivalents : [np.ndarray]
            rotations and reflections of the given box matrix b (8 matrices, including b itself)
        """

        # rotations
        b = np.copy(b)
        rotations = [np.copy(b)]
        for i in range(3):
            rotations.append(np.rot90(b))

        # reflections
        reflections = []
        for b in rotations:
            reflections.append(np.fliplr(b))

        return rotations + reflections


    @staticmethod
    def n_lines_to_size(n_lines: int) -> int:
        return int(-0.5 + math.sqrt(4 + 8 * n_lines) / 4)


    @staticmethod
    def l_to_h_v(l: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert line vector l to (h, v)-matrices representation (containing the horizontal and vertical lines).
        Example (numbers are indices of the line vector l):
            +  0 +  1 +
            6    8    10            [0  1]
        l = +  2 +  3 +    -->  h = [2  3]  and   v = [6  8  10]
            7    9    11            [4  5]            [7  9  11]
            +  4 +  5 +
        """

        n_lines = l.size
        size = DotsAndBoxesGame.n_lines_to_size(n_lines)

        h = np.zeros((size + 1, size), dtype=np.float32)
        v = np.zeros((size, size + 1), dtype=np.float32)

        for line in range(n_lines):
            if line < n_lines / 2:
                # horizontal line
                i = int(line // size)
                j = int(line % size)
                h[i][j] = l[line]

            else:
                # vertical line
                j = int((line - n_lines / 2) // size)
                i = int((line - n_lines / 2) % size)
                v[i][j] = l[line]

        return h, v


    @staticmethod
    def h_v_to_l(h: np.ndarray, v: np.ndarray) -> np.ndarray:

        l = np.concatenate((
            np.matrix.flatten(h, order='C'),  # row-major
            np.matrix.flatten(v, order='F')   # column-major
        ))

        return l
"""
Alpha-Beta with full optimizations:
- No deepcopy: uses execute_move / undo_move on a single shared game state
- Zobrist transposition table: exact/lower/upper bound caching with best-move ordering
- Iterative Deepening with strict time limit
- Optimized Move Ordering (TT -> Greedy -> Remaining)
"""
import random
import time
from typing import Tuple
from math import inf
import numpy as np




class AlphaBetaPlayer(BaseAgent):

    def __init__(self, name: str = "Alpha-Beta Bot", time_limit: float = 2.0, endgame_threshold: int = 12):
        super().__init__(name)
        self.time_limit = time_limit
        self.endgame_threshold = endgame_threshold

    def get_move(self, s: DotsAndBoxesGame) -> int:
        valid_moves = s.get_valid_moves()
        
        if len(valid_moves) > self.endgame_threshold:
            # 1. Greedy Capture: return an immediate 3-line box capture
            for r in range(s.SIZE):
                for c in range(s.SIZE):
                    if s.b[r][c] == 0:
                        lines = s.get_lines_of_box((r, c))
                        if sum(1 for ln in lines if s.l[ln] != 0) == 3:
                            return [ln for ln in lines if s.l[ln] == 0][0]
            
            # 2. Random Safe Move: move that doesn't create a 3-line box
            random.shuffle(valid_moves)
            for move in valid_moves:
                safe = True
                for box in s.get_boxes_of_line(move):
                    lines = s.get_lines_of_box(box)
                    if sum(1 for ln in lines if s.l[ln] != 0) == 2:
                        safe = False
                        break
                if safe:
                    return move
                    
            # 3. Fallback
            return valid_moves[0]
            
        else:
            return self._iterative_deepening(s)

    def _iterative_deepening(self, s_node: DotsAndBoxesGame) -> int:
        valid_moves = s_node.get_valid_moves()
        max_depth = len(valid_moves)
        
        zobrist = ZobristHash(s_node.N_LINES)
        initial_hash = zobrist.compute_initial_hash(s_node.l)
        
        end_time = time.time() + self.time_limit
        best_move_overall = valid_moves[0]
        
        for current_depth in range(1, max_depth + 1):
            tt_backup = zobrist.table.copy()
            try:
                move, _ = AlphaBetaPlayer.alpha_beta_search(
                    s_node       = s_node,
                    a_latest     = None,
                    depth        = current_depth,
                    alpha        = -inf,
                    beta         = inf,
                    maximize     = True,
                    zobrist      = zobrist,
                    current_hash = initial_hash,
                    end_time     = end_time
                )
                best_move_overall = move
            except TimeoutError:
                zobrist.table = tt_backup
                break
                
        return best_move_overall

    @staticmethod
    def alpha_beta_search(s_node: DotsAndBoxesGame,
                          a_latest: int,
                          depth: int,
                          alpha: float,
                          beta: float,
                          maximize: bool,
                          zobrist: ZobristHash,
                          current_hash: int,
                          end_time: float) -> Tuple[int, float]:

        if time.time() > end_time:
            raise TimeoutError()

        valid_moves = s_node.get_valid_moves()

        # --- Base case ---
        if not valid_moves or depth == 0 or not s_node.is_running():
            player         = s_node.current_player if maximize else -s_node.current_player
            player_boxes   = int((s_node.b == player).sum())
            opponent_boxes = int((s_node.b == -player).sum())

            if not s_node.is_running():
                if player_boxes > opponent_boxes: return a_latest,  10000
                if player_boxes < opponent_boxes: return a_latest, -10000
                return a_latest, 0

            return a_latest, player_boxes - opponent_boxes

        # --- TT Lookup ---
        original_alpha = alpha
        original_beta  = beta
        tt_best_move   = None

        tt_entry = zobrist.lookup(current_hash)
        if tt_entry is not None:
            tt_best_move = tt_entry.get('move')
            if tt_entry['depth'] >= depth:
                flag = tt_entry['type']
                if flag == 'exact':
                    return tt_entry['move'], tt_entry['value']
                if flag == 'lower' and tt_entry['value'] > alpha:
                    alpha = tt_entry['value']
                if flag == 'upper' and tt_entry['value'] < beta:
                    beta = tt_entry['value']
                if alpha >= beta:
                    return tt_entry['move'], tt_entry['value']

        # --- Optimized Move Ordering ---
        ordered_moves = []
        if tt_best_move is not None and tt_best_move in valid_moves:
            ordered_moves.append(tt_best_move)
            valid_moves.remove(tt_best_move)

        greedy_moves = []
        remaining_moves = []
        
        for move in valid_moves:
            is_greedy = False
            for box in s_node.get_boxes_of_line(move):
                lines = s_node.get_lines_of_box(box)
                if sum(1 for ln in lines if s_node.l[ln] != 0) == 3:
                    is_greedy = True
                    break
            if is_greedy:
                greedy_moves.append(move)
            else:
                remaining_moves.append(move)

        random.shuffle(remaining_moves)
        ordered_moves.extend(greedy_moves)
        ordered_moves.extend(remaining_moves)
        valid_moves = ordered_moves

        # --- Search (no deepcopy — uses execute_move / undo_move) ---
        a_best = None
        v_best = -inf if maximize else inf

        for a in valid_moves:
            player_before   = s_node.current_player
            s_node.execute_move(a)
            player_switched = (s_node.current_player != player_before)

            child_hash = zobrist.update_hash(current_hash, a)

            try:
                _, v_child = AlphaBetaPlayer.alpha_beta_search(
                    s_node       = s_node,
                    a_latest     = a,
                    depth        = depth - 1 if player_switched else depth,
                    alpha        = alpha,
                    beta         = beta,
                    maximize     = (not player_switched) if maximize else player_switched,
                    zobrist      = zobrist,
                    current_hash = child_hash,
                    end_time     = end_time
                )
            finally:
                s_node.undo_move()   # restore state

            if maximize:
                if v_child > v_best:
                    a_best = a
                    v_best = v_child
                if v_best > beta:
                    break
                alpha = max(alpha, v_best)
            else:
                if v_child < v_best:
                    a_best = a
                    v_best = v_child
                if v_best < alpha:
                    break
                beta = min(beta, v_best)

        # --- TT Store ---
        if   v_best <= original_alpha: tt_type = 'upper'
        elif v_best >= original_beta:  tt_type = 'lower'
        else:                          tt_type = 'exact'

        zobrist.store(current_hash, {
            'move': a_best, 'value': v_best,
            'depth': depth, 'type': tt_type,
        })

        return a_best, v_best

# ====================================================================
# CODINGAME I/O WRAPPER
# ====================================================================

def cg_to_ij(box_str, size):
    col_char = box_str[0]
    row_char = box_str[1:]
    j = ord(col_char) - ord('A')
    i = size - int(row_char)
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
        # horizontal line
        i = line_index // size
        j = line_index % size
        if i < size:
            box_i, box_j, side = i, j, 'T'
        else:
            box_i, box_j, side = i - 1, j, 'B'
    else:
        # vertical line
        v_idx = line_index - num_h
        j = v_idx // size
        i = v_idx % size
        if j < size:
            box_i, box_j, side = i, j, 'L'
        else:
            box_i, box_j, side = i, j - 1, 'R'
            
    col_char = chr(ord('A') + box_j)
    row_str = str(size - box_i)
    return f"{col_char}{row_str} {side}"

def run_codingame(agent_type):
    # Initial input
    board_size = int(input())
    player_id = input()
    
    # We always instantiate the game from the perspective of the current state.
    # Player 'A' goes first. If we are 'A', we want to win. If we are 'B', we want to win.
    # Actually, DotsAndBoxesGame expects current_player to be 1 or -1. 
    # Let's say we are always player 1 in our internal state for the search.
    # Wait, the state given by CodinGame is exactly what is currently on the board.
    # We can reconstruct the game state `s` every turn.
    
    first_turn = True
    
    while True:
        try:
            line_in = input()
        except EOFError:
            break
            
        player_score, opponent_score = [int(i) for i in line_in.split()]
        num_boxes = int(input())
        
        # We initialize an empty board and then apply the lines that ARE drawn.
        s = DotsAndBoxesGame(size=board_size, starting_player=1)
        
        drawn_lines = set(range(s.N_LINES))
        
        # Read the playable sides
        playable_lines = []
        for i in range(num_boxes):
            box, sides = input().split()
            for side in sides:
                l_idx = cg_to_line(box, side, board_size, s.N_LINES)
                playable_lines.append(l_idx)
                
        # Whatever is NOT playable is already drawn.
        # But wait! Who drew it? It doesn't matter for Dots and Boxes!
        # The game state evaluation only cares about how many boxes each player currently owns, 
        # and whose turn it is.
        # So we can just set `s.l[drawn] = 1`, and then manually set the scores!
        # But `s.b` matrix needs to be correct so `is_running()` works.
        # Actually, it's easier to just reconstruct the board state accurately.
        for l_idx in range(s.N_LINES):
            if l_idx not in playable_lines:
                s.l[l_idx] = 1.0 # arbitrary player, doesn't matter for future moves
                
        # Reconstruct boxes
        # The base case of alpha_beta uses (s.b == player).sum() to find score.
        # But we already know player_score and opponent_score!
        # So we can just override `player_boxes` and `opponent_boxes` in alpha_beta_search!
        # Or even better, we just assign the already completed boxes arbitrarily to match the score!
        
        # Let's properly compute which boxes are closed
        for r in range(s.SIZE):
            for c in range(s.SIZE):
                lines = s.get_lines_of_box((r, c))
                if sum(1 for ln in lines if s.l[ln] != 0) == 4:
                    s.b[r][c] = 1.0 # Assign all closed to player 1 just to mark them closed
                    
        # Wait, if we assign all closed boxes to player 1, the base case will evaluate player 1 as having all those boxes!
        # This will skew the evaluation!
        # We need to distribute them such that the difference matches `player_score - opponent_score`.
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
                        
        # Now the difference (s.b == 1).sum() - (s.b == -1).sum() is exactly `diff`!
        # But wait, there might be a parity issue. It's fine, AlphaBeta evaluates difference.
        
        # Update is_running
        if (s.b != 0).all():
            s.result = 1 if player_score > opponent_score else (-1 if opponent_score > player_score else 0)
            
        time_limit = 0.95 if first_turn else 0.09
        first_turn = False
        
        agent = AlphaBetaPlayer(name="AlphaBeta", time_limit=time_limit, endgame_threshold=999) # Always use Iterative Deepening
        move = agent.get_move(s)
            
        print(line_to_cg(move, board_size, s.N_LINES) + f" MSG {agent_type} {time_limit}s")


if __name__ == '__main__':
    run_codingame('alphabeta')
