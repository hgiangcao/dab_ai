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

    def clone(self, track_history: bool = True) -> 'DotsAndBoxesGame':
        """Fast clone: copies only essential mutable state, avoiding copy.deepcopy overhead.
        
        Args:
            track_history: If False, skip copying the undo history (saves time when
                           the caller will never call undo_move on the clone, e.g. MCTS).
        """
        c = object.__new__(DotsAndBoxesGame)
        c.SIZE = self.SIZE
        c.N_LINES = self.N_LINES
        c.N_BOXES = self.N_BOXES
        c.current_player = self.current_player
        c.result = self.result
        c.l = self.l.copy()
        c.b = self.b.copy()
        c._history = list(self._history) if track_history else []
        return c


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

        p1_score = int(np.sum(self.b == 1))
        p2_score = int(np.sum(self.b == -1))
        remaining_boxes = self.N_BOXES - p1_score - p2_score

        # Early termination: result is already mathematically decided
        # If one player leads by more than all remaining uncaptured boxes,
        # the opponent cannot catch up regardless of who captures what remains.
        if p1_score > p2_score + remaining_boxes:
            self.result = 1
        elif p2_score > p1_score + remaining_boxes:
            self.result = -1
        elif remaining_boxes == 0:
            # All boxes captured: normal end-of-game scoring
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
            b = np.rot90(b)
            rotations.append(np.copy(b))

        # reflections
        reflections = []
        for b_rot in rotations:
            reflections.append(np.fliplr(b_rot))

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
