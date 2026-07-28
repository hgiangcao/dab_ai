"""
arm_bot.py – Wrapper for the Armando8766 minimax bot.

The Armando8766 bot uses its own internal board representation (a 2-D matrix
of dots/lines/box-scores).  This wrapper translates the project's
DotsAndBoxesGame state into that format, runs Armando's minimax (alpha-beta)
algorithm to pick a move, then translates the result back to a line index.

Board conventions:
    Armando's matrix is (2*SIZE+1) rows × (2*SIZE+1) columns:
        even-row / even-col  → dot  '*'
        even-row / odd-col   → horizontal edge slot (char '-' or ' ')
        odd-row  / even-col  → vertical edge slot   (char '|' or ' ')
        odd-row  / odd-col   → box cell             (integer score)

    This project's line vector for a SIZE×SIZE grid:
        indices 0 … SIZE*(SIZE+1)-1       → horizontal lines (row-major)
        indices SIZE*(SIZE+1) … N_LINES-1 → vertical lines   (col-major)
"""

import sys
import os
import random

# ---------------------------------------------------------------------------
# Make the Armando8766 package importable regardless of the working directory
# ---------------------------------------------------------------------------
_ARM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ref_bots", "Armando8766"
)
if _ARM_DIR not in sys.path:
    sys.path.insert(0, _ARM_DIR)

from Algorithm import Algo   # miniMax lives here
from Board import Game        # Game board
from Nodes import Thing       # Node wrapper

from agent_interface import BaseAgent


# ---------------------------------------------------------------------------
# Helper – build an Armando "Thing" node from a DotsAndBoxesGame state
# ---------------------------------------------------------------------------

def _game_state_to_thing(s) -> Thing:
    """
    Convert a DotsAndBoxesGame object into an Armando Thing node so that
    Algo.miniMax can be called on it.

    The Armando matrix uses random box-point integers that affect the
    internal score tracking.  Since we only need the *move* (not a score
    we will use externally), we assign every box-cell a constant value of 1.
    """
    SIZE = s.SIZE
    dim = 2 * SIZE + 1          # matrix side length

    # Build the 2-D matrix
    mat = []
    for row in range(dim):
        r_list = []
        for col in range(dim):
            if row % 2 == 0 and col % 2 == 0:
                r_list.append('*')          # dot
            elif row % 2 == 1 and col % 2 == 1:
                r_list.append(1)            # box value (constant for move-picking)
            else:
                r_list.append(' ')          # open edge slot (will be filled below)
        mat.append(r_list)

    # Fill already-drawn lines from s.l
    # Horizontal lines: line index h = row_box * SIZE + col_box
    #   → matrix position: mat_row = row_box * 2 + (0 if top, 2 if bottom... )
    #   We iterate over line indices directly.
    n_horiz = SIZE * (SIZE + 1)

    for line_idx in range(s.N_LINES):
        if s.l[line_idx] == 0:
            continue            # edge not drawn

        if line_idx < n_horiz:
            # Horizontal line
            # row in h-grid (0 … SIZE): which horizontal row of edges
            h_row = line_idx // SIZE        # 0 … SIZE
            h_col = line_idx % SIZE         # 0 … SIZE-1
            mat_row = h_row * 2             # even
            mat_col = h_col * 2 + 1         # odd
            mat[mat_row][mat_col] = '-'
        else:
            # Vertical line
            v_idx = line_idx - n_horiz
            # v_idx encodes: col_group * SIZE + row_in_group
            v_col = v_idx // SIZE           # 0 … SIZE  (column group of v-edges)
            v_row = v_idx % SIZE            # 0 … SIZE-1
            mat_row = v_row * 2 + 1         # odd
            mat_col = v_col * 2             # even
            mat[mat_row][mat_col] = '|'

    # Wrap in Armando's Game and Thing
    board = Game(mat, dimX=dim, dimY=dim)
    node  = Thing(board)

    # Reconstruct the current score from captured boxes
    # (Armando's score is human_score − AI_score; since we are the AI here,
    #  s.current_player = +1 means we are player-1.)
    p1_score = int((s.b == 1).sum())
    p2_score = int((s.b == -1).sum())
    if s.current_player == 1:
        # We are player-1, opponent is player-2; Armando's perspective: AI positive
        node.CurrentScore = p1_score - p2_score
    else:
        node.CurrentScore = p2_score - p1_score

    return node


# ---------------------------------------------------------------------------
# Helper – convert an Armando (mat_col, mat_row) move back to a line index
# ---------------------------------------------------------------------------

def _arm_move_to_line(arm_col, arm_row, SIZE) -> int:
    """
    Armando returns a move as (mat_col, mat_row) (it calls them (i, j) but
    stores children keyed as (col, row)).

    Convert back to this project's flat line index.
    """
    mat_row = arm_row
    mat_col = arm_col

    n_horiz = SIZE * (SIZE + 1)

    if mat_row % 2 == 0:
        # Horizontal edge: mat_row is even, mat_col is odd
        h_row = mat_row // 2        # 0 … SIZE
        h_col = (mat_col - 1) // 2  # 0 … SIZE-1
        return h_row * SIZE + h_col
    else:
        # Vertical edge: mat_row is odd, mat_col is even
        v_row = (mat_row - 1) // 2  # 0 … SIZE-1
        v_col = mat_col // 2        # 0 … SIZE
        return n_horiz + v_col * SIZE + v_row


# ---------------------------------------------------------------------------
# The actual wrapper bot
# ---------------------------------------------------------------------------

class ArmandoBot(BaseAgent):
    """
    Wrapper around the Armando8766 minimax bot.

    Parameters
    ----------
    name : str
        Display name used in tournaments.
    ply : int
        Search depth (number of plies) for the minimax algorithm.
        Higher values are stronger but slower.

        Performance note: Armando's algorithm expands the game tree lazily
        each call (no caching across moves), so deep searches on a 5×5 board
        can be slow.  ply=2 is fast; ply=3 is tractable for small boards.
        Default is 2.
    """

    def __init__(self, name: str = "ArmandoBot", ply: int = 2):
        super().__init__(name)
        self.ply = ply

    def get_move(self, s) -> int:
        """
        Given a DotsAndBoxesGame state *s*, return the line index chosen by
        the Armando minimax algorithm.
        """
        valid_moves = s.get_valid_moves()
        if not valid_moves:
            return -1

        # Build the Armando internal state
        node = _game_state_to_thing(s)

        # Run miniMax – it returns (mat_col, mat_row)
        arm_move = Algo.miniMax(node, self.ply)

        # Translate back to a line index
        line = _arm_move_to_line(arm_move[0], arm_move[1], s.SIZE)

        # Safety check: if translation produced an invalid move, fall back
        if line not in valid_moves:
            line = random.choice(valid_moves)

        return line
