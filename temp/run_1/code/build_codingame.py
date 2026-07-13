import re
import os

def read_file_no_imports(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    out_lines = []
    for line in lines:
        if line.strip().startswith('import sys'):
            continue
        if line.strip().startswith('import os'):
            continue
        if line.strip().startswith('from agent_interface'):
            continue
        if line.strip().startswith('from lookup_board'):
            continue
        if line.strip().startswith('sys.path.insert'):
            continue
        if line.strip().startswith('from game'):
            continue
        if line.strip().startswith('from bots.alpha_beta'):
            continue
        out_lines.append(line)
    text = "".join(out_lines)
    if not text.endswith('\n'):
        text += '\n'
    return text

COMMON_IMPORTS = """import sys
import math
import random
import time
import numpy as np
from typing import Tuple, List, Dict

class BaseAgent:
    def __init__(self, name=""):
        self.name = name
"""

CG_WRAPPER_CODE = """
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
        
        if agent_type == 'alphabeta':
            agent = AlphaBetaPlayer(name="AlphaBeta", time_limit=time_limit, endgame_threshold=999) # Always use Iterative Deepening
            move = agent.get_move(s)
        else:
            agent = MCTSGAgent(name="MCTS", time_limit=time_limit)
            agent.mcts_parameters['time_limit'] = time_limit
            move = agent.get_move(s)
            
        print(line_to_cg(move, board_size, s.N_LINES) + f" MSG {agent_type} {time_limit}s")

if __name__ == '__main__':
    # When building the final file, this will be replaced
    pass
"""

def build():
    # 1. codingame_Agent_alphabeta.py
    with open('codingame_Agent_alphabeta.py', 'w') as f:
        f.write(COMMON_IMPORTS)
        f.write(read_file_no_imports('lookup_board.py'))
        f.write(read_file_no_imports('game.py'))
        f.write(read_file_no_imports('bots/alpha_beta.py'))
        f.write(CG_WRAPPER_CODE)
        f.write("if __name__ == '__main__':\n    run_codingame('alphabeta')\n")
        
    # 2. codingame_Agent_mcts.py
    with open('codingame_Agent_mcts.py', 'w') as f:
        f.write(COMMON_IMPORTS)
        f.write(read_file_no_imports('lookup_board.py'))
        f.write(read_file_no_imports('game.py'))
        f.write(read_file_no_imports('bots/alpha_beta.py'))
        f.write(read_file_no_imports('bots/mcts_heuristic.py'))
        f.write(read_file_no_imports('bots/mcts_x.py'))
        f.write(CG_WRAPPER_CODE)
        f.write("if __name__ == '__main__':\n    run_codingame('mcts')\n")
        
if __name__ == '__main__':
    build()
