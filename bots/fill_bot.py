import random
import numpy as np
import sys
import os

# Add parent directory to path to import agent_interface
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_interface import BaseAgent

class FillBot(BaseAgent):
    """
    Greedy heuristic bot designed for backward training fill.
    Priority 1: Complete a box if possible (score a point).
    Priority 2: Play a safe edge (do not give the opponent a box).
    Priority 3: Play a random edge (sacrifice).
    """
    def __init__(self, name: str = "FillBot"):
        super().__init__(name)

    def get_move(self, game_state) -> int:
        available_edges = game_state.get_valid_moves()
        if not available_edges:
            return None

        fill = game_state.box_fill_count        # O(1) array, precomputed
        line_to_boxes = game_state.line_to_boxes # precomputed static mapping

        safe_count = 0
        safe_choice = None
        sac_choice = available_edges[0]  # fallback sacrifice, avoid extra pass

        for edge in available_edges:
            boxes = line_to_boxes[edge]
            max_fill = 0
            for box in boxes:
                f = fill[box]
                if f == 3:
                    return edge          # PRIORITY 1: immediate capture, early exit
                if f > max_fill:
                    max_fill = f

            if max_fill < 2:
                # PRIORITY 2 candidate — reservoir sampling, no list built
                safe_count += 1
                if random.randrange(safe_count) == 0:
                    safe_choice = edge

        if safe_choice is not None:
            return safe_choice

        # PRIORITY 3: forced sacrifice — just return first valid edge,
        # or reservoir-sample if you need true uniformity (usually not needed for rollouts)
        return sac_choice
