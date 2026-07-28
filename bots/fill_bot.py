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
        
        # ƯU TIÊN 1: Tìm cạnh thứ 4 (Ăn điểm)
        for edge in available_edges:
            adjacent_boxes = game_state.get_boxes_of_line(edge)
            for box in adjacent_boxes:
                lines = game_state.get_lines_of_box(box)
                filled_edges = np.count_nonzero(game_state.l[lines])
                if filled_edges == 3:
                    return edge # Điền ngay để ăn điểm!
                    
        # ƯU TIÊN 2: Tìm cạnh an toàn (Cạnh thứ 1 hoặc 2)
        safe_edges = []
        for edge in available_edges:
            adjacent_boxes = game_state.get_boxes_of_line(edge)
            is_safe = True
            
            # Kiểm tra xem việc điền cạnh này có làm ô nào lên 3 cạnh không
            for box in adjacent_boxes:
                lines = game_state.get_lines_of_box(box)
                filled_edges = np.count_nonzero(game_state.l[lines])
                if filled_edges == 2:
                    is_safe = False
                    break
                    
            if is_safe:
                safe_edges.append(edge)
                
        if safe_edges:
            # Chọn ngẫu nhiên một nước đi an toàn để tăng tính đa dạng cho dữ liệu
            return random.choice(safe_edges)
            
        # ƯU TIÊN 3: Bắt buộc hy sinh (Điền cạnh thứ 3)
        # Vì không còn cách nào khác, chọn đại một cạnh còn trống
        if available_edges:
            return random.choice(available_edges)
            
        return None
