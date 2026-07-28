import random
from agent_interface import BaseAgent

class UCLABot_v6(BaseAgent):
    """
    Bitboard-based Greedy Bot.
    Fast execution, but lacks advanced endgame chain logic.
    """
    def __init__(self, name="UCLABot_v6", size=5):
        super().__init__(name)
        
        # Khởi tạo động dựa trên kích thước bàn cờ
        self.size = size
        self.n_lines = 2 * size * (size + 1)
        self.half_lines = self.n_lines // 2
        
        self.BOX_MASKS = []
        self.LINE_TO_BOXES = [[] for _ in range(self.n_lines)]
        
        self._precompute_bitboards()

    def _precompute_bitboards(self):
        for r in range(self.size):
            for c in range(self.size):
                # Cạnh ngang (Lưu theo hàng - Row-Major)
                top = r * self.size + c
                bottom = (r + 1) * self.size + c
                
                # Cạnh dọc (Lưu theo cột - Column-Major - Khớp với môi trường game)
                left = self.half_lines + c * self.size + r
                right = self.half_lines + (c + 1) * self.size + r
                
                # Tạo Bitmask cho ô vuông
                mask = (1 << top) | (1 << bottom) | (1 << left) | (1 << right)
                self.BOX_MASKS.append(mask)
                
                # Ánh xạ từ cạnh ngược về ô
                self.LINE_TO_BOXES[top].append(mask)
                self.LINE_TO_BOXES[bottom].append(mask)
                self.LINE_TO_BOXES[left].append(mask)
                self.LINE_TO_BOXES[right].append(mask)

    def get_move(self, s) -> int:
        valid_moves = s.get_valid_moves()
        if not valid_moves:
            return -1

        # Cập nhật kích thước và tạo lại bitboard nếu môi trường thay đổi kích thước đột ngột
        if len(s.l) != self.n_lines:
            self.size = s.SIZE
            self.n_lines = s.N_LINES
            self.half_lines = self.n_lines // 2
            self.BOX_MASKS = []
            self.LINE_TO_BOXES = [[] for _ in range(self.n_lines)]
            self._precompute_bitboards()

        # Chuyển trạng thái bàn cờ sang Bitboard
        state_bb = 0
        for i, val in enumerate(s.l):
            if val != 0:
                state_bb |= (1 << i)

        # Ưu tiên 1: Tìm nước ăn điểm
        for move in valid_moves:
            if self._is_capture(state_bb, move):
                return move

        # Ưu tiên 2: Tìm các nước an toàn (không tạo ô 3 cạnh)
        safe_moves = [m for m in valid_moves if self._is_safe(state_bb, m)]
        
        if safe_moves:
            return random.choice(safe_moves)

        # Ưu tiên 3: Hết cách, đành bốc bừa
        return random.choice(valid_moves)

    def _is_capture(self, state_bb, move):
        new_state = state_bb | (1 << move)
        for box_mask in self.LINE_TO_BOXES[move]:
            if (new_state & box_mask) == box_mask:
                return True
        return False

    def _is_safe(self, state_bb, move):
        new_state = state_bb | (1 << move)
        for box_mask in self.LINE_TO_BOXES[move]:
            overlap = new_state & box_mask
            # Lưu ý: int.bit_count() yêu cầu Python 3.10 trở lên
            if overlap.bit_count() == 3:
                return False
        return True