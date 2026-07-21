import random
from agent_interface import BaseAgent

class UCLABot(BaseAgent):
    def __init__(self, name="UCLA JS Bot"):
        super().__init__(name)

    def get_move(self, s) -> int:
        valid_moves = s.get_valid_moves()
        
        # 1. Identify all currently capturable boxes (3 lines drawn)
        captures = []
        for r in range(s.SIZE):
            for c in range(s.SIZE):
                if s.b[r][c] == 0:
                    lines = s.get_lines_of_box((r, c))
                    free = [ln for ln in lines if s.l[ln] == 0]
                    if len(free) == 1:
                        captures.append(free[0])

        # 2. Chain processing & Sacrifice (The Double-Cross)
        if captures:
            chain_moves, remaining_safe = self._trace_chain(s)
            # If taking this chain forces us to open a new one (no safe moves left),
            # and the chain is long enough, sacrifice the last 2 boxes to maintain control.
            if len(chain_moves) >= 3 and remaining_safe == 0:
                return chain_moves[-1]  # The declining move
            return captures[0]          # Otherwise, take the box

        # 3. Play safe moves (moves that do not create a 3-line box)
        safe_moves = []
        for m in valid_moves:
            is_safe = True
            for box in s.get_boxes_of_line(m):
                lines = s.get_lines_of_box(box)
                if sum(1 for ln in lines if s.l[ln] != 0) == 2:
                    is_safe = False
                    break
            if is_safe:
                safe_moves.append(m)
        
        if safe_moves:
            return random.choice(safe_moves)

        # 4. Fallback (Singletons, Doubletons, or Any)
        # Mimics the JS logic by picking the move that gives away the fewest boxes
        best_move = valid_moves[0]
        min_giveaway = float('inf')
        
        for m in valid_moves:
            giveaway = self._count_giveaway(s, m)
            if giveaway < min_giveaway:
                min_giveaway = giveaway
                best_move = m
                
        return best_move

    def _trace_chain(self, s):
        """Simulates capturing a chain to count its length and check board state."""
        moves = 0
        chain = []
        mover = s.current_player
        
        while True:
            caps = []
            for r in range(s.SIZE):
                for c in range(s.SIZE):
                    if s.b[r][c] == 0:
                        lines = s.get_lines_of_box((r, c))
                        free = [ln for ln in lines if s.l[ln] == 0]
                        if len(free) == 1:
                            caps.append(free[0])
            if not caps:
                break
            
            move = caps[0]
            chain.append(move)
            s.execute_move(move)
            moves += 1
            
            # Stop if the turn passes to the opponent
            if s.current_player != mover:
                break
                
        # Count remaining safe moves on the board after the chain is resolved
        safe_count = 0
        if s.is_running():
            for m in s.get_valid_moves():
                is_safe = True
                for box in s.get_boxes_of_line(m):
                    if sum(1 for ln in s.get_lines_of_box(box) if s.l[ln] != 0) == 2:
                        is_safe = False
                        break
                if is_safe: 
                    safe_count += 1
                
        # Undo all simulated moves to restore board state
        for _ in range(moves):
            s.undo_move()
            
        return chain, safe_count

    def _count_giveaway(self, s, move):
        """Simulates a move and counts how many boxes the opponent immediately gets."""
        s.execute_move(move)
        chain, _ = self._trace_chain(s)
        giveaway = len(chain)
        s.undo_move()
        return giveaway