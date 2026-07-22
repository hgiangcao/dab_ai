import random
import numpy as np
from agent_interface import BaseAgent

class SimpleBotV2(BaseAgent):
    """
    Simple Bot V2:
    An advanced heuristic agent for Dots and Boxes featuring:
    1. Static O(1) checks for captures and 3-sided boxes to avoid slow state changes.
    2. True connected-component chain and loop detection.
    3. Double-cross and loop-handling logic during capture phases to retain control.
    4. Safe move maximization and chain-creation avoidance in non-capture phases.
    """
    def __init__(self, name="SimpleBotV2"):
        super().__init__(name)

    def get_move(self, s):
        valid_moves = s.get_valid_moves()
        if not valid_moves:
            return -1

        # 1. Take forced captures if available, prioritizing chain control (double-cross)
        capture_moves = [m for m in valid_moves if self._is_capture_static(s, m)]
        if capture_moves:
            return self._best_capture(s, capture_moves)

        # 2. Evaluate non-capture moves using chain and giveaway heuristics
        scores = {}
        for m in valid_moves:
            s.execute_move(m)
            try:
                # Count how many boxes this move gives away to the opponent
                giveaway = self._count_giveaway_after_move(s)
                score = -1000 * giveaway

                # If it is a safe move (no giveaway), prioritize safety and discourage building future opponent chains
                if giveaway == 0:
                    # Count safe moves remaining for us
                    safe_moves = sum(1 for next_m in s.get_valid_moves() if not self._creates_three_side_box_static(s, next_m))
                    score += 10 * safe_moves

                    # Discourage creating 2-sided boxes (which are one line away from opening a chain)
                    two_sided_boxes = 0
                    for r in range(s.SIZE):
                        for c in range(s.SIZE):
                            b = (r, c)
                            if sum(s.l[x] != 0 for x in s.get_lines_of_box(b)) == 2:
                                two_sided_boxes += 1
                    score -= 5 * two_sided_boxes

                scores[m] = score
            finally:
                s.undo_move()

        best = max(scores.values())
        best_moves = [m for m, v in scores.items() if v == best]
        return random.choice(best_moves)

    def _is_capture_static(self, s, move):
        # Statically check if playing 'move' completes any box (drawn lines == 3)
        for b in s.get_boxes_of_line(move):
            drawn = sum(s.l[x] != 0 for x in s.get_lines_of_box(b))
            if drawn == 3:
                return True
        return False

    def _creates_three_side_box_static(self, s, move):
        # Statically check if playing 'move' leaves any box with exactly 3 drawn lines (drawn lines == 2)
        for b in s.get_boxes_of_line(move):
            drawn = sum(s.l[x] != 0 for x in s.get_lines_of_box(b))
            if drawn == 2:
                return True
        return False

    def _get_chains(self, s):
        # Identify all connected components of boxes with 2 or 3 drawn lines
        active_boxes = []
        for r in range(s.SIZE):
            for c in range(s.SIZE):
                b = (r, c)
                drawn = sum(s.l[x] != 0 for x in s.get_lines_of_box(b))
                if 2 <= drawn < 4:
                    active_boxes.append((b, drawn))

        box_to_idx = {b: i for i, (b, _) in enumerate(active_boxes)}
        adj = {i: [] for i in range(len(active_boxes))}
        for i, (b1, _) in enumerate(active_boxes):
            lines1 = set(s.get_lines_of_box(b1))
            for j in range(i + 1, len(active_boxes)):
                b2, _ = active_boxes[j]
                lines2 = set(s.get_lines_of_box(b2))
                shared_lines = lines1.intersection(lines2)
                if shared_lines:
                    shared_line = list(shared_lines)[0]
                    if s.l[shared_line] == 0:
                        adj[i].append(j)
                        adj[j].append(i)

        visited = set()
        components = []
        for i in range(len(active_boxes)):
            if i not in visited:
                comp = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    comp.append(active_boxes[curr])
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                           visited.add(neighbor)
                           queue.append(neighbor)
                components.append(comp)

        classified = []
        for comp in components:
            length = len(comp)
            box_ids = {b for b, _ in comp}
            
            degrees = {}
            for b, _ in comp:
                neighbors = 0
                for l in s.get_lines_of_box(b):
                    if s.l[l] == 0:
                        for other_b in s.get_boxes_of_line(l):
                            if other_b != b and other_b in box_ids:
                                neighbors += 1
                degrees[b] = neighbors

            is_loop = False
            is_chain = False
            if length >= 4 and all(deg == 2 for deg in degrees.values()):
                is_loop = True
            elif length >= 3 and list(degrees.values()).count(1) == 2:
                is_chain = True
            elif length < 3:
                is_chain = True

            classified.append({
                "boxes": comp,
                "length": length,
                "is_loop": is_loop,
                "is_chain": is_chain
            })
        return classified

    def _count_completed_boxes_static(self, s):
        count = 0
        for r in range(s.SIZE):
            for c in range(s.SIZE):
                b = (r, c)
                if sum(s.l[x] != 0 for x in s.get_lines_of_box(b)) == 4:
                    count += 1
        return count


    def _count_giveaway_after_move(self, s):
        # Simulates the opponent's turn of capturing all available boxes
        initial_completed = self._count_completed_boxes_static(s)
        moves_made = []
        try:
            while True:
                caps = [m for m in s.get_valid_moves() if self._is_capture_static(s, m)]
                if not caps:
                    break
                m = caps[0]
                s.execute_move(m)
                moves_made.append(m)
            final_completed = self._count_completed_boxes_static(s)
            return final_completed - initial_completed
        finally:
            for m in reversed(moves_made):
                s.undo_move()

    def _best_capture(self, s, moves):
        # Determine the best capture move to play during our turn
        # We simulate playing out the captures, applying double-cross logic at the end of chains/loops
        chains = self._get_chains(s)
        long_chains_and_loops = [c for c in chains if c["length"] >= 3 or c["is_loop"]]
        
        scores = {}
        for m in moves:
            # Identify which chain 'm' belongs to
            current_chain = None
            for c in chains:
                c_boxes = {box for box, _ in c["boxes"]}
                if any(b in c_boxes for b in s.get_boxes_of_line(m)):
                    current_chain = c
                    break
                    
            leave = 0
            other_long_chains = len(long_chains_and_loops)
            if current_chain:
                if current_chain["is_loop"]:
                    leave = 4
                elif current_chain["length"] >= 3:
                    leave = 2
                if current_chain in long_chains_and_loops:
                    other_long_chains -= 1
                    
            initial_completed = self._count_completed_boxes_static(s)
            
            s.execute_move(m)
            moves_made = [m]
            control_retained = False
            try:
                # Play out remaining forced captures
                while True:
                    caps = [c for c in s.get_valid_moves() if self._is_capture_static(s, c)]
                    if not caps:
                        break

                    current_completed = self._count_completed_boxes_static(s)
                    boxes_taken = current_completed - initial_completed
                    
                    if current_chain:
                        remaining = current_chain["length"] - boxes_taken
                        
                        # Double-cross condition
                        if remaining == leave and other_long_chains > 0:
                            non_caps = [nc for nc in s.get_valid_moves() if not self._is_capture_static(s, nc)]
                            if non_caps:
                                select_move = non_caps[0]
                                s.execute_move(select_move)
                                moves_made.append(select_move)
                                control_retained = True
                                break

                    # Choose the next capture, preferably in the same chain
                    next_cap = caps[0]
                    if current_chain:
                        c_boxes = {box for box, _ in current_chain["boxes"]}
                        for c in caps:
                            if any(b in c_boxes for b in s.get_boxes_of_line(c)):
                                next_cap = c
                                break

                    s.execute_move(next_cap)
                    moves_made.append(next_cap)

                final_completed = self._count_completed_boxes_static(s)
                total_boxes_taken = final_completed - initial_completed
                
                score = total_boxes_taken
                if control_retained:
                    score += 100
                else:
                    score -= 100

                scores[m] = score
            finally:
                for played_m in reversed(moves_made):
                    s.undo_move()

        best = max(scores.values())
        best_moves = [m for m in moves if scores[m] == best]
        return random.choice(best_moves)

SimpleBot_v2 = SimpleBotV2

