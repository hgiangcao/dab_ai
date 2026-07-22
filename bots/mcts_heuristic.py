"""
Clean, standard UCT Monte Carlo Tree Search implementation for Dots and Boxes.
- Node-based statistics tracking (visits, total_value).
- UCT Selection.
- Incremental expansion with capturing -> safe -> remaining ordering.
- Heuristic Rollouts.
- Correct handling of extra turns without alternating assumption.
"""
import math
import random
import time
from agent_interface import BaseAgent

class Node:
    __slots__ = ('parent', 'move', 'children', 'visits', 'value_sum', 'player_to_move', 'untried_moves', 'terminal', 'bias')

    def __init__(self, parent, move, player_to_move, valid_moves, terminal, bias=0.0):
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.player_to_move = player_to_move
        self.terminal = terminal
        self.untried_moves = valid_moves if valid_moves is not None else []
        self.bias = bias

class MCTS:
    def __init__(self, n_simulations: int = 100, time_limit: float = None, c_puct: float = 1.4):
        self.n_simulations = n_simulations
        self.time_limit = time_limit
        self.c_puct = c_puct

    def search(self, game_state):
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None
            
        root = Node(parent=None, move=None, player_to_move=game_state.current_player, 
                    valid_moves=self._order_expansion_moves(game_state, valid_moves),
                    terminal=not game_state.is_running())
                    
        simulations_done = 0
        max_tree_depth = 0
        start_time = time.time()
        
        while True:
            if self.time_limit is not None:
                if time.time() - start_time >= self.time_limit:
                    break
            elif simulations_done >= self.n_simulations:
                break
                
            simulations_done += 1
            
            node = root
            moves_made = 0
            
            # SELECTION
            while not node.untried_moves and node.children:
                node = self._select_child(node)
                game_state.execute_move(node.move)
                moves_made += 1
                
            # EXPANSION
            if node.untried_moves:
                move = node.untried_moves.pop(0)
                
                bias = self._evaluate_bias(game_state, move)
                
                game_state.execute_move(move)
                moves_made += 1
                
                terminal = not game_state.is_running()
                valid_for_child = self._order_expansion_moves(game_state, game_state.get_valid_moves()) if not terminal else []
                child = Node(parent=node, move=move, player_to_move=game_state.current_player,
                             valid_moves=valid_for_child,
                             terminal=terminal, bias=bias)
                node.children.append(child)
                node = child
                
            max_tree_depth = max(max_tree_depth, moves_made)
            
            # ROLLOUT
            rollout_player = node.player_to_move
            val, rollout_moves = self._rollout(game_state, rollout_player)
            moves_made += rollout_moves
            
            # BACKPROPAGATION
            curr = node
            while curr is not None:
                curr.visits += 1
                curr.value_sum += val
                
                if curr.parent is not None:
                    if curr.parent.player_to_move != curr.player_to_move:
                        val = -val
                
                curr = curr.parent
                
            # UNDO Tree phase + Rollout moves
            for _ in range(moves_made):
                game_state.undo_move()
                
        total_time = time.time() - start_time
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            return best_child.move, max_tree_depth, simulations_done, total_time
        return None, 0, 0, 0.0

    def _select_child(self, node):
        best_score = -float('inf')
        best_child = None
        log_n = math.log(node.visits) if node.visits > 0 else 0
        
        for child in node.children:
            if child.visits == 0:
                ucb = float('inf')
            else:
                q = child.value_sum / child.visits
                if node.player_to_move != child.player_to_move:
                    q = -q
                ucb = q + self.c_puct * math.sqrt(log_n / child.visits) + (child.bias / (child.visits + 1))
            if ucb > best_score:
                best_score = ucb
                best_child = child
        return best_child

    def _rollout(self, s, rollout_player):
        moves_made = 0
        while s.is_running():
            valid = s.get_valid_moves()
            
            # 1. Capture if possible
            move = self._find_capture(s, valid)
            # 2. Heuristic rollout scoring
            if move is None:
                best_score = -float('inf')
                best_moves = []
                for m in valid:
                    score = self._score_rollout_move(s, m)
                    if score > best_score:
                        best_score = score
                        best_moves = [m]
                    elif score == best_score:
                        best_moves.append(m)
                move = random.choice(best_moves)
                    
            s.execute_move(move)
            moves_made += 1
            
        import numpy as np
        p1_score = int(np.sum(s.b == 1))
        p2_score = int(np.sum(s.b == -1))
        my_score = p1_score if rollout_player == 1 else p2_score
        opp_score = p2_score if rollout_player == 1 else p1_score
        val = float(my_score - opp_score) / (s.SIZE * s.SIZE)
            
        return val, moves_made

    def _score_rollout_move(self, s, move):
        score = 0
        is_capture = False
        is_safe = True
        
        for box in s.get_boxes_of_line(move):
            lines = s.get_lines_of_box(box)
            drawn = sum(1 for ln in lines if s.l[ln] != 0)
            if drawn == 3:
                is_capture = True
            if drawn == 2:
                is_safe = False
                
        if is_capture:
            score += 100
        elif is_safe:
            if len(s.get_boxes_of_line(move)) == 1:
                score += 30
            else:
                score += 20
        else:
            score -= 50
            
        return score

    def _evaluate_bias(self, s, move):
        return self._score_rollout_move(s, move) / 100.0

    def _find_capture(self, s, valid_moves):
        for move in valid_moves:
            for box in s.get_boxes_of_line(move):
                lines = s.get_lines_of_box(box)
                drawn = sum(1 for ln in lines if s.l[ln] != 0)
                if drawn == 3:
                    return move
        return None

    def _order_expansion_moves(self, s, valid_moves):
        if not valid_moves: return []
        
        captures = []
        safe_edge = []
        safe_center = []
        dangerous = []
        
        for move in valid_moves:
            is_capture = False
            is_safe = True
            for box in s.get_boxes_of_line(move):
                lines = s.get_lines_of_box(box)
                drawn = sum(1 for ln in lines if s.l[ln] != 0)
                if drawn == 3:
                    is_capture = True
                if drawn == 2:
                    is_safe = False
                    
            if is_capture:
                captures.append(move)
            elif is_safe:
                if len(s.get_boxes_of_line(move)) == 1:
                    safe_edge.append(move)
                else:
                    safe_center.append(move)
            else:
                dangerous.append(move)
                
        random.shuffle(captures)
        random.shuffle(safe_edge)
        random.shuffle(safe_center)
        random.shuffle(dangerous)
        
        if captures or safe_edge or safe_center:
            return captures + safe_edge + safe_center
            
        return dangerous


# ---------------------------------------------------------------------------
# Heuristic Agent wrapping MCTS
# ---------------------------------------------------------------------------

class MCTSHeuristicAgent(BaseAgent):
    def __init__(self, name: str, mcts_parameters: dict):
        super().__init__(name)
        self.mcts_parameters = mcts_parameters

    def _capturable_boxes(self, game_state) -> list:
        result = []
        for r in range(game_state.SIZE):
            for c in range(game_state.SIZE):
                if game_state.b[r][c] == 0:
                    lines = game_state.get_lines_of_box((r, c))
                    free = [ln for ln in lines if game_state.l[ln] == 0]
                    if len(free) == 1:
                        result.append(free[0])
        return result

    def _is_capturing_move(self, game_state, line) -> bool:
        for box in game_state.get_boxes_of_line(line):
            lines = game_state.get_lines_of_box(box)
            if sum(1 for ln in lines if game_state.l[ln] != 0) == 3:
                return True
        return False

    def _trace_capture_chain(self, game_state):
        run_lines = []
        moves = 0
        mover = game_state.current_player
        simple = True
        while True:
            caps = self._capturable_boxes(game_state)
            if not caps:
                break
            if len(caps) > 1:
                simple = False
                break
            free_line = caps[0]
            run_lines.append(free_line)
            game_state.execute_move(free_line)
            moves += 1
            if game_state.current_player != mover:
                break
        remaining_free = len(game_state.get_valid_moves())
        for _ in range(moves):
            game_state.undo_move()
        if not simple:
            return None, 0
        return run_lines, remaining_free

    def _get_capture_move(self, game_state):
        caps = self._capturable_boxes(game_state)
        if not caps:
            return None

        run_lines, remaining_free = self._trace_capture_chain(game_state)
        if run_lines is None:
            return caps[0]

        if len(run_lines) == 2 and remaining_free > 0:
            declining = run_lines[-1]
            if not self._is_capturing_move(game_state, declining):
                return declining

        return run_lines[0]

    def _get_safe_moves(self, game_state) -> list:
        """Filters out moves that would hand a 3-line box to the opponent."""
        valid_moves = game_state.get_valid_moves()
        safe_moves  = []
        for move in valid_moves:
            is_safe = True
            for box in game_state.get_boxes_of_line(move):
                lines = game_state.get_lines_of_box(box)
                drawn_count = sum(1 for l in lines if game_state.l[l] != 0)
                if drawn_count == 2:
                    is_safe = False
                    break
            if is_safe:
                safe_moves.append(move)
        return safe_moves

    def _find_chains(self, game_state) -> list:
        capturable = {}
        for r in range(game_state.SIZE):
            for c in range(game_state.SIZE):
                if game_state.b[r][c] == 0:
                    lines = game_state.get_lines_of_box((r, c))
                    free  = [l for l in lines if game_state.l[l] == 0]
                    if len(free) == 1:
                        capturable[(r, c)] = free[0]

        parent = {box: box for box in capturable}
        size   = {box: 1 for box in capturable}

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                if size[root_i] < size[root_j]:
                    parent[root_i] = root_j
                    size[root_j] += size[root_i]
                else:
                    parent[root_j] = root_i
                    size[root_i] += size[root_j]

        for r, c in capturable:
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                if (nr, nc) in capturable:
                    union((r, c), (nr, nc))

        components = {}
        for box in capturable:
            root = find(box)
            if root not in components:
                components[root] = []
            components[root].append((box, capturable[box]))

        return list(components.values())

    def _minimize_sacrifice(self, game_state, preferred_move, valid_moves):
        best_move = preferred_move
        best_cost = float('inf')
        for move in valid_moves:
            game_state.execute_move(move)
            cost = sum(len(c) for c in self._find_chains(game_state))
            game_state.undo_move()
            if cost < best_cost:
                best_cost = cost
                best_move = move
        return best_move if best_move is not None else valid_moves[0]

    def get_move(self, game_state) -> int:
        # 1. Take immediate captures if available
        capture_move = self._get_capture_move(game_state)
        if capture_move is not None:
            self.last_depth = 0
            self.last_simuls = 0
            self.last_time = 0.0
            return capture_move

        # 2. Run MCTS with a higher simulation count (e.g., 5000+)
        mcts = MCTS(
            n_simulations=self.mcts_parameters.get("n_simulations", 5000), 
            time_limit=self.mcts_parameters.get("time_limit", None),
            c_puct=self.mcts_parameters.get("c_puct", 1.4)
        )
        best_move, depth, simuls, t = mcts.search(game_state)
        
        self.last_depth = depth
        self.last_simuls = simuls
        self.last_time = t

        # 3. Trust the MCTS output (Removed the destructive safe_moves randomizer override)
        return best_move if best_move is not None else game_state.get_valid_moves()[0]