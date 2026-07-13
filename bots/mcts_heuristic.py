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
import numpy as np
from agent_interface import BaseAgent

class Node:
    __slots__ = ('parent', 'move', 'children', 'visits', 'value_sum', 'player_to_move', 'untried_moves', 'terminal')

    def __init__(self, parent, move, player_to_move, valid_moves, terminal):
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.player_to_move = player_to_move
        self.terminal = terminal
        self.untried_moves = valid_moves if valid_moves is not None else []

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
                game_state.execute_move(move)
                moves_made += 1
                
                terminal = not game_state.is_running()
                valid_for_child = self._order_expansion_moves(game_state, game_state.get_valid_moves()) if not terminal else []
                child = Node(parent=node, move=move, player_to_move=game_state.current_player,
                             valid_moves=valid_for_child,
                             terminal=terminal)
                node.children.append(child)
                node = child
                
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
                
        # Return best child
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            return best_child.move
        return None

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
                ucb = q + self.c_puct * math.sqrt(log_n / child.visits)
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
            # 2. Safe moves
            if move is None:
                safe_moves = self._find_safe(s, valid)
                if safe_moves:
                    move = random.choice(safe_moves)
                else:
                    move = random.choice(valid)
                    
            s.execute_move(move)
            moves_made += 1
            
        result = s.result
        if result == rollout_player:
            val = 1.0
        elif result == 0:
            val = 0.0
        else:
            val = -1.0
            
        return val, moves_made

    def _find_capture(self, s, valid_moves):
        for move in valid_moves:
            for box in s.get_boxes_of_line(move):
                lines = s.get_lines_of_box(box)
                drawn = sum(1 for ln in lines if s.l[ln] != 0)
                if drawn == 3:
                    return move
        return None

    def _find_safe(self, s, valid_moves):
        safe_moves = []
        for move in valid_moves:
            is_safe = True
            for box in s.get_boxes_of_line(move):
                lines = s.get_lines_of_box(box)
                drawn = sum(1 for ln in lines if s.l[ln] != 0)
                if drawn == 2:
                    is_safe = False
                    break
            if is_safe:
                safe_moves.append(move)
        return safe_moves
        
    def _order_expansion_moves(self, s, valid_moves):
        if not valid_moves: return []
        captures = []
        safe = []
        remaining = []
        for move in valid_moves:
            is_capture = False
            for box in s.get_boxes_of_line(move):
                lines = s.get_lines_of_box(box)
                if sum(1 for ln in lines if s.l[ln] != 0) == 3:
                    is_capture = True
                    break
            if is_capture:
                captures.append(move)
                continue
                
            is_safe = True
            for box in s.get_boxes_of_line(move):
                lines = s.get_lines_of_box(box)
                if sum(1 for ln in lines if s.l[ln] != 0) == 2:
                    is_safe = False
                    break
            if is_safe:
                safe.append(move)
            else:
                remaining.append(move)
                
        random.shuffle(captures)
        random.shuffle(safe)
        random.shuffle(remaining)
        return captures + safe + remaining


# ---------------------------------------------------------------------------
# Heuristic Agent wrapping MCTS
# ---------------------------------------------------------------------------

class MCTSHeuristicAgent(BaseAgent):
    def __init__(self, name: str, mcts_parameters: dict):
        super().__init__(name)
        self.mcts_parameters = mcts_parameters

    def _get_greedy_move(self, game_state) -> int:
        """Returns a move that instantly completes a 3-line box."""
        for r in range(game_state.SIZE):
            for c in range(game_state.SIZE):
                if game_state.b[r][c] == 0:
                    lines = game_state.get_lines_of_box((r, c))
                    drawn_count = sum(1 for line in lines if game_state.l[line] != 0)
                    if drawn_count == 3:
                        for line in lines:
                            if game_state.l[line] == 0:
                                return line
        return None

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

    def _get_double_cross_move(self, game_state) -> int:
        chains = self._find_chains(game_state)

        for chain in chains:
            if len(chain) >= 3:
                return chain[0][1]

        safe_moves = self._get_safe_moves(game_state)
        if not safe_moves:
            valid_moves = game_state.get_valid_moves()
            best_move   = None
            best_cost   = float('inf')
            for move in valid_moves:
                game_state.execute_move(move)
                cost = sum(len(c) for c in self._find_chains(game_state))
                game_state.undo_move()
                if cost < best_cost:
                    best_cost = cost
                    best_move = move
            return best_move

        return None

    def get_move(self, game_state) -> int:
        # 1. Greedy: capture any immediate boxes
        greedy_move = self._get_greedy_move(game_state)
        if greedy_move is not None:
            return greedy_move

        # 2. Chain-Handling (Double-Cross)
        chain_move = self._get_double_cross_move(game_state)
        if chain_move is not None:
            return chain_move

        # 3. Hybrid Endgame: Delegate to Alpha-Beta if <= 12 moves remaining
        valid_moves = game_state.get_valid_moves()
        if len(valid_moves) <= 12:
            from bots.alpha_beta import AlphaBetaPlayer
            ab = AlphaBetaPlayer(time_limit=self.mcts_parameters.get("time_limit", 2.0))
            return ab.get_move(game_state)

        # 4. Pure MCTS play
        mcts = MCTS(
            n_simulations=self.mcts_parameters.get("n_simulations", 100),
            time_limit=self.mcts_parameters.get("time_limit", None),
            c_puct=self.mcts_parameters.get("c_puct", 1.4)
        )
        
        best_move = mcts.search(game_state)

        # 5. Safety override: don't let MCTS pick a move that gives away a box if safe moves exist
        safe_moves = self._get_safe_moves(game_state)
        if safe_moves and len(safe_moves) < len(game_state.get_valid_moves()):
            if best_move not in safe_moves:
                return random.choice(safe_moves)

        return best_move if best_move is not None else valid_moves[0]