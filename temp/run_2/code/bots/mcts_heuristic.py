"""
MCTS with pure UCT (Upper Confidence bounds applied to Trees):
- No deepcopy: single shared game state walked with execute_move / undo_move
- Traditional Monte Carlo rollouts (no neural networks)
- Stores only MCTS statistics in AZNode
"""
import math
import random
import sys
import os
import numpy as np
from agent_interface import BaseAgent

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lookup_board import ZobristHash


# ---------------------------------------------------------------------------
# MCTS Tree Node
# ---------------------------------------------------------------------------

class AZNode:
    """
    Stores UCT statistics for one board position.
    """
    __slots__ = ('a', 'hash_val', 'parent', 'children', 'Q', 'N', 'unvisited_moves', 'is_expanded', 'is_solved', 'solved_value', 'val_for_parent')

    def __init__(self, parent, a: int, hash_val: int = 0, valid_moves: list = None):
        self.a        = a          # action that led to this node
        self.hash_val = hash_val   # Zobrist hash of this node's board state
        self.parent   = parent
        self.children: dict = {}   # action -> AZNode
        self.Q: dict  = {}         # action -> Q-value (from perspective of player who moves AT this node)
        self.N: dict  = {}         # action -> visit count
        self.unvisited_moves = list(valid_moves) if valid_moves is not None else []
        self.is_expanded = False
        
        # MCTS-Solver properties
        self.is_solved = False
        self.solved_value = 0.0
        self.val_for_parent = 0.0

    def get_child(self, a: int):
        return self.children.get(a, None)


# ---------------------------------------------------------------------------
# MCTS Engine (Pure UCT)
# ---------------------------------------------------------------------------

class MCTS:
    def __init__(self, mcts_parameters: dict):
        self.n_simulations  = mcts_parameters.get("n_simulations", 100)
        self.time_limit     = mcts_parameters.get("time_limit", None)
        self.c_puct         = mcts_parameters.get("c_puct", 1.0) # exploration constant

    def play(self, game_state, temp: int) -> list:
        import time
        # Zobrist hash for transposition table lookups
        zobrist   = ZobristHash(game_state.N_LINES)
        root_hash = zobrist.compute_initial_hash(game_state.l)

        root = AZNode(parent=None, a=None, hash_val=root_hash, valid_moves=game_state.get_valid_moves())

        if self.time_limit is not None:
            start_time = time.time()
            while time.time() - start_time < self.time_limit:
                self._search(game_state, root, zobrist)
        else:
            for _ in range(self.n_simulations):
                self._search(game_state, root, zobrist)

        # Build probability distribution based on visit counts
        counts = [root.N.get(a, 0) for a in range(game_state.N_LINES)]
        total_counts = sum(counts)

        if total_counts == 0:
            probs = [0.0] * game_state.N_LINES
            valid_moves = game_state.get_valid_moves()
            if valid_moves:
                u = 1.0 / len(valid_moves)
                for a in valid_moves:
                    probs[a] = u
            return probs

        if temp == 0:
            probs = [0] * len(counts)
            probs[int(np.argmax(counts))] = 1
            return probs

        probs     = [n ** (1.0 / temp) for n in counts]
        total_sum = float(sum(probs))
        if total_sum > 0:
            probs = [p / total_sum for p in probs]
        else:
            probs = [1.0 / len(probs)] * len(probs)
        return probs

    def _search(self, s, node: AZNode, zobrist: ZobristHash) -> float:
        """
        Runs a single UCT simulation step: selection, expansion, simulation, backpropagation.
        Returns value from the perspective of the current player of s.
        """
        if not s.is_running():
            result = s.result
            if s.current_player == result:
                val = 1.0
            else:
                val = 0.0 if result == 0 else -1.0
            node.is_solved = True
            node.solved_value = val
            return val
            
        if node.is_solved:
            return node.solved_value

        # --- Expansion ---
        if node.unvisited_moves:
            # Expand one unvisited action
            a = node.unvisited_moves.pop(random.randrange(len(node.unvisited_moves)))
            
            # Recurse: apply move, compute child hash, simulate/rollout, undo
            player_before = s.current_player
            s.execute_move(a)
            player_switched = (s.current_player != player_before)
            
            child_hash = zobrist.update_hash(node.hash_val, a)
            child = AZNode(parent=node, a=a, hash_val=child_hash, valid_moves=s.get_valid_moves())
            node.children[a] = child
            
            if not s.is_running():
                result = s.result
                if s.current_player == result:
                    val = 1.0
                else:
                    val = 0.0 if result == 0 else -1.0
                child.is_solved = True
                child.solved_value = val
                v_child = val
            else:
                v_child = self._rollout(s)
            
            s.undo_move()
            
            v_val = v_child if not player_switched else -v_child
            if child.is_solved:
                child.val_for_parent = v_val
            
            # Backup statistics
            node.N[a] = 1
            node.Q[a] = v_val
            
            self._backprop_solved(node)
            return v_val

        # --- Selection ---
        a = self._select(node, s.get_valid_moves())
        child = node.get_child(a)

        player_before = s.current_player
        s.execute_move(a)
        player_switched = (s.current_player != player_before)

        v_child = self._search(s, child, zobrist)
        s.undo_move()

        v_val = v_child if not player_switched else -v_child
        if child.is_solved:
            child.val_for_parent = v_val

        # --- Backup ---
        n = node.N[a]
        node.Q[a] = (n * node.Q[a] + v_val) / (n + 1)
        node.N[a] = n + 1
        
        self._backprop_solved(node)

        return v_val

    def _backprop_solved(self, node: AZNode):
        if node.is_solved:
            return
            
        has_unvisited = len(node.unvisited_moves) > 0
        all_children_solved = not has_unvisited
        
        best_val = -1.0
        
        for child in node.children.values():
            if not child.is_solved:
                all_children_solved = False
                continue
            
            val = child.val_for_parent
            if val == 1.0:
                # Guaranteed win found!
                node.is_solved = True
                node.solved_value = 1.0
                node.Q[child.a] = 1.0
                return
            
            if val > best_val:
                best_val = val
                
        if all_children_solved:
            node.is_solved = True
            node.solved_value = best_val

    def _select(self, node: AZNode, valid_moves: list) -> int:
        N_sum  = sum(node.N.values())
        log_N = math.log(N_sum) if N_sum > 0 else 0.0

        best_score = float('-inf')
        best_a     = valid_moves[0]
        for a in valid_moves:
            child = node.children.get(a)
            if child and child.is_solved:
                val = child.val_for_parent
                if val == 1.0:
                    return a # Immediate win, take it!
                elif val == -1.0:
                    ucb = -10000.0 # Avoid guaranteed losses
                else:
                    ucb = 0.0 # Draw
            else:
                q = node.Q.get(a, 0.0)
                n = node.N.get(a, 0)
                if n == 0:
                    ucb = float('inf')
                else:
                    ucb = q + self.c_puct * math.sqrt(log_N / n)

            if ucb > best_score:
                best_score = ucb
                best_a = a

        return best_a

    def _rollout(self, s) -> float:
        """
        Rollout simulation: plays heuristically to the end of the game, then restores the game state.
        """
        player = s.current_player
        moves_made = 0
        
        while s.is_running():
            valid = s.get_valid_moves()
            
            # Simple heuristic: try to grab a 3-line box if possible, otherwise random.
            move_to_play = None
            for r in range(s.SIZE):
                for c in range(s.SIZE):
                    if s.b[r][c] == 0:
                        lines = s.get_lines_of_box((r, c))
                        drawn = sum(1 for line in lines if s.l[line] != 0)
                        if drawn == 3:
                            for line in lines:
                                if s.l[line] == 0:
                                    move_to_play = line
                                    break
                        if move_to_play is not None:
                            break
                if move_to_play is not None:
                    break
                    
            if move_to_play is None:
                move_to_play = random.choice(valid)
                
            s.execute_move(move_to_play)
            moves_made += 1
            
        result = s.result
        val = 1.0 if result == player else (-1.0 if result != 0 else 0.0)
        
        for _ in range(moves_made):
            s.undo_move()
            
        return val


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
        mcts  = MCTS(self.mcts_parameters)
        probs = mcts.play(game_state, temp=0)

        # 4. Safety override: don't let MCTS pick a move that gives away a box if safe moves exist
        safe_moves = self._get_safe_moves(game_state)
        if safe_moves and len(safe_moves) < len(game_state.get_valid_moves()):
            safe_probs = np.zeros_like(probs)
            for move in safe_moves:
                safe_probs[move] = probs[move]
            if np.sum(safe_probs) > 0:
                return int(np.argmax(safe_probs))

        return int(np.argmax(probs))