import copy
import math
import numpy as np
from agent_interface import BaseAgent

class AZNode:
    def __init__(self, parent, s, a: int):
        self.a = a
        self.s = s
        self.children = []
        self.Q = {}
        self.N = {}
        self.P = None
        if parent is not None:
            parent.children.append(self)

    def get_child_by_move(self, a: int):
        for child in self.children:
            if child.a == a:
                return child
        return None

class MCTS:
    def __init__(self, model, mcts_parameters: dict):
        self.model = model
        self.n_simulations = mcts_parameters["n_simulations"]
        self.c_puct = mcts_parameters["c_puct"]
        self.dirichlet_eps = mcts_parameters["dirichlet_eps"]
        self.dirichlet_alpha = mcts_parameters["dirichlet_alpha"]

    def play(self, game_state, temp: int) -> list:
        root = AZNode(parent=None, s=copy.deepcopy(game_state), a=None)
        valid_moves = root.s.get_valid_moves()
        
        dirichlet_noise = np.zeros((root.s.N_LINES,), dtype=np.float32)
        if len(valid_moves) > 0:
            dirichlet_noise[valid_moves] = np.random.dirichlet([self.dirichlet_alpha] * len(valid_moves))

        for _ in range(self.n_simulations):
            self.search(root, is_root=True, dirichlet_noise=dirichlet_noise)

        counts = [root.N[a] if a in root.N else 0 for a in range(root.s.N_LINES)]

        if temp == 0:
            probs = [0] * len(counts)
            probs[np.array(counts).argmax()] = 1
            return probs

        probs = [n ** (1. / temp) for n in counts]
        total_sum = float(sum(probs))
        probs = [p / total_sum for p in probs] if total_sum > 0 else [1.0/len(probs)] * len(probs)
        return probs

    def search(self, node: AZNode, is_root: bool = False, dirichlet_noise: np.ndarray = None) -> float:
        if not node.s.is_running():
            result = node.s.result
            if node.s.current_player == result:
                return 1.0
            return 0.0 if result == 0 else -1.0

        if node.P is None:
            h, v = node.s.l_to_h_v(node.s.get_canonical_lines())
            p1_boxes = np.where(node.s.get_canonical_boxes() == 1, 1.0, 0.0)
            p2_boxes = np.where(node.s.get_canonical_boxes() == -1, 1.0, 0.0)
            
            stacked_board = np.stack([h[:-1, :], v[:, :-1], p1_boxes, p2_boxes], axis=0)
            p, v_val = self.model.predict(stacked_board)
            node.P = p
            return v_val

        a = self.select(node, is_root, dirichlet_noise)
        child = node.get_child_by_move(a)
        
        if child is None:
            child = self.expand(node, a)

        v_child = self.search(child, is_root=False, dirichlet_noise=None)
        v_val = v_child if node.s.current_player == child.s.current_player else -v_child

        self.backup(node, a, v_val)
        return v_val

    def select(self, node: AZNode, is_root: bool, dirichlet_noise: np.ndarray) -> int:
        maximum = float('-inf')
        a_max = -1
        N_sum = sum(node.N.values())
        N_sqrt = math.sqrt(N_sum) if N_sum > 0 else 1

        P = node.P if not is_root else (1 - self.dirichlet_eps) * node.P + self.dirichlet_eps * dirichlet_noise

        for a in node.s.get_valid_moves():
            p = P[a]
            q = node.Q[a] if a in node.N else 0.0
            n = node.N[a] if a in node.N else 0
            
            u = self.c_puct * p * N_sqrt / (1 + n)
            if q + u > maximum:
                maximum = q + u
                a_max = a
        return a_max

    def expand(self, node: AZNode, a: int) -> AZNode:
        s = copy.deepcopy(node.s)
        s.execute_move(a)
        return AZNode(parent=node, s=s, a=a)

    def backup(self, node: AZNode, a: int, v: float):
        if a not in node.N:
            node.Q[a] = v
            node.N[a] = 1
        else:
            n = node.N[a]
            self.backup_q_update(node, a, v, n)

    def backup_q_update(self, node, a, v, n):
        node.Q[a] = (n * node.Q[a] + v) / (n + 1)
        node.N[a] += 1


class MCTSHeuristicAgent(BaseAgent):
    def __init__(self, name: str, model, mcts_parameters: dict):
        super().__init__(name)
        self.model = model
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
        safe_moves = []
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

    def get_move(self, game_state) -> int:
        # 1. Greedy Rule: Take any free boxes immediately
        greedy_move = self._get_greedy_move(game_state)
        if greedy_move is not None:
            return greedy_move

        # 2. Run standard MCTS
        mcts = MCTS(self.model, self.mcts_parameters)
        probs = mcts.play(game_state, temp=0)

        # 3. Safety Rule: Override MCTS if it tries to give away a free box
        safe_moves = self._get_safe_moves(game_state)
        if safe_moves and len(safe_moves) < len(game_state.get_valid_moves()):
            safe_probs = np.zeros_like(probs)
            for move in safe_moves:
                safe_probs[move] = probs[move]
            
            if np.sum(safe_probs) > 0:
                return int(np.argmax(safe_probs))

        return int(np.argmax(probs))