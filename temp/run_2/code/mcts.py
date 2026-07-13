import copy
import math
import numpy as np

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
        self.model = model  # This is your NNetWrapper instance
        self.n_simulations = mcts_parameters["n_simulations"]
        self.c_puct = mcts_parameters["c_puct"]
        self.dirichlet_eps = mcts_parameters["dirichlet_eps"]
        self.dirichlet_alpha = mcts_parameters["dirichlet_alpha"]

    def play(self, game_state, temp: int) -> list:
        root = AZNode(parent=None, s=copy.deepcopy(game_state), a=None)
        valid_moves = root.s.get_valid_moves()
        
        # FIX: Generate Dirichlet noise ONCE per move decision, not per simulation step
        dirichlet_noise = np.zeros((root.s.N_LINES,), dtype=np.float32)
        if len(valid_moves) > 0:
            dirichlet_noise[valid_moves] = np.random.dirichlet([self.dirichlet_alpha] * len(valid_moves))

        # Track maximum depth reached in this play step
        self.max_depth_reached = 0
        for _ in range(self.n_simulations):
            self.search(root, is_root=True, dirichlet_noise=dirichlet_noise, current_depth=0)

        counts = [root.N[a] if a in root.N else 0 for a in range(root.s.N_LINES)]

        if temp == 0:
            probs = [0] * len(counts)
            probs[np.array(counts).argmax()] = 1
            return probs

        probs = [n ** (1. / temp) for n in counts]
        total_sum = float(sum(probs))
        probs = [p / total_sum for p in probs] if total_sum > 0 else [1.0/len(probs)] * len(probs)
        return probs

    def search(self, node: AZNode, is_root: bool = False, dirichlet_noise: np.ndarray = None, current_depth: int = 0) -> float:
        self.max_depth_reached = max(self.max_depth_reached, current_depth)
        
        if not node.s.is_running():
            result = node.s.result
            if node.s.current_player == result:
                return 1.0
            return 0.0 if result == 0 else -1.0

        # Leaf expansion
        if node.P is None:
            # FIX: Format tracking states into a 4-channel tensor matching model.py input profile
            h, v = node.s.l_to_h_v(node.s.get_canonical_lines())
            # Create matching dummy 4-plane feature representation, padded to (size+1)x(size+1)
            size = node.s.SIZE
            
            c1 = np.zeros((size+1, size+1))
            c1[:size+1, :size] = h
            
            c2 = np.zeros((size+1, size+1))
            c2[:size, :size+1] = v
            
            p1_boxes = np.where(node.s.get_canonical_boxes() == 1, 1.0, 0.0)
            c3 = np.zeros((size+1, size+1))
            c3[:size, :size] = p1_boxes
            
            p2_boxes = np.where(node.s.get_canonical_boxes() == -1, 1.0, 0.0)
            c4 = np.zeros((size+1, size+1))
            c4[:size, :size] = p2_boxes
            
            stacked_board = np.stack([c1, c2, c3, c4], axis=0)
            
            # Call our model wrapper's predict function
            p, v = self.model.predict(stacked_board)
            node.P = p
            return v

        a = self.select(node, is_root, dirichlet_noise)
        child = node.get_child_by_move(a)
        
        if child is None:
            child = self.expand(node, a)

        v_child = self.search(child, is_root=False, dirichlet_noise=None, current_depth=current_depth + 1)
        v = v_child if node.s.current_player == child.s.current_player else -v_child

        self.backup(node, a, v)
        return v

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


import numpy as np
from agent_interface import BaseAgent

class MCTSAgent(BaseAgent):
    def __init__(self, name: str, model, mcts_parameters: dict):
        super().__init__(name)
        self.mcts = MCTS(model, mcts_parameters)

    def get_move(self, game_state) -> int:
        # Use temp=0 to always select the strongest move during evaluation/play
        probs = self.mcts.play(game_state, temp=0)
        return int(np.argmax(probs))