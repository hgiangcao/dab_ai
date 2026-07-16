import time
import math
import numpy as np

class AZNode:
    __slots__ = ('a', 's', 'children', 'Q', 'N', 'total_N', 'P', 'valid_moves')

    def __init__(self, parent, s, a: int):
        self.a = a
        self.s = s
        self.children = {}      # dict keyed by action: O(1) lookup vs O(n) list scan
        self.Q = {}
        self.N = {}
        self.total_N = 0        # cached sum of N values; updated in backup() for O(1) sqrt
        self.P = None
        self.valid_moves = None # cached after leaf expansion; avoids repeated get_valid_moves()
        if parent is not None:
            parent.children[a] = self

    def get_child_by_move(self, a: int):
        return self.children.get(a)


class MCTS:
    def __init__(self, model, mcts_parameters: dict):
        self.model = model
        self.n_simulations = mcts_parameters.get("n_simulations", 200)
        self.c_puct = mcts_parameters["c_puct"]
        self.dirichlet_eps = mcts_parameters.get("dirichlet_eps", 0.0)
        self.dirichlet_alpha = mcts_parameters.get("dirichlet_alpha", 0.0)
        self.time_limit = mcts_parameters.get("time_limit", None)
        self.max_depth_reached = 0

    def play(self, game_state, temp: int, add_root_noise: bool = False) -> list:
        root = AZNode(parent=None, s=game_state.clone(track_history=False), a=None)
        valid_moves = root.s.get_valid_moves()
        self.max_depth_reached = 0

        if not valid_moves:
            return [0.0] * root.s.N_LINES

        dirichlet_noise = None
        if add_root_noise and self.dirichlet_eps > 0:
            dirichlet_noise = np.zeros((root.s.N_LINES,), dtype=np.float64)
            dirichlet_noise[valid_moves] = np.random.dirichlet([self.dirichlet_alpha] * len(valid_moves))

        if self.time_limit is not None:
            start_time = time.time()
            while time.time() - start_time < self.time_limit:
                self.search(root, is_root=True, dirichlet_noise=dirichlet_noise, current_depth=0)
        else:
            for _ in range(self.n_simulations):
                self.search(root, is_root=True, dirichlet_noise=dirichlet_noise, current_depth=0)

        counts = np.array([root.N.get(a, 0) for a in range(root.s.N_LINES)], dtype=np.float64)
        counts_sum = float(counts.sum())

        if counts_sum == 0:
            fallback = self._root_policy(root, valid_moves, dirichlet_noise)
            if temp == 0:
                probs = np.zeros(root.s.N_LINES, dtype=np.float64)
                probs[int(np.argmax(fallback))] = 1.0
                return probs.tolist()
            return fallback.tolist()

        if temp == 0:
            probs = np.zeros(root.s.N_LINES, dtype=np.float64)
            probs[int(np.argmax(counts))] = 1.0
            return probs.tolist()

        probs = counts ** (1.0 / temp)
        total_sum = float(sum(probs))
        if total_sum > 0:
            return (probs / total_sum).tolist()
        return self._uniform_policy(root.s.N_LINES, valid_moves).tolist()

    def search(self, node: AZNode, is_root: bool = False, dirichlet_noise: np.ndarray = None, current_depth: int = 0) -> float:
        self.max_depth_reached = max(self.max_depth_reached, current_depth)

        if not node.s.is_running():
            result = node.s.result
            if node.s.current_player == result:
                return 1.0
            return 0.0 if result == 0 else -1.0

        # Leaf expansion: call NN once, cache policy and valid moves on this node
        if node.P is None:
            h, v = node.s.l_to_h_v(node.s.get_canonical_lines())
            size = node.s.SIZE

            c1 = np.zeros((size+1, size+1))
            c1[:size+1, :size] = h

            c2 = np.zeros((size+1, size+1))
            c2[:size, :size+1] = v

            canonical_boxes = node.s.get_canonical_boxes()
            p1_boxes = np.where(canonical_boxes == 1, 1.0, 0.0)
            c3 = np.zeros((size+1, size+1))
            c3[:size, :size] = p1_boxes

            p2_boxes = np.where(canonical_boxes == -1, 1.0, 0.0)
            c4 = np.zeros((size+1, size+1))
            c4[:size, :size] = p2_boxes

            stacked_board = np.stack([c1, c2, c3, c4], axis=0)

            p, v = self.model.predict(stacked_board)
            v = float(np.asarray(v).reshape(-1)[0])

            # Cache valid_moves on the node — avoid redundant get_valid_moves() calls
            node.valid_moves = node.s.get_valid_moves()
            node.P = self._mask_and_normalize_policy(p, node.s.N_LINES, node.valid_moves)
            return v

        a = self.select(node, is_root, dirichlet_noise)
        child = node.children.get(a)  # O(1) dict lookup

        if child is None:
            child = self.expand(node, a)

        v_child = self.search(child, is_root=False, dirichlet_noise=None, current_depth=current_depth + 1)
        v = v_child if node.s.current_player == child.s.current_player else -v_child

        self.backup(node, a, v)
        return v

    def select(self, node: AZNode, is_root: bool, dirichlet_noise: np.ndarray) -> int:
        maximum = float('-inf')
        a_max = -1

        # O(1) total_N lookup instead of O(children) sum()
        N_sqrt = math.sqrt(node.total_N) if node.total_N > 0 else 1

        # Use cached valid_moves — no repeated get_valid_moves() call
        valid_moves = node.valid_moves
        P = self._root_policy(node, valid_moves, dirichlet_noise) if is_root else node.P

        for a in valid_moves:
            p = P[a]
            q = node.Q.get(a, 0.0)
            n = node.N.get(a, 0)

            u = self.c_puct * p * N_sqrt / (1 + n)
            if q + u > maximum:
                maximum = q + u
                a_max = a
        if a_max == -1:
            raise RuntimeError("MCTS select called with no valid moves")
        return a_max

    def expand(self, node: AZNode, a: int) -> AZNode:
        # MCTS never calls undo_move() on tree nodes, so skip copying history
        s = node.s.clone(track_history=False)
        s.execute_move(a)
        return AZNode(parent=node, s=s, a=a)

    def backup(self, node: AZNode, a: int, v: float):
        if a not in node.N:
            node.Q[a] = v
            node.N[a] = 1
        else:
            n = node.N[a]
            node.Q[a] = (n * node.Q[a] + v) / (n + 1)
            node.N[a] = n + 1
        # Update cached total_N for O(1) sqrt in select()
        node.total_N += 1

    @staticmethod
    def _uniform_policy(n_actions: int, valid_moves: list) -> np.ndarray:
        probs = np.zeros(n_actions, dtype=np.float64)
        if valid_moves:
            probs[valid_moves] = 1.0 / len(valid_moves)
        return probs

    def _mask_and_normalize_policy(self, policy, n_actions: int, valid_moves: list) -> np.ndarray:
        if policy is None:
            return self._uniform_policy(n_actions, valid_moves)

        policy = np.asarray(policy, dtype=np.float64).reshape(-1)
        if policy.shape[0] != n_actions:
            raise ValueError(f"Model policy has length {policy.shape[0]}, expected {n_actions}")

        probs = np.zeros(n_actions, dtype=np.float64)
        policy = np.where(np.isfinite(policy), policy, 0.0)
        policy = np.maximum(policy, 0.0)
        probs[valid_moves] = policy[valid_moves]

        total = float(probs.sum())
        if total > 0:
            return probs / total
        return self._uniform_policy(n_actions, valid_moves)

    def _root_policy(self, node: AZNode, valid_moves: list, dirichlet_noise: np.ndarray = None) -> np.ndarray:
        # node.P is already normalized and masked from leaf expansion.
        # Only re-mix if we have Dirichlet noise to add.
        if dirichlet_noise is None:
            return node.P
        base_policy = self._mask_and_normalize_policy(node.P, node.s.N_LINES, valid_moves)
        policy = (1.0 - self.dirichlet_eps) * base_policy + self.dirichlet_eps * dirichlet_noise
        return self._mask_and_normalize_policy(policy, node.s.N_LINES, valid_moves)


from agent_interface import BaseAgent

class MCTSAgent(BaseAgent):
    def __init__(self, name: str, model, mcts_parameters: dict):
        super().__init__(name)
        self.mcts = MCTS(model, mcts_parameters)

    def get_move(self, game_state) -> int:
        probs = self.mcts.play(game_state, temp=0, add_root_noise=False)
        return int(np.argmax(probs))
