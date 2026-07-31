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
        # Tree reuse: keep the root node between play() calls.
        # After each move is chosen we advance _root to the chosen child,
        # so all simulations done on that subtree are immediately available
        # on the next call instead of being discarded.
        self._root: AZNode = None

    def reset_tree(self):
        """Discard the cached tree (call at the start of every new game)."""
        self._root = None

    def _advance_root(self, last_action: int, game_state) -> AZNode:
        """
        Advance the cached root to the child corresponding to last_action.
        If the child doesn't exist yet (opponent played an unexplored move),
        build a fresh root from the current game state instead.
        Returns the new root node.
        """
        if self._root is not None and last_action is not None:
            child = self._root.children.get(last_action)
            if child is not None:
                # Detach from parent to allow GC of the old subtree
                child.a = None
                return child

        # Cache miss or no previous tree: create a fresh root
        return AZNode(parent=None, s=game_state.clone(track_history=False), a=None)

    def play(self, game_state, temp: int, add_root_noise: bool = False,
             last_action: int = None) -> list:
        """
        Run MCTS from the current game_state and return a policy vector.

        Before MCTS we apply two rule-based shortcuts:

        Rule 1 – Forced completion (priority 0):
            If any box has 3 edges occupied, complete it immediately.
            No search needed — this is always correct.

        Rule 2 – Safe isolated move (priority 1):
            Among edges adjacent only to 0-edge boxes AND where both
            endpoints have degree 0 (drawing it creates zero new exposure),
            pick uniformly at random.  These moves are strategically neutral
            and waste no MCTS budget.

        Rule 3 – MCTS with tier-biased prior:
            Explore 0-edge-adjacent moves first, 1/2-edge-adjacent moves
            later, using a multiplicative prior boost applied on top of the
            NN policy.  Within the same tier MCTS UCB takes over.
        """
        # Advance (or create) the root using the tree from the previous move
        root = self._advance_root(last_action, game_state)

        # Safety check: if the cached state doesn't match, rebuild from scratch
        if root.s != game_state:
            root = AZNode(parent=None, s=game_state.clone(track_history=False), a=None)

        valid_moves = root.s.get_valid_moves()
        self.max_depth_reached = 0

        if not valid_moves:
            self._root = root
            return [0.0] * root.s.N_LINES

        # ── Rule 1: forced completion ─────────────────────────────────────────
        forced = self._forced_completion_moves(root.s)
        if forced:
            probs = np.zeros(root.s.N_LINES, dtype=np.float64)
            # Pick one forced move (first is fine — all complete a box)
            probs[forced[0]] = 1.0
            self.max_depth_reached = -1   # sentinel: rule-based, exclude from avg depth
            self._root = root
            return probs.tolist()

        # ── Rule 2: safe isolated move ─────────────────────────────────────────
        isolated = self._safe_isolated_moves(root.s)
        if isolated:
            probs = np.zeros(root.s.N_LINES, dtype=np.float64)
            chosen = isolated[int(np.random.randint(len(isolated)))]
            probs[chosen] = 1.0
            self.max_depth_reached = -1   # sentinel: rule-based, exclude from avg depth
            self._root = root
            return probs.tolist()

        # ── Rule 3: MCTS with tier-biased prior ────────────────────────────────
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
                self._root = root
                return probs.tolist()
            self._root = root
            return fallback.tolist()

        if temp == 0:
            probs = np.zeros(root.s.N_LINES, dtype=np.float64)
            probs[int(np.argmax(counts))] = 1.0
            self._root = root
            return probs.tolist()

        probs = counts ** (1.0 / temp)
        total_sum = float(sum(probs))

        self._root = root
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
        # Apply tier-based multiplier so 0-edge moves are explored first,
        # then 1/2-edge moves — within the same tier MCTS UCB governs.
        P = node.P
        tier_weights = self._tier_weights(node.s, valid_moves, node.s.N_LINES)
        P = P * tier_weights
        total = float(P.sum())
        if total > 0:
            P = P / total
        else:
            P = self._uniform_policy(node.s.N_LINES, valid_moves)
        if dirichlet_noise is None:
            return P
        policy = (1.0 - self.dirichlet_eps) * P + self.dirichlet_eps * dirichlet_noise
        return self._mask_and_normalize_policy(policy, node.s.N_LINES, valid_moves)

    # ------------------------------------------------------------------
    # Rule-based helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _forced_completion_moves(game_state) -> list:
        """
        Return all valid line indices that would complete a 3-edge box.
        These are deterministically correct moves — no search needed.
        """
        bm = game_state.board_manager
        lm = game_state._line_to_bm_edge
        forced = []
        three_edge_box_ids = bm.box_by_edge_count[3]   # O(1) set lookup
        if not three_edge_box_ids:
            return forced
        # Collect unoccupied edges of every 3-edge box
        for box_id in three_edge_box_ids:
            box = bm.boxes[box_id]
            for edge in box.edges.values():
                if not edge.occupied:
                    # Find its line index (reverse map via id)
                    forced.append(edge.id)  # edge.id == position in bm.edges
        # Convert bm edge IDs to game line indices
        edge_id_to_line = {e.id: line for line, e in lm.items()}
        return [edge_id_to_line[eid] for eid in forced if eid in edge_id_to_line]

    @staticmethod
    def _safe_isolated_moves(game_state) -> list:
        """
        Return line indices that are 'safe isolated' moves:
          - The edge is unoccupied
          - Both its endpoints have degree 0 (drawing it creates no new
            chain exposure)
          - ALL adjacent boxes have 0 edges currently
        These moves are strategically neutral early-game plays.
        """
        bm  = game_state.board_manager
        lm  = game_state._line_to_bm_edge
        safe = []
        edge_id_to_line = {e.id: line for line, e in lm.items()}
        for edge_id in bm.available_edges:
            edge = bm.edges[edge_id]
            # Isolated: both endpoints untouched
            if edge.p1.degree != 0 or edge.p2.degree != 0:
                continue
            # All adjacent boxes must have 0 edges
            if all(box.edge_count == 0 for box in edge.boxes):
                line_idx = edge_id_to_line.get(edge_id)
                if line_idx is not None:
                    safe.append(line_idx)
        return safe

    @staticmethod
    def _tier_weights(game_state, valid_moves: list, n_lines: int) -> np.ndarray:
        """
        Build a per-move multiplicative weight vector based on the maximum
        edge-count of adjacent boxes:

          Tier 0  (0-edge adjacent)  → weight 4.0   (explore first)
          Tier 1  (1-edge adjacent)  → weight 1.0
          Tier 2  (2-edge adjacent)  → weight 1.0
          Tier 3+ (3-edge adjacent)  → weight 0.0   (should be forced; skip)

        Weights are applied multiplicatively on top of the NN prior so that
        within the same tier, the NN policy and UCB govern selection.
        """
        TIER_W = {0: 4.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0}
        bm  = game_state.board_manager
        lm  = game_state._line_to_bm_edge
        weights = np.zeros(n_lines, dtype=np.float64)
        for line_idx in valid_moves:
            edge = lm[line_idx]
            max_ec = max((box.edge_count for box in edge.boxes), default=0)
            weights[line_idx] = TIER_W.get(max_ec, 1.0)
        return weights


from agent_interface import BaseAgent

class MCTSAgent(BaseAgent):
    def __init__(self, name: str, model, mcts_parameters: dict):
        super().__init__(name)
        self.mcts = MCTS(model, mcts_parameters)
        self._last_action: int = None

    def reset(self):
        """Call at the start of each new game to discard stale tree state."""
        self.mcts.reset_tree()
        self._last_action = None

    def get_move(self, game_state) -> int:
        probs = self.mcts.play(game_state, temp=0, add_root_noise=False,
                               last_action=self._last_action)
        action = int(np.argmax(probs))
        self._last_action = action
        return action
