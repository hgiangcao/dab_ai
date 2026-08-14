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
        # --- Early stopping parameters ---
        # Mathematical dominance: stop when N1 > N2 + N_remaining
        self.early_stop_dominance = mcts_parameters.get("early_stop_dominance", True)
        # Dynamic uncertainty: stop when N2/N1 < epsilon after min_sims simulations
        self.early_stop_entropy = mcts_parameters.get("early_stop_entropy", True)
        self.early_stop_entropy_eps = mcts_parameters.get("early_stop_entropy_eps", 0.02)
        self.early_stop_min_sims = mcts_parameters.get("early_stop_min_sims", 50)
        # --- Dynamic simulation scaling parameters ---
        # When enabled, n_simulations is multiplied by a phase-dependent factor
        # determined by the total number of moves already played on the board:
        #   move < 25          → 0.5×  (opening: save budget)
        #   25 ≤ move < 35     → 2.0×  (mid-game spike: critical decisions)
        #   35 ≤ move < 45     → 1.5×  (late-mid: still important)
        #   move ≥ 45          → 0.5×  (endgame: position usually forced)
        # Disabled by default; enable with dynamic_simulations=True.
        self.dynamic_simulations = mcts_parameters.get("dynamic_simulations", False)
        # Randomize simulation count by ±0-20% to increase exploration variety
        self.random_simulation = mcts_parameters.get("random_simulation", False)
        # Tree reuse: keep the root node between play() calls.
        # After each move is chosen we advance _root to the chosen child,
        # so all simulations done on that subtree are immediately available
        # on the next call instead of being discarded.
        self._root: AZNode = None

    def reset_tree(self):
        """Discard the cached tree (call at the start of every new game)."""
        self._root = None

    def _get_dynamic_n_simulations(self, game_state) -> int:
        """
        Return the effective simulation budget for this move, scaled by the
        current game phase (total lines already drawn on the board).

        Phase schedule (total moves played):
            < 25       → 0.50 × n_simulations  (opening)
            25 – 34    → 2.00 × n_simulations  (critical mid-game)
            35 – 44    → 1.50 × n_simulations  (late mid-game)
            ≥ 45       → 0.50 × n_simulations  (endgame / forced)
        """
        # Count drawn lines directly from the game-state array (no extra API needed)
        total_moves = int((game_state.l != 0).sum())

        if total_moves < 15:
            scale = 0.5
        elif total_moves < 40:
            scale = 2.0
        elif total_moves < 50:
            scale = 1.5
        else:
            scale = 1

        return max(1, int(self.n_simulations * scale))

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

        Parameters
        ----------
        game_state    : current DotsAndBoxesGame
        temp          : temperature for move selection
        add_root_noise: add Dirichlet noise at root (self-play only)
        last_action   : the move that was just played on the board
                        (used to advance the cached tree; pass None for the
                        very first move of a game or after reset_tree()).
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

        dirichlet_noise = None
        if add_root_noise and self.dirichlet_eps > 0:
            dirichlet_noise = np.zeros((root.s.N_LINES,), dtype=np.float64)
            dirichlet_noise[valid_moves] = np.random.dirichlet([self.dirichlet_alpha] * len(valid_moves))

        # --- n_simulations == 0: pure NN policy, no tree search ---
        # Query the network once and return its masked policy directly.
        # Board → NN → π → argmax/sample. No visit counts involved.
        if self.n_simulations == 0 and self.time_limit is None:
            raw_p, _ = self.model.predict(self._encode_state(root.s))
            nn_policy = self._mask_and_normalize_policy(raw_p, root.s.N_LINES, valid_moves)
            if add_root_noise and dirichlet_noise is not None:
                nn_policy = (1.0 - self.dirichlet_eps) * nn_policy + self.dirichlet_eps * dirichlet_noise
                nn_policy = self._mask_and_normalize_policy(nn_policy, root.s.N_LINES, valid_moves)
            self._root = root
            if temp == 0:
                probs = np.zeros(root.s.N_LINES, dtype=np.float64)
                probs[int(np.argmax(nn_policy))] = 1.0
                return probs.tolist()
            total = float(nn_policy.sum())
            return (nn_policy / total).tolist() if total > 0 else self._uniform_policy(root.s.N_LINES, valid_moves).tolist()

        # Expand root once before MCTS simulations
        if root.P is None:
            p, v_val = self.model.predict(self._encode_state(root.s))
            root.valid_moves = valid_moves
            root.P = self._mask_and_normalize_policy(p, root.s.N_LINES, valid_moves)

        if self.time_limit is not None:
            start_time = time.time()
            while time.time() - start_time < self.time_limit:
                self.search(root, is_root=True, dirichlet_noise=dirichlet_noise, current_depth=0)
        else:
            # Resolve effective simulation budget for this move
            n_sims = (
                self._get_dynamic_n_simulations(game_state)
                if self.dynamic_simulations
                else self.n_simulations
            )
            
            if self.random_simulation:
                import random
                variance = random.uniform(-0.2, 0.2)
                n_sims = max(1, int(n_sims * (1 + variance)))

            for sim_idx in range(n_sims):
                self.search(root, is_root=True, dirichlet_noise=dirichlet_noise, current_depth=0)

                # --- Early stopping (only meaningful when ≥2 moves exist) ---
                if root.valid_moves is not None and len(root.valid_moves) >= 2:
                    # Collect the two highest visit counts in a single pass
                    n1 = n2 = 0
                    for a in root.valid_moves:
                        n = root.N.get(a, 0)
                        if n > n1:
                            n2, n1 = n1, n
                        elif n > n2:
                            n2 = n

                    sims_done = sim_idx + 1
                    n_remaining = n_sims - sims_done

                    # 1. Mathematical dominance: N1 > N2 + N_remaining
                    #    Even giving all remaining budget to the runner-up, it can't overtake.
                    if self.early_stop_dominance and n1 > n2 + n_remaining:
                        break

                    # 2. Dynamic uncertainty / entropy threshold: N2/N1 < epsilon
                    #    The search has converged; the second-best move is negligible.
                    if (self.early_stop_entropy
                            and sims_done >= self.early_stop_min_sims
                            and n1 > 0
                            and n2 / n1 < self.early_stop_entropy_eps):
                        break

        counts = np.array([root.N.get(a, 0) for a in range(root.s.N_LINES)], dtype=np.float64)
        counts_sum = float(counts.sum())

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
            p, v_val = self.model.predict(self._encode_state(node.s))
            v = float(np.asarray(v_val).reshape(-1)[0])

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

        N_sqrt = math.sqrt(node.total_N + 1.0)

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

    def _encode_state(self, s) -> np.ndarray:
        """Encode a game state into the (4, size+1, size+1) float32 tensor expected by the NN.

        Channels:
            0 — horizontal lines (current player canonical)
            1 — vertical   lines (current player canonical)
            2 — boxes owned by current player
            3 — boxes owned by opponent
        """
        size = s.SIZE
        sp1 = size + 1

        # Reuse a pre-allocated buffer to avoid per-call numpy allocations.
        buf = getattr(self, '_board_buf', None)
        if buf is None or buf.shape != (4, sp1, sp1):
            self._board_buf = np.zeros((4, sp1, sp1), dtype=np.float32)
            buf = self._board_buf
        else:
            buf[:] = 0.0

        canonical_lines = s.get_canonical_lines()
        h, v = s.l_to_h_v(canonical_lines)
        buf[0, :sp1, :size] = h          # horizontal lines
        buf[1, :size, :sp1] = v          # vertical lines

        canonical_boxes = s.get_canonical_boxes()
        buf[2, :size, :size] = np.where(canonical_boxes == 1,  1.0, 0.0)  # current-player boxes
        buf[3, :size, :size] = np.where(canonical_boxes == -1, 1.0, 0.0)  # opponent boxes

        return buf

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
        if node.P is None:
            # Node was never expanded (e.g. n_simulations=0). Return uniform over valid moves
            # so we never pick an already-drawn line via np.argmax(None) == 0.
            return self._uniform_policy(node.s.N_LINES, valid_moves)
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
