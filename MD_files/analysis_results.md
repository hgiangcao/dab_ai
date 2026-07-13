# Dots-and-Boxes AI: Comprehensive Code Review

Based on a thorough inspection of the repository—specifically `bots/alpha_beta.py`, `game.py`, and `lookup_board.py`—here is the evaluation of your classical game-playing engine against standard adversarial search techniques.

## 1. Search Techniques
| Feature | Status | Comments |
|---------|--------|----------|
| **Minimax** | ✔ Fully implemented | Underpins the search algorithm natively. |
| **Negamax** | ✘ Missing | The engine uses explicit `maximize` boolean flags and distinct `if maximize:` / `else:` branches instead of the cleaner Negamax formulation (where you return `-search(..., -alpha, -beta)`). |
| **Alpha-Beta pruning** | ✔ Fully implemented | Correctly prunes branches when `v_best > beta` or `v_best < alpha` (Lines 195, 202). |
| **Principal Variation Search (PVS) / NegaScout** | ✘ Missing | Only standard Alpha-Beta windows are used. Zero-window searches are not implemented to optimize PV lines. |
| **Iterative Deepening** | ✔ Fully implemented | Uses a `for current_depth in range(1, max_depth + 1):` loop combined with a time limit. It correctly catches a `TimeoutError` to break and return the best move found at the last fully completed depth. |

## 2. State Representation
| Feature | Status | Comments |
|---------|--------|----------|
| **Incremental board updates** | ✔ Fully implemented | Modifies the single `game.py` state using `self.l[line] = 1.0` and `self.b[r][c] = player`. |
| **Undo/redo moves** | ✔ Fully implemented | Uses an explicit `_history` stack to `execute_move()` and `undo_move()`, successfully avoiding highly expensive deepcopies. |
| **Efficient move generation** | ◐ Partially implemented | `get_valid_moves()` uses `np.where(self.l == 0)[0].tolist()`. While this avoids looping manually over a list in Python, standard NumPy operations are computationally heavy at leaf nodes compared to pure integer bitwise operations. |
| **Zobrist Hashing** | ✔ Fully implemented | `lookup_board.py` assigns a random 64-bit integer to each line and implements $O(1)$ XOR incremental updates (`current_hash ^ self.line_hashes[move_idx]`). |

## 3. Transposition Table (TT)
| Feature | Status | Comments |
|---------|--------|----------|
| **Hash key generation** | ✔ Fully implemented | XOR incremental updates. |
| **Exact / Lower / Upper Bound entries** | ✔ Fully implemented | Correctly tags TT entries and applies them to alpha/beta bounds during lookup. |
| **Replacement strategy** | ✘ Missing | `lookup_board.py` blindly overwrites collisions with `self.table[h] = data`. There is no two-tier system, aging, or depth-preference. |
| **Depth-aware storage** | ✘ Missing | Although the depth is *stored* in the TT entry, the replacement logic does not protect deep searches from being overwritten by shallow collisions. |
| **Best move storage** | ✔ Fully implemented | Correctly stores `a_best` and attempts to retrieve it during lookup. |
| **TT lookup & storage** | ✔ Fully implemented | Implemented correctly at the start and end of `alpha_beta_search`. |

## 4. Move Ordering
| Feature | Status | Comments |
|---------|--------|----------|
| **Capturing move ordering** | ✔ Fully implemented | Detects moves that create a 3-line box (giving the current player a free box) and pushes them to `greedy_moves`, evaluating them first. |
| **TT best move ordering** | ✔ Fully implemented | Places the TT best move at the very front of the ordered moves list. |
| **Safe move ordering** | ◐ Partially implemented | Implemented *only* at the root level inside `get_move()` when `valid_moves > endgame_threshold`. It checks if a move creates a 3-line box for the opponent. However, this logic is completely ignored inside the `alpha_beta_search` tree! |
| **History heuristic** | ✘ Missing | No tracking of which moves frequently cause cutoffs across different branches. |
| **Killer Move heuristic** | ✘ Missing | No storage of sibling cutoffs. |
| **PV ordering** | ✘ Missing | No PV table or array is maintained to force the Principal Variation to the front. |
| **Chain-aware ordering** | ✘ Missing | Does not order moves based on whether they split chains or give up the smallest chains. |

## 5. Evaluation Function
| Feature | Status | Comments |
|---------|--------|----------|
| **Current score difference** | ✔ Fully implemented | Returns `player_boxes - opponent_boxes` at leaf nodes. |
| **Potential boxes / Chains / Loops** | ✘ Missing | There is zero topological evaluation of the board. The engine is entirely blind to chains, loops, and long chain advantages. |
| **Sacrifice / Double-cross strategy** | ✘ Missing | Cannot evaluate whether sacrificing 2 boxes to secure a 14-box chain is mathematically sound, because it does not compute chains at all. |
| **Endgame evaluation** | ✘ Missing | The evaluation remains static regardless of the phase of the game. |

## 6. Endgame Solver
| Feature | Status | Comments |
|---------|--------|----------|
| **Chain/Loop decomposition** | ✘ Missing | Does not reduce the game state into a graph of independent chains and loops. |
| **Controlled value computation** | ✘ Missing | |
| **Mathematical chain strategy** | ✘ Missing | (e.g., Berlekamp's "Long Chain Rule"). |

## 7. Performance Optimizations
| Feature | Status | Comments |
|---------|--------|----------|
| **Incremental evaluation** | ✘ Missing | At every leaf node, it evaluates `player_boxes = int((s_node.b == player).sum())`. This performs an expensive NumPy array scan of the board. An incremental evaluation counter updated during `execute_move` would be orders of magnitude faster. |
| **Bitboards / Bitmasking** | ✘ Missing | Uses standard NumPy float arrays for the board state (`self.l`, `self.b`). In highly optimized engines, board states are represented by 64-bit integers with bitwise operations. |
| **Symmetry reduction** | ✘ Missing | Dots-and-Boxes has $D_4$ symmetry (8 rotations/reflections). The TT does not check isomorphic board configurations. |
| **Branch reduction (Null-move)**| ✘ Missing | |

## 8. Statistics
| Feature | Status | Comments |
|---------|--------|----------|
| **Nodes, Cutoffs, TT Hit Rate** | ✘ Missing | The `AlphaBetaPlayer` tracks absolutely no analytical statistics during its search. |

---

## Overall Assessment & Engine Strength

**Estimated Strength:** **Beginner to Intermediate**

### Reasoning
Your engine fundamentally transitions from a beginner AI to an intermediate one because it implements the holy trinity of basic classical engines: **Alpha-Beta Pruning**, **Iterative Deepening**, and a **Zobrist Transposition Table**. By avoiding deepcopies, it ensures that search isn't cripplingly slow.

However, the engine hits a hard ceiling and cannot be classified as "Advanced" or "State-of-the-art" for two major reasons:
1. **Blindness to Dots-and-Boxes Theory:** Dots and Boxes is mathematically solved at the tactical level using "Chains" and the "Double-Cross" strategy (Nim-string theory). Because your evaluation function only looks at absolute score (`player_boxes - opponent_boxes`), your engine will naively capture every box available to it, inevitably handing over massive long chains to its opponent. It has absolutely no mechanism to sacrifice 2 boxes to secure the rest of the board.
2. **Speed Bottlenecks:** While you avoid `deepcopy`, heavily relying on `numpy.sum()`, `numpy.where()`, and NumPy array accesses at every node in a Python loop drastically kills node throughput (NPS). A state-of-the-art Python classical engine would either use pure bitwise logic (Bitboards) or compile the heavy loops in C/Cython.

### Highest Priority Improvements
1. **Remove NumPy from Leaf Nodes**: Track `player1_score` and `player2_score` as integers that incrementally update inside `capture_box()`. This will remove `np.sum()` from the leaf nodes and instantly double or triple your search depth.
2. **Implement Chain Evaluation**: The engine *must* learn how to count chains. A basic heuristic that adds bonus points for being the player to close the first box in a "Long Chain" (length $\ge 3$) will drastically change the bot's strength.
3. **Safe Move Ordering inside Tree**: You implemented safe-move ordering at the root, but placing safe moves directly after greedy moves inside the actual Alpha-Beta recursive tree will cause massive cutoffs and speed up the search immensely.
