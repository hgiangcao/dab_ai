alpha_beta.py

  1. Transposition table key omits box ownership and whose turn it is (correctness bug).
  The Zobrist hash is a function of the line vector only — compute_initial_hash(s_node.l) (line 72) and update_hash(current_hash, a) (line 243) take lines/move indices, never s.b or
  current_player. In Dots and Boxes the owner of a completed box is whoever drew its 4th line, which depends on move order, so two nodes in the same search can share an identical set of
  drawn lines yet differ in score and in whose turn it is.

  Concrete example: box with lines {1,2,3,4} plus an unrelated line 5.
  - Path A: P draws 1→Q, Q draws 2→P, P draws 3→Q, Q draws 4 (completes box, Q owns, Q continues), Q draws 5→P. End: box owned by Q, turn = P.
  - Path B: P draws 1→Q, Q draws 2→P, P draws 3→Q, Q draws 5→P, P draws 4 (completes box, P owns). End: box owned by P, turn = P.

  Both reach the identical line vector l = {1,2,3,4,5}, so they collide to the same TT entry (lines 161–173) despite opposite box ownership and different true values. The cached
  value/move from one is returned for the other. The table also persists across get_move calls (cleared only above 1,000,000 entries, line 69), so the collision is not confined to one
  search.

  2. Move-ordering heuristic scores the child with the parent's maximize flag (strength bug).
  In heuristic_score (lines 218–224), execute_move(m) may switch current_player, but evaluate(s_node, maximize) is called with the node's original maximize. Inside evaluate, player =
  s.current_player if maximize else -s.current_player (line 99), so after a player-switching move player becomes the opponent and the score is computed in the opponent's frame. Every move
  in safe_moves/sacrifice_moves is non-capturing (captures are pulled out into greedy_moves at lines 193–194), so at a maximize node all of them switch the player and are ranked by the
  opponent's benefit, then sorted reverse=True — worst-first. This inverts the ordering, weakens pruning, and reduces the depth reached under the time limit. It does not change the value
  of a fully completed depth (alpha-beta is order-independent for that), so the effect is shallower search and weaker play, not a wrong value at a completed depth.

  3. evaluate heuristic terms are turn-independent (heuristic weakness).
  unsafe (boxes with 3 filled lines), chains, and mobility (lines 107–125) are applied identically regardless of whose turn it is, even though a 3-filled box benefits the side to move and
  hurts the other. Not a crash; it just makes the static evaluation noisier.

  Minor: the cutoffs use strict >/< (lines 264, 271) rather than >=/<=, which prunes slightly less but stays correct.

  mcts_heuristic.py

  1. The double-cross branch is dead code (logic bug).
  _get_double_cross_move (lines 289–294) calls _find_chains, which only records boxes with exactly one free line, i.e. 3 lines already drawn (line 252). But get_move returns a greedy
  capture first whenever any box has 3 drawn lines (lines 313–316, _get_greedy_move). So by the time _get_double_cross_move runs, no box has 3 drawn lines, _find_chains always returns [],
  and the if len(chain) >= 3: return chain[0][1] branch never executes. The intended endgame chain/double-cross play is never reached through this path; only the "no safe moves"
  sacrifice-minimization branch (lines 296–308) ever does anything.

  2. Endgame delegation import path is inconsistent with the module layout (likely ImportError).
  Line 326 does from bots.alpha_beta import AlphaBetaPlayer, while the same file imports from agent_interface import BaseAgent at top level (flat layout, matching the directory). If the
  layout is flat, the ≤12-move branch raises ModuleNotFoundError the first time it triggers. Separately, the threshold mismatches: AlphaBetaPlayer defaults to endgame_threshold=8, so for
  9–12 remaining moves the delegated bot only does greedy/random (no deep search) — the same thing MCTS could do — so the <= 12 gate does not buy the intended alpha-beta search until ≤8
  moves.

  3. Trivial: import numpy as np (line 12) is unused.

  What I verified as correct

  - MCTS perspective handling for extra turns: value_sum is stored in each node's own player_to_move frame, backprop flips on parent.player_to_move != curr.player_to_move (lines 89–91),
  and _select_child negates q on the same condition (lines 115–116). Consistent, including capture-continues-turn.
  - Move/undo bookkeeping in MCTS: moves_made counts selection + expansion + rollout moves and undoes exactly that many (lines 96–97); each simulation is balanced, so game_state is
  restored before the post-search safety check.
  - AB maximize/depth update for the extra-turn rule (lines 249, 252) and the terminal vs. heuristic base case (lines 146–154) keep values consistently in the root player's frame.

  The two findings most likely to change results in play are the TT key collision (alpha_beta.py) and the dead double-cross branch (mcts_heuristic.py); the delegation import will
  hard-fail if the package is flat. 