A good approach is to let an AI coding agent complete one well-defined task at a time. Below is a roadmap from your current heuristic bot to a strong Hybrid MCTS bot.

Phase 1. Refactor the heuristic bot
Task 1

Separate the heuristic from the agent.

Goal

Create

heuristic/
    heuristic.py

class HeuristicPolicy:
    def get_best_move(game)

instead of embedding everything in get_move().

Task 2

Separate board utilities.

Create

board.py

Functions

count_box_sides()

safe_moves()

capture_moves()

chain_detection()

loop_detection()
Task 3

Remove global variables

Replace

self.x
self.y
self.zz
...

with

Move
Chain
Loop

objects.

Phase 2. Fast game clone
Task 4

Implement

clone()

undo()

apply_move()

The clone should be optimized for MCTS and avoid deepcopy().

Task 5

Incremental update

When a move is played

only update

affected boxes
affected edges
score
turn

instead of recomputing the board.

Phase 3. MCTS
Task 6

Implement

Node

containing

parent
children
N
W
Q
P
move
Task 7

Implement Selection

PUCT

Q + c*P*sqrt(parent)/(1+N)
Task 8

Implement Expansion

Expand only

Top heuristic moves

instead of every legal move.

Task 9

Implement Backpropagation

Support

extra turn

score

winner

correctly for Dots and Boxes.

Phase 4. Better rollout
Task 10

Replace random rollout with

HeuristicPolicy

Play until terminal.

Task 11

Implement early termination

If

remaining chains are deterministic

stop rollout

and evaluate directly.

Phase 5. Better move ordering
Task 12

Categorize every legal move into

Capture

Safe

Sacrifice

Chain

Loop

Danger
Task 13

Sort expansion

Capture

↓

Safe

↓

Chain

↓

Others
Phase 6. Progressive widening
Task 14

Initially expand only

Top 6 heuristic moves

Increase expansion as

visit count

grows.

Phase 7. Transposition table
Task 15

Implement

Zobrist Hash

for

edges

scores

player
Task 16

Reuse nodes

when

same position

same player

appears.

Phase 8. Better evaluation
Task 17

Implement heuristic evaluation

score difference

+

safe move count

+

chain parity

+

loop parity

+

estimated future captures
Task 18

Instead of rollout

after

depth > 20

evaluate directly.

Phase 9. Performance
Task 19

Optimize

legal move generation

Maintain

available moves

incrementally.

Task 20

Optimize

capture detection

Only inspect adjacent boxes.

Task 21

Optimize

safe move generation

Cache safe moves.

Phase 10. Stronger search
Task 22

Implement

RAVE

for faster convergence.

Task 23

Implement

Virtual Loss

for parallel MCTS.

Task 24

Implement

Root Dirichlet Noise

for self-play.

Task 25

Implement

Principal Variation

Return

best line

visit count

Q value

for debugging.

Phase 11. Debugging
Task 26

Create a search visualization tool

Display

Tree depth
Visits
Win rate
Selected move
Chain detection
Rollout statistics
Task 27

Benchmark against

Current UCLABot_v3
Greedy bot
Alpha-Beta bot
Random bot

Generate a complete win-rate report.

Phase 12. Final hybrid improvements
Task 28

Use the heuristic as the rollout policy.

Task 29

Use heuristic scores as PUCT priors.

Task 30

Automatically bypass MCTS when the heuristic has a provably optimal action, such as:

Forced capture sequences
Deterministic chain resolution
Endgame double-cross decisions

Run MCTS only for the opening and midgame where strategic planning matters.

Recommended implementation order
Refactor heuristic bot
Fast clone / undo
MCTS framework
Heuristic rollout
Heuristic move ordering
Progressive widening
Zobrist transposition table
Incremental board updates
Static evaluation
RAVE
Parallel MCTS
Visualization and benchmarking

This progression keeps the bot functional after each stage while steadily increasing playing strength.