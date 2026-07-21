# MCTS Improvement Roadmap for Dots and Boxes

This document describes a roadmap for improving the current classical UCT Monte Carlo Tree Search implementation (`mcts_heuristic.py`).

Each section includes:

- Motivation
- Expected strength improvement
- Difficulty
- Files to modify
- Implementation steps
- Validation tests

---

# Priority 1 — Endgame Alpha-Beta Solver

## Motivation

Classical MCTS performs poorly in deterministic endgames.

When only a small number of moves remain, Alpha-Beta can compute the exact result much faster than random rollouts.

This is the single highest ROI improvement.

---

## Expected Improvement

★★★★★

Typically increases winrate by 15–30%.

---

## Difficulty

Easy

---

## Files

```
mcts_heuristic.py

alpha_beta.py
```

---

## Implementation

### Add threshold

```python
ENDGAME_MOVES = 12
```

During rollout:

```python
remaining = len(game_state.get_valid_moves())

if remaining <= ENDGAME_MOVES:
    return alpha_beta.evaluate(game_state)
```

Instead of performing random rollout.

---

## Required Alpha-Beta API

```python
value = alphabeta.evaluate(game_state)

# returns

[-1,1]
```

or

```python
box_difference / total_boxes
```

---

## Validation

Compare

```
Old rollout

vs

AB rollout
```

Measure

- average rollout error
- search speed
- winrate

---

# Priority 2 — Heavy Rollout Policy

## Motivation

Current rollout is

```
Capture

↓

Safe

↓

Random
```

This ignores

- chain control
- parity
- sacrifice
- edge preference

Most playing strength comes from these.

---

## Expected Improvement

★★★★★

---

## Difficulty

Medium

---

## Files

```
mcts_heuristic.py
```

---

## Implementation

Replace

```python
random.choice(valid)
```

with scoring.

Example

```
score(move)

+100 capture

+30 safe

+10 edge

+5 parity

-50 create chain

-100 open long chain
```

Choose

```
best score

or

softmax(score)
```

instead of random.

---

## Helper Functions

Implement

```
is_capture()

is_safe()

creates_chain()

chain_length()

edge_move()

parity_score()
```

---

## Validation

Compare rollout quality against

Greedy Chain agent.

---

# Priority 3 — Score Difference Backup

## Motivation

Current backup stores

```
win

loss

draw
```

Much information is lost.

Winning

```
20-5
```

is treated exactly the same as

```
13-12
```

---

## Expected Improvement

★★★★☆

---

## Difficulty

Easy

---

## Files

```
mcts_heuristic.py
```

---

## Implementation

Current

```python
if result == rollout_player:

    value = 1
```

Replace with

```python
my_score

opp_score

value = (my_score - opp_score)/total_boxes
```

Result

```
[-1,1]
```

---

## Validation

Observe smoother convergence.

---

# Priority 4 — Progressive Bias

## Motivation

UCT ignores domain knowledge.

Guide search toward promising moves.

---

## Expected Improvement

★★★★☆

---

## Difficulty

Easy

---

## Files

```
mcts_heuristic.py
```

---

## Formula

Current

```
Q

+

UCT
```

Replace

```
Q

+

UCT

+

bias/(visits+1)
```

---

## Bias

Example

```
+100 capture

+40 safe

+20 edge

-80 chain opening

-30 dangerous
```

Bias decreases automatically as

```
visits

↑

bias

↓

```

---

## Validation

Print

```
UCT

Bias

Final score
```

during debugging.

---

# Priority 5 — Better Expansion Ordering

## Motivation

Current ordering

```
Capture

↓

Safe

↓

Remaining
```

Can be improved.

---

## Expected Improvement

★★★☆☆

---

## Difficulty

Easy

---

## Files

```
mcts_heuristic.py
```

---

## New Ordering

```
Immediate capture

↓

Safe edge

↓

Safe center

↓

Small chain

↓

Large chain

↓

Dangerous
```

Randomize only inside each category.

---

## Validation

Observe

Average depth

before

after

---

# Priority 6 — Smart Safe Move Selection

## Motivation

All safe moves are not equally good.

Current

```
random.choice(safe)
```

---

## Expected Improvement

★★★☆☆

---

## Difficulty

Medium

---

## Features

Evaluate

```
future safe moves

future chain count

future parity

future sacrifices
```

Choose best.

---

# Priority 7 — Transposition Table

## Motivation

Many states repeat.

Current tree

```
A

 \

  X

B

 \

  X
```

stores two nodes.

Wasteful.

---

## Expected Improvement

★★★★☆

---

## Difficulty

Medium

---

## Files

```
mcts_heuristic.py

game_state.py
```

---

## Implementation

Implement

```
hash(board)
```

Store

```python
dict

hash

↓

Node
```

Before expansion

```
if hash exists

reuse node
```

---

## Recommended

Use

```
Zobrist hashing
```

---

## Validation

Print

```
Tree nodes

Unique hashes

Reuse rate
```

---

# Priority 8 — RAVE / AMAF

## Motivation

UCT learns slowly.

RAVE shares information between move orders.

---

## Expected Improvement

★★★★☆

---

## Difficulty

Medium

---

## Files

```
mcts_heuristic.py
```

---

## Node Data

Add

```python
rave_visits

rave_value
```

---

## Selection

Replace

```
Q
```

with

```
β × RAVE

+

(1-β) × UCT
```

where

```
β

=

sqrt(K/(3N+K))
```

---

## Validation

Plot

```
Winrate

vs

Simulation count
```

Should improve especially under

```
<1000 simulations
```

---

# Priority 9 — Progressive Widening

## Motivation

Large branching factor wastes simulations.

Instead of expanding everything.

---

## Expected Improvement

★★★☆☆

---

## Difficulty

Medium

---

## Rule

Expand only

```
children

<

C × visits^α
```

Example

```
α = 0.5
```

---

## Validation

Observe

```
Average children

↓

Average depth

↑
```

---

# Priority 10 — Dynamic Exploration Constant

## Motivation

Large exploration early.

Small exploration late.

---

## Expected Improvement

★★★☆☆

---

## Difficulty

Easy

---

## Formula

Instead of

```
C = 1.4
```

Use

```
C

=

2.0

↓

0.6
```

based on

```
remaining moves

or

root visits
```

---

## Validation

Track

```
Visit distribution
```

---

# Priority 11 — Move Pruning

## Motivation

Many moves are obviously bad.

Do not waste simulations.

---

## Expected Improvement

★★★☆☆

---

## Difficulty

Easy

---

## Examples

Prune

```
Open long chain

without capture

Create third edge

without compensation
```

Never prune

```
Capture

Safe

Forced
```

---

# Priority 12 — Parallel MCTS

## Motivation

Current implementation uses one CPU core.

Parallel search scales well.

---

## Expected Improvement

★★★☆☆

---

## Difficulty

Hard

---

## Methods

### Root Parallel

Simplest.

Each thread

```
Own tree

↓

Vote
```

---

### Tree Parallel

Shared tree

Virtual loss

Atomic visits

---

## Validation

Measure

```
Simulation/sec

Winrate
```

---

# Priority 13 — Opening Book

## Motivation

Opening positions repeat frequently.

Avoid searching solved openings.

---

## Expected Improvement

★★☆☆☆

---

## Difficulty

Medium

---

## Implementation

Dictionary

```
Board hash

↓

Best move
```

Loaded before search.

---

# Priority 14 — Hybrid Evaluation

## Motivation

Instead of playing to terminal every rollout.

Stop early.

Evaluate heuristically.

---

## Expected Improvement

★★★☆☆

---

## Difficulty

Medium

---

## Example

After

```
15 rollout moves
```

Evaluate

```
box difference

+

chain score

+

safe move count

+

parity
```

instead of continuing.

---

# Priority 15 — Neural Evaluation

## Motivation

Ultimate replacement for rollouts.

Same philosophy as AlphaZero.

---

## Expected Improvement

★★★★★

---

## Difficulty

Very Hard

---

## Pipeline

```
Selection

↓

Expansion

↓

Neural Network

↓

Value

↓

Backup
```

Policy network also replaces

```
Expansion ordering
```

---

# Suggested Implementation Order

| Step | Feature | Difficulty | Gain |
|-------|----------|------------|------|
| 2 | Heavy Rollout | Medium | ★★★★★ |
| 3 | Score Difference Backup | Easy | ★★★★☆ |
| 4 | Progressive Bias | Easy | ★★★★☆ |
| 5 | Better Expansion Ordering | Easy | ★★★☆☆ |
| 6 | Smart Safe Move Selection | Medium | ★★★☆☆ |
| 11 | Move Pruning | Easy | ★★★☆☆ |