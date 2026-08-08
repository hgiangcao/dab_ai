# AlphaZero Training System — Complete Instructions

## 1. Objective

Train a Dots and Boxes AlphaZero agent from a supervised pretrained model while preventing regression in absolute playing strength.

The system must optimize both:

1. Improvement against the current AlphaZero checkpoint.
2. Preservation and improvement against fixed external bots.

A model must not be promoted merely because it beats the previous AlphaZero model.

## 2. Model Definitions

### self

`self` is the latest promoted checkpoint used by workers for normal self-play.

Normal self-play:

```text
self + MCTS vs self + MCTS
```

The checkpoint is frozen during game generation.

### best

`best` is the historically strongest checkpoint according to the external-bot benchmark.

It is not necessarily the latest checkpoint.

Example:

```text
current/self = P20
best = P12
```

Workers may play P20 vs P12.

### past

`past` contains older promoted checkpoints. Workers can randomly select historical checkpoints to increase opponent diversity and reduce policy cycling.

## 3. Worker and Trainer Responsibilities

### Workers

Workers generate all training games:

- curriculum-bot games
- self-play games
- games against `best`
- games against historical `past` models

Workers upload replay data to the trainer/server.

### Trainer

The trainer only:

1. Receives replay data.
2. Trains candidate models.
3. Evaluates candidates.
4. Promotes or rejects candidates.
5. Maintains `current/self`.
6. Maintains `best`.
7. Publishes checkpoints.

The trainer must not generate fallback self-play games.

If workers are unavailable, the trainer waits for new replay data.

## 4. Main Flow

```text
Worker
  ↓
Generate games using latest promoted checkpoint
  ↓
Upload replay
  ↓
Trainer receives replay
  ↓
Train candidate
  ↓
Evaluate candidate
  ↓
Promotion decision
  ↓
If promoted:
    candidate becomes current/self
  ↓
Workers download new checkpoint
  ↓
Generate next replay
```

## 5. Self-Play

Standard AlphaZero self-play is:

```text
current/self + MCTS
        vs
current/self + MCTS
```

Do not use `best vs best` as normal self-play.

`best` and `past` are additional opponents used for diversity and stability.

## 6. Opponent Diversity

Recommended steady-state distribution:

```python
FINAL_OPPONENTS = [
    ("self",          0.40),
    ("best",          0.20),
    ("past",          0.15),
    ("ucla_bot_v3",   0.20),
    ("simple_bot_v2", 0.05),
]
```

The exact percentages can be tuned, but self-play should remain the largest component while fixed and historical opponents prevent policy drift.

## 7. Curriculum

Use gradual accumulation instead of abruptly replacing opponents.

```python
PHASES_CONFIG = [
    [("random", 0.8), ("self", 0.2)],

    [("random", 0.4), ("greedy", 0.4), ("self", 0.2)],

    [("random", 0.2), ("greedy", 0.3),
     ("greedy_chain", 0.3), ("self", 0.2)],

    [("greedy", 0.2), ("greedy_chain", 0.3),
     ("simple_bot", 0.3), ("self", 0.2)],

    [("greedy_chain", 0.15), ("simple_bot", 0.25),
     ("simple_bot_v2", 0.35), ("self", 0.25)],

    [("simple_bot", 0.10), ("simple_bot_v2", 0.25),
     ("ucla_bot_v3", 0.40), ("self", 0.25)],

    [("simple_bot_v2", 0.10), ("ucla_bot_v3", 0.45),
     ("self", 0.35), ("best", 0.10)],

    [("simple_bot_v2", 0.05), ("ucla_bot_v3", 0.35),
     ("self", 0.40), ("best", 0.10), ("past", 0.10)],

    [("greedy_chain", 0.05), ("simple_bot_v2", 0.10),
     ("ucla_bot_v3", 0.25), ("self", 0.40),
     ("best", 0.10), ("past", 0.10)],
]
```

All probabilities must sum to 1.0.

Do not abruptly switch from a strong mixed phase such as:

```text
60% UCLA + 40% self
```

to:

```text
100% self
```

because this creates a large training-distribution shift.

## 8. Strong External Reference

`ucla_bot_v3` is the strong external reference.

It should remain in later training and preferably remain at approximately 10–20% in steady-state training.

This provides an external anchor that pure self-play does not provide.

## 9. Replay Buffers

Maintain two separate replay sources.

### RL replay

Contains recent RL/curriculum/self-play experience.

Current configuration:

```text
20,000 games × 8 augmentation
= 160,000 training examples
```

### Pretrained/expert replay

Keep pretrained supervised data separate from the RL replay.

The expert dataset is read-only and used mainly during early RL.

## 10. Pretrained Data During Early RL

The pretrained model already performs strongly against external bots, so early RL should not immediately discard all supervised information.

Initial expert sample budget:

```text
100,000 examples
```

Gradually reduce the expert contribution, for example:

```text
Promotion 0: 100,000
Promotion 1:  90,000
Promotion 2:  80,000
...
Promotion 10: 0
```

Prefer controlling the expert contribution by training-batch ratio.

Suggested approximate schedule:

```text
Early RL:       30–40% expert
Intermediate:   10–20% expert
Later RL:        0–10% expert
Final RL:         0% expert
```

Eventually the expert contribution should reach zero.

## 11. Mixing Expert and RL Data

Train expert and RL examples together.

Do not train:

```text
expert-only → RL-only
```

or:

```text
RL-only → expert-only
```

Instead, mix them within batches.

Example with batch size 64:

```text
48 RL examples
16 expert examples
```

Randomly resample expert examples each training iteration/epoch.

## 12. Why Pure Self-Play Can Regress

Self-play guarantees relative improvement only.

The condition:

```text
candidate > current
```

does not guarantee:

```text
candidate > fixed external bot
```

Possible causes:

- policy drift
- cyclic strategies
- self-play overfitting
- loss of ability to exploit weak-bot mistakes
- changing state distributions
- MCTS/value feedback loops
- noisy promotion decisions

Therefore external evaluation must be part of promotion.

## 13. Candidate Evaluation

For every candidate, perform screening:

```text
10 games × each external bot
50 games vs current checkpoint
```

Promotion requires:

```text
minimum win rate against required bots >= 0.80

AND

win rate vs current >= 0.55
```

This prevents promotion when the candidate severely regresses against one external opponent.

## 14. Best Model

`best` is the model with the highest overall external-bot benchmark score.

It is not automatically the latest promoted model.

Maintain:

```text
current.pt
best.pt
past/
```

A promoted candidate can become `current` while `best` remains an older checkpoint.

## 15. Updating Best

Requested rule:

```text
candidate_overall_score >= 0.90 × best_overall_score
```

Use this as the eligibility condition for updating `best`.

Recommended separation:

```python
promote_candidate = (
    min_bot_winrate >= 0.80
    and winrate_vs_current >= 0.55
)

update_best = (
    candidate_overall_score >= 0.90 * best_overall_score
)
```

For a stricter benchmark, use:

```python
update_best = candidate_overall_score > best_overall_score
```

The 0.90 rule should not replace the minimum-per-bot promotion requirement.

## 16. Best-Model Evaluation

The 10-game screening evaluation is noisy.

When a candidate qualifies to update `best`, perform a larger benchmark, preferably:

```text
200 games per external bot
```

Use the large benchmark to make the final `best` decision.

## 17. Checkpoint Roles

### `current.pt`

Latest promoted model.

Used as `self` by workers.

### `best.pt`

Historically strongest model according to the external benchmark.

Used as a frozen opponent and reference.

### `past/`

Historical promoted checkpoints used for opponent diversity.

## 18. Worker Opponent Selection

Recommended steady-state:

```text
40% self
20% best
15% past
20% UCLA
5% simple_bot_v2
```

Meaning:

```text
self:
    current vs current

best:
    current vs best

past:
    current vs randomly selected historical model

UCLA:
    current vs ucla_bot_v3
```

## 19. Promotion Flow

```text
1. Receive worker replay.

2. Train candidate from current checkpoint.

3. Screen candidate:
       10 × each external bot
       50 × current

4. Check:
       min bot win rate >= 0.80
       AND
       candidate vs current >= 0.55

5. If failed:
       reject candidate
       keep current

6. If passed:
       promote candidate
       candidate becomes current

7. Calculate external benchmark score.

8. If eligible:
       update best.pt

9. Workers download new current checkpoint.

10. Workers generate next replay.
```

## 20. Important: Current and Best Are Independent

Example:

```text
current = P30
best = P12
```

If P31 beats P30 by 56% but has worse external performance than P12:

```text
current = P31
best = P12
```

This is valid.

`current` represents the latest evolutionary state.

`best` represents the strongest known absolute model.

## 21. Loss Behavior

A sudden loss increase after a curriculum change can be caused by a distribution shift.

For example:

```text
60% UCLA + 40% self
```

to:

```text
100% self
```

changes both policy and value targets.

A sudden jump is therefore understandable.

However, if loss remains high while external-bot performance continuously decreases, investigate:

- replay composition
- self-play quality
- opponent diversity
- policy drift
- value-target quality
- promotion frequency

Do not use loss alone as a strength metric.

## 22. Metrics

Track for every promoted model:

```text
Win rate vs Random
Win rate vs Greedy
Win rate vs Greedy Chain
Win rate vs Simple Bot
Win rate vs Simple Bot v2
Win rate vs UCLA
Win rate vs Current
Win rate vs Best
Overall external-bot score
Policy loss
Value loss
Policy entropy
```

External-bot performance is more important than training loss.

## 23. Absolute vs Relative Strength

Relative strength:

```text
Candidate beats Current
```

Absolute strength:

```text
Candidate maintains/improves performance
against fixed external bots.
```

The system should optimize both.

The promotion rule protects against major external regression:

```text
min bot win rate >= 0.80
```

The current-model evaluation measures local AlphaZero improvement:

```text
candidate vs current >= 0.55
```

The benchmark determines the historical `best`.

## 24. Final Architecture

```text
                         WORKERS
                            |
              +-------------+-------------+
              |                           |
          Self-play                   Bot games
              |                           |
              +-------------+-------------+
                            |
                    RL Replay Buffer
                            |
                            |
Expert Replay --------------+
(decaying early RL)         |
                            v
                    Candidate Training
                            |
                            v
                       Evaluation
                            |
          +-----------------+-----------------+
          |                 |                 |
          v                 v                 v
      External bots     Current model      Best model
          |                 |                 |
          +-----------------+-----------------+
                            |
                    Promotion Decision
                            |
                 +----------+----------+
                 |                     |
               Reject                Promote
                 |                     |
            keep current         current = candidate
                                       |
                                       v
                                Best benchmark
                                       |
                                       v
                                    Workers
```

## 25. Core Rules

1. `self` = latest promoted checkpoint.
2. `best` = strongest historical model by external benchmark.
3. `past` = historical promoted checkpoints.
4. Normal self-play = `self vs self`.
5. Do not use `best vs best` as normal self-play.
6. Workers generate all training games.
7. Trainer only trains and evaluates.
8. No trainer fallback self-play.
9. Keep multiple opponent types in later training.
10. Keep a small strong external-bot component in steady-state training.
11. Keep pretrained data separate from RL replay.
12. Mix pretrained and RL examples within batches during early RL.
13. Gradually reduce expert-data influence to zero.
14. Avoid abrupt curriculum-to-pure-self-play transitions.
15. Promote only when both local and external-strength conditions pass.
16. Maintain a separate `best.pt`.
17. Use larger evaluation for reliable best-model decisions.
18. Track absolute benchmark performance continuously.
19. Do not use loss alone as a playing-strength metric.
20. Prefer absolute-strength improvement over merely beating the immediately previous checkpoint.
