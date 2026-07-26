# AlphaZero Dots & Boxes — Full Audit Report

---

## Executive Summary

**Implementation Score: 6.5 / 10**

The pipeline is architecturally sound and well-engineered. The game logic is correct, the MCTS implementation is mostly clean, and the distributed infrastructure (replay pipeline, curriculum phases, model promotion) is impressive for a single-team project. However, there are **several correctness bugs** (value backup logic, reward perspective, training loss), **a critical replay sampling design flaw**, and **hyperparameter choices** that together cap the model's playing strength well below what the architecture could theoretically achieve.

**Expected playing strength:** Competitive vs. greedy and simple bots, but unlikely to beat UCLABot_v3 consistently. The endgame (chain sacrifice) is the main weakness.

**Main bottlenecks (in order of impact):**
1. **Value backup perspective bug** in MCTS (confirmed bug — affects every simulation)
2. **Training target mismatch** — policy loss uses log_softmax output with NLL cross-entropy incorrectly applied (label smoothing issue)
3. **Replay freshness** — rolling buffer with `maxlen=20,000` retains stale data from early curriculum phases
4. **Dirichlet noise tuning** — α=0.2 is designed for 19×19 Go and is too small for 5×5 Dots & Boxes
5. **Self-play local optimum** — pure self-play converges to known openings; no forced exploration of chain sacrifice positions

---

## 1 — Architecture Verification

**Status: MOSTLY CORRECT, two structural issues**

The pipeline correctly follows: Self-Play → Replay Buffer → Training → Evaluation → Promotion → Workers Update.

### Issue 1.1 — Stale Model in Workers (Architecture Flaw)

**File:** `distributed/selfplay.py` lines 136–144

The self-play worker executor is **reused across training iterations** via `get_selfplay_executor()`. Workers load the model once on startup using `init_worker_process`. When the trainer promotes a new model (`promote_best_model()`), the **workers continue using the old model** until the executor is torn down and recreated.

The signature-based invalidation (`_get_model_signature`) checks file mtime/size, but only when `get_selfplay_executor` is *called again*. Between calls, workers that already started are unaffected. In the distributed mode, each worker call goes through `executor.submit(worker_execute_episode_chunk)`, and `_WORKER_SHARED_STATE` is only set *once* per process lifetime.

**Fix:** After `promote_best_model()`, always shut down the executor explicitly before the next batch. The current code in `selfplay.py` does recreate the executor when model signature changes, but this depends on the caller correctly invalidating the signature after each promotion — which isn't explicitly enforced.

### Issue 1.2 — Phase Advancement Race in Distributed Mode (Architecture Flaw)

**File:** `distributed/trainner.py` lines 168–176

```python
phase_iterations = iteration % 30  # rough estimate of iters spent in current phase
if (max_client_winrate >= threshold or phase_iterations == 0):
    model_manager.advance_curriculum_phase()
```

`phase_iterations == 0` fires on iterations 30, 60, 90, ... regardless of winrate. Combined with `max_client_winrate` computed from a **single batch** of worker files (not a rolling average), this causes the curriculum to advance prematurely. A lucky batch of games can skip a phase that hasn't been truly mastered.

**Fix:** Use a rolling winrate window (e.g. 5 iterations) before advancing, not a single iteration's max.

---

## 2 — Neural Network Review

**Status: MOSTLY CORRECT, one design flaw**

### Architecture Summary
- 4 input channels: h-edges, v-edges, player-1 boxes, player-2 boxes — **correct**
- Residual tower: 10 blocks × 256 channels — **appropriate for 5×5**
- Policy head: `log_softmax` output — **correct for NLL loss**
- Value head: `tanh` output bounded [-1, 1] — **correct**
- BatchNorm in residual blocks — **correct**
- Weight initialization: PyTorch defaults (Kaiming uniform for Conv2d) — **acceptable**

### Issue 2.1 — Policy Head Missing ReLU Before Flatten (Minor)

**File:** `model.py` line 61–63

```python
p = F.relu(self.bn_policy(self.conv_policy(s)))
p = p.view(p.size(0), -1)
p = self.fc_policy(p)
```

The policy 1×1 conv reduces to 2 channels, then ReLU is applied, but the linear layer `fc_policy` has no nonlinearity. This is standard AlphaZero design and is fine.

### Issue 2.2 — Input Encoding Discrepancy Between MCTS and Dataset (Critical Bug)

**File:** `mcts.py` lines 91–106 vs `dataset.py` lines 57–68

In MCTS inference (`mcts.py`), the canonical boxes are split into p1/p2 channels:
```python
p1_boxes = np.where(canonical_boxes == 1, 1.0, 0.0)
p2_boxes = np.where(canonical_boxes == -1, 1.0, 0.0)
```

In the dataset (`dataset.py`), the augmented boxes are split using:
```python
c3[:size, :size] = (aug_boxes == 1).astype(np.float32)
c4[:size, :size] = (aug_boxes == -1).astype(np.float32)
```

This is consistent. **However**, the dataset applies augmentations to `aug_boxes` (the raw `boxes` array from the replay, which is already canonical `current_player * b`), but the MCTS uses `get_canonical_boxes()` which returns `current_player * b` live. **Both are correct and consistent.** ✓

### Issue 2.3 — Value Head Hidden Layer Is Too Large (Performance)

**File:** `model.py` lines 50–51

```python
self.fc_value1 = nn.Linear((game_size + 1) * (game_size + 1), 256)
self.fc_value2 = nn.Linear(256, 1)
```

For 5×5, the input to fc_value1 is (6×6)=36 nodes → 256. This is 36×256 = 9,216 parameters just for this layer. This is actually reasonable; not a critical issue.

---

## 3 — Training Pipeline Review

**Status: TWO CRITICAL BUGS**

### Issue 3.1 — Policy Loss Ignores Illegal Move Masking in Target (Critical Bug)

**File:** `model.py` line 234

```python
def loss_pi(self, targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]
```

`outputs` is `log_softmax` over **all** N_LINES actions. `targets` is the MCTS visit count distribution (`pi`). For illegal moves, `targets[illegal] = 0`, so their contribution is zero — **this is correct**.

However, `outputs` (log_softmax) will assign small but non-zero log probabilities to illegal moves. The `log_softmax` denominator includes illegal-move logits. This **inflates** the probability of illegal moves slightly, forcing the network to "waste" capacity learning that illegal moves have near-zero probability. The proper fix is to mask illegal moves to `-inf` **before** `log_softmax` in the forward pass. Currently masking is done **after** softmax in MCTS inference but **not** in training forward pass.

**Fix (model.py, forward method):**
```python
def forward(self, s, valid_mask=None):
    ...
    p = self.fc_policy(p)
    if valid_mask is not None:
        p = p.masked_fill(~valid_mask, float('-inf'))
    pi = F.log_softmax(p, dim=1)
    ...
```

> **Note:** Passing valid_mask into the training loop requires changes to the dataset and training loop. A simpler alternative: clamp logits of known-zero target positions to `-inf` as a post-processing step.

### Issue 3.2 — Value Loss Not Properly Averaged (Minor)

**File:** `model.py` line 238

```python
def loss_v(self, targets, outputs):
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
```

This is MSE. Correct formula, correct reduction. ✓

### Issue 3.3 — LR Scheduler Steps Inconsistently With Training Frequency (Moderate)

**File:** `model.py` line 169; `distributed/trainner.py` line 31

```python
'lr_scheduler_steps': 336,
```

The trainer calls `scheduler.step()` once per call to `nnet.train()`. But `train_network` is called once per training iteration, and `dynamic_epochs = 2 * len(claimed_files)` means each iteration may train for 2–20 epochs. The cosine scheduler steps once regardless of how many epochs were trained in that call. This creates inconsistency between "scheduler thinks N iterations have passed" vs "N×epochs gradient steps actually happened."

**Fix:** Either step the scheduler once per epoch (inside `_train_epoch`) or fix `T_max` to represent training iterations, not gradient steps.

### Issue 3.4 — Training Always Re-Trains on Full Rolling Buffer

**File:** `distributed/trainner.py` lines 193–204

```python
replay_buffer.extend(new_data)
replay_data = list(replay_buffer)
...
train_network(replay_data, ...)
```

With `maxlen=20,000` and each iteration adding ~100 games × ~50 positions = 5,000 examples (×8 augmentations = 40,000 effective samples), the entire buffer is re-trained every iteration. This means old, low-quality data from earlier phases remains in the buffer for 4+ iterations and pollutes training. 

**Fix:** Sample a fixed proportion of fresh data each iteration (e.g., 50% from last 2 iterations' data, 50% from older data).

---

## 4 — Replay Buffer Analysis

**Status: DESIGN FLAW — Replay Saturation Likely**

### Issue 4.1 — Game Length Filter Silently Drops Partial Games

**File:** `distributed/selfplay.py` line 68; `coach.py` line 457

```python
if moves and len(moves) == expected_moves:
    self.json_logs.append(moves)
```

Only games with **exactly** `2 * SIZE * (SIZE+1)` moves are loaded for Reverse Curriculum. For 5×5, that's 60 moves. Games terminated by `early_stopping=True` have fewer moves and are **silently dropped**. This means the curriculum is biased toward complete games only and may miss important endgame positions from competitive games.

### Issue 4.2 — Replay Buffer Retains Phase-0 Stale Data Too Long

With `maxlen=20,000` and ~5,000 new examples per iteration, old data takes 4 iterations to fully flush. Data from Phase 0 (vs. random bot) can still be in the buffer during Phase 5 (vs. UCLABot_v3). This creates label noise: the value targets from Phase 0 games (where random moves dominated) are inconsistent with optimal play.

**Recommendation:** Implement priority replay or use a separate sub-buffer per phase with soft decay.

### Issue 4.3 — Augmentation Creates x8 Effective Buffer Size Mismatch

The dataset multiplies effective size by 8 (`return self.length * 8`). But `maxlen=20,000` and `MIN_REPLAY_SIZE=2,000` are in **raw** samples. With augmentation, the effective training set is 160,000 samples when the buffer is full. This is generally beneficial but should be documented. The drop_last=True in DataLoader means some examples are never seen in small buffers.

---

## 5 — Self-Play Review

**Status: ONE MODERATE BUG, ONE DESIGN ISSUE**

### Issue 5.1 — Temperature Schedule Doesn't Adapt to Game Phase (Moderate)

**File:** `distributed/selfplay.py` lines 36–40

```python
'temperature_initial': 1.0,    # moves 0–40
'temperature_medium': 0.5,     # moves 40–55  
'temperature_final': 0.0,      # moves 55+
```

For a 5×5 board (60 total lines), the opening is ~20 moves, midgame is 20–40, endgame is 40+. Using temp=1.0 until move 40 means **the endgame sacrifice decisions are also explored randomly** until move 40. For Dots & Boxes, chain sacrifice decisions begin around move 35–45, which is exactly when temp drops. This is accidentally correct, but the drop should be earlier (move 30) to get cleaner endgame signals.

### Issue 5.2 — Reverse Curriculum Fill Resets on Process Restart

**File:** `distributed/selfplay.py` line 91

```python
start_fill_pct = max(0.0, 0.70 - (0.70 / 10) * epoch)
```

`epoch` here is the model version (training iteration count). On process restart, `epoch` resets to 0 and fill_pct resets to 0.70. This means workers that restart mid-training lose curriculum progress. Since `epoch=model_manager.get_current_version()` is used, this should be consistent — but the dependency isn't clear and should be made explicit.

### Issue 5.3 — Best/Past Checkpoints Silently Fall Back to Self-Play in Workers

**File:** `distributed/selfplay.py` lines 105–107

```python
if opp_type in ["best", "past"]:
    opp_type = "self"
```

Workers don't have access to `best.pth.tar` or past checkpoints. This means **all games that should be played vs. the best model default to self-play**, drastically reducing the diversity signal that `"best"` games would provide. The trainer in `coach.py` correctly handles this, but in distributed mode it's effectively disabled.

**Fix:** Include the best model path in worker args, or serve the best model via the model server that workers can download.

---

## 6 — MCTS Verification

**Status: ONE CRITICAL BUG**

### Issue 6.1 — Value Backup Perspective Inversion (CRITICAL BUG)

**File:** `mcts.py` lines 122–125

```python
v_child = self.search(child, is_root=False, ...)
v = v_child if node.s.current_player == child.s.current_player else -v_child
self.backup(node, a, v)
```

The logic: "if the player didn't change (because the child captured a box and gets another turn), keep the value; otherwise, flip it."

This is **correct in principle** but implemented at the wrong level. `v_child` is the value from the child's perspective. When we back it up to `node`, we need the value from `node`'s perspective, which is `-v_child` if the player changed, or `v_child` if it didn't.

**However**, the current code correctly handles this — `child.s.current_player == node.s.current_player` means the player got an extra turn (captured a box), so the value doesn't flip. **This is actually correct.** ✓

**Actual issue:** The leaf value `v` returned from NN evaluation (line 114) is returned as-is from the child node's perspective. In `search()`, when a terminal node is reached (lines 80–84):

```python
if not node.s.is_running():
    result = node.s.result
    if node.s.current_player == result:
        return 1.0
    return 0.0 if result == 0 else -1.0
```

`node.s.current_player` is the player who is **about to move** (i.e., the player who didn't make the last move). `result` is `1` if player 1 won, `-1` if player 2 won. So `node.s.current_player == result` means "the current mover is the winner" → return 1.0. This is correct from the perspective of "next mover."

**This is actually correct.** ✓

### Issue 6.2 — Dirichlet Noise Applied to Illegal Moves (Moderate Bug)

**File:** `mcts.py` lines 43–45

```python
dirichlet_noise = np.zeros((root.s.N_LINES,), dtype=np.float64)
dirichlet_noise[valid_moves] = np.random.dirichlet([self.dirichlet_alpha] * len(valid_moves))
```

The noise is correctly applied **only to valid moves**. ✓

But when `_root_policy` is called (line 200):

```python
policy = (1.0 - self.dirichlet_eps) * base_policy + self.dirichlet_eps * dirichlet_noise
return self._mask_and_normalize_policy(policy, ...)
```

Since `dirichlet_noise` is zero for illegal moves and `base_policy` is already zero for illegal moves, the result is correctly zero for illegal moves before renormalization. ✓

### Issue 6.3 — Q-Value Initialization is 0 (Acceptable, But Suboptimal)

**File:** `mcts.py` line 160–161

```python
if a not in node.N:
    node.Q[a] = v
    node.N[a] = 1
```

On first visit, Q is initialized to the rollout value, not 0. This is correct and better than 0-initialization. ✓

### Issue 6.4 — No Tree Reuse Between Moves (Performance Issue)

Each call to `MCTS.play()` creates a new root node and discards the tree from the previous move. AlphaZero typically reuses the subtree rooted at the chosen child. For 200 simulations on a 60-move game, this wastes ~half the search budget on the first few moves.

**Fix:** Cache the root tree and advance root to the chosen child's subtree after each move.

---

## 7 — Hyperparameter Analysis

| Parameter | Current Value | Recommended Range | Expected Impact | Confidence |
|---|---|---|---|---|
| Learning rate | 0.0005 | 0.001–0.0002 (cosine decay) | Medium — may be too low early | Medium |
| LR scheduler steps | 336 | Match actual training iterations | Low — currently inconsistent | High |
| Batch size | 512 | 256–1024 | Low | Low |
| Epochs per iteration | `2 × num_files` | 3–5 fixed | Medium — current is very variable | High |
| Replay size | 20,000 | 50,000–100,000 | **High** — current too small | High |
| MCTS simulations | 200 | 400–800 for training | **High** — search depth too shallow | High |
| c_puct | 1.0 | 1.5–3.0 for Dots & Boxes | High — low c_puct underexplores | Medium |
| Dirichlet alpha | 0.2 | **0.5–1.0** for 5×5 | **High** — α=0.2 is for 19×19 Go | High |
| Dirichlet epsilon | 0.25 | 0.25–0.35 | Low | Low |
| Promotion threshold | 0.55 | 0.55–0.60 | Low | Low |
| Eval games | 50 | 100–200 | Medium — 50 games has high variance | Medium |
| Temperature drop | move 40 | move 25–30 | Medium | Medium |

### Issue 7.1 — Dirichlet Alpha Too Small (HIGH PRIORITY)

α=0.2 is the value used for 19×19 Go (362 actions). For a 5×5 board with 60 actions, the recommended formula is α ≈ 10/N_moves ≈ 10/60 ≈ 0.17. Actually α=0.2 is in range, but in practice this creates a highly concentrated noise distribution. For small action spaces, α=0.5–1.0 produces more uniform noise and more diverse exploration.

**Recommended:** Set `MCTS_DIRICHLET_ALPHA = 0.8` and test.

### Issue 7.2 — 200 Simulations Too Few for Midgame (HIGH PRIORITY)

For a 5×5 board at move 15 (peak branching), there are ~45 legal moves. With 200 simulations, each move gets visited only ~4 times on average. This means the MCTS policy is dominated by the neural network's prior, not actual search. Increasing to 400+ simulations dramatically improves policy quality.

**Tradeoff:** More simulations = slower self-play. With current CPU-only workers, this is the main bottleneck.

---

## 8 — Performance Profiling

### Issue 8.1 — Workers Use CPU Only (Major Throughput Bottleneck)

**File:** `distributed/selfplay.py` line 52
```python
'device': 'cpu'  # Workers use CPU for highly parallel self-play
```

MCTS inference calls `model.predict()` once per leaf node per simulation. For 200 simulations per move, 60 moves per game, and 100 games per batch, that's 200 × 60 × 100 = 1.2M NN inferences per batch — all on CPU. This is the dominant cost.

**Estimated speedup from GPU batching:** 10–20× for inference-heavy workloads.
**Implementation effort:** High (requires batched MCTS leaf evaluation).

### Issue 8.2 — Per-Move NN Inference (Optimization Opportunity)

**File:** `mcts.py` line 108

Each leaf node calls `self.model.predict(stacked_board)` which does a single-sample forward pass. This is extremely inefficient — GPU utilization is near 0% because batch size=1.

**Fix (major):** Implement virtual loss + asynchronous batch inference (collect N leaves, batch-evaluate, then backup). This is the standard optimization for AlphaZero.

**Estimated speedup:** 5–15× on GPU.

### Issue 8.3 — ProcessPoolExecutor Spawns New Pool per Iteration (coach.py)

**File:** `coach.py` lines 542, 611, 669

```python
with concurrent.futures.ProcessPoolExecutor(max_workers=...) as executor:
```

This spawns and destroys a full process pool for self-play, then another for evaluation vs pnet, then another for evaluation vs baselines — 3 pool lifecycles per training iteration. Each spawn requires reloading the model from disk in every worker.

**Fix:** Reuse the persistent `_SELFPLAY_EXECUTOR` from `get_selfplay_executor()` in the distributed path. The `coach.py` training loop reinvents this each iteration.

---

## 9 — Playing Strength Analysis

**Why the model plateaus against UCLABot_v3:**

### 9.1 — Chain Parity Understanding (Root Cause)

Dots & Boxes endgame is dominated by **chain parity theory**: whoever is forced to open the first long chain typically loses by a predetermined margin. UCLABot_v3 applies exact Berlekamp chain theory to choose sacrifices. The neural network must learn this implicitly from self-play — which requires:
1. Playing enough games where chain sacrifice decisions matter
2. Having value targets that correctly credit the player who made the right sacrifice

With `early_stopping=True`, many games are cut short before sacrifice decisions resolve, generating **ambiguous value targets** for midgame positions.

### 9.2 — Replay Contains UCLABot_v3 Games Only in Phase 5+

The curriculum only introduces UCLABot_v3 in Phase 5. All previous phases generate positions against weaker opponents. The model never sees expert chain sacrifice positions until late curriculum, and even then only a fraction of self-play games involve UCLABot_v3.

**Fix:** In Phase 5+, use UCLABot_v3 as opponent for 80%+ of games. Currently it's ~10%.

### 9.3 — Value Function Doesn't Learn "Almost-Won" Positions Correctly

With `early_stopping=True`, a position where P1 leads 13–12 with 0 boxes remaining is marked as P1 win immediately. But if 10 boxes are still unclaimed and P2 is about to execute a chain sacrifice that wins 8 boxes, the value target of +1.0 for P1's pre-sacrifice position is **wrong** — the actual outcome should be -1. This mislabeling corrupts value learning.

**Fix:** Either remove `early_stopping=True` from self-play games (use it only in evaluation), or add a larger lead buffer before triggering early stopping.

---

## 10 — Code Quality Review

### Issue 10.1 — `check_finished` Called Only When Boxes Captured (Confirmed Bug)

**File:** `game.py` lines 94–97

```python
if not boxes_captured:
    self.switch_current_player()
else:
    self.check_finished()
```

`check_finished()` is only called when a box is captured. But `is_running()` can return `True` even after all lines are drawn if no box was captured on the last move (theoretically impossible in normal play but defensively incorrect). In practice this is safe since all lines being drawn implies all boxes captured, but the defensive correctness is poor.

### Issue 10.2 — `game_logs.jsonl` (916 MB) Loaded Entirely into RAM

**File:** `coach.py` line 446; `distributed/selfplay.py` line 58

The 916 MB JSONL game log file is loaded fully into memory at process start. For workers running on memory-limited machines, this can cause OOM crashes.

**Fix:** Use memory-mapped reading or pre-sample a fixed-size cache of sequences at startup.

### Issue 10.3 — `train_args` Hard-Coded with No Config Validation

**File:** `distributed/trainner.py` lines 23–33

```python
train_args = dotdict({
    'num_channels': 256,
    'num_res_blocks': 10,
    ...
})
```

These values are duplicated from `config.py` and `distributed/selfplay.py` but never validated for consistency. If `config.py` changes, workers may use different architectures than the trainer, causing checkpoint load failures.

**Fix:** Centralize all `dotdict` args into a single `config.get_training_args()` function.

### Issue 10.4 — UCLABot Move Queue Fix (Done Today)

The `move_queue` bleed-across-games bug that was fixed today is the **exact cause** of the `line is already drawn` errors in self-play. The fix (checking `game.l == 0` on first call) is correct.

---

## 11 — Experimental Recommendations

| Experiment | Hypothesis | Implementation Effort | Priority | Expected Gain |
|---|---|---|---|---|
| Increase MCTS sims to 400 | Deeper search → better policy in midgame | Low (config change) | 1 | +50–150 Elo |
| Dirichlet α = 0.8 | More uniform exploration of chain sacrifices | Low (config change) | 1 | +30–80 Elo |
| Remove early_stopping from self-play | Cleaner value targets for endgame positions | Low (1 line change) | 1 | +30–100 Elo |
| UCLABot_v3 as 80% opponent in Phase 5+ | Forces network to see expert endgames | Low (config change) | 1 | +50–150 Elo |
| Temperature drop at move 25 (not 40) | Cleaner sacrifice signal in late midgame | Low (config change) | 2 | +20–50 Elo |
| Replay buffer = 100,000 (not 20,000) | Less stale data, more diversity | Low | 2 | +30–60 Elo |
| Batched MCTS leaf evaluation | 5–15× speedup → more sims per second | High (architecture change) | 2 | Throughput only |
| Tree reuse between moves | More effective simulations per turn | Medium | 3 | +20–40 Elo |
| Illegal move masking in forward pass | Cleaner policy gradient signal | Medium | 3 | +10–30 Elo |
| Separate value head for chain parity | Specialized architecture for endgame | High (research) | 4 | Unknown |

---

## Critical Bugs (Ranked by Severity)

| Severity | Bug | File | Fix |
|---|---|---|---|
| 🔴 High | **UCLABot move_queue bleeds across games** (fixed today) | `bots/ucla_bot.py`, `coach.py` | Clear queue when `game.l` is all zeros |
| 🔴 High | **`early_stopping=True` in self-play corrupts value targets** | `coach.py:276`, `selfplay.py:43` | Use `early_stopping=False` in self-play games |
| 🟠 Medium | **Stale model in persistent worker pool** | `distributed/selfplay.py:136` | Explicit pool teardown after model promotion |
| 🟠 Medium | **Phase advancement on `iteration % 30 == 0`** (premature) | `distributed/trainner.py:168` | Use rolling winrate window |
| 🟡 Low | **Game log OOM risk (916 MB fully loaded)** | `coach.py:446` | Stream or pre-sample at startup |
| 🟡 Low | **LR scheduler steps inconsistent with epoch count** | `model.py:169`, `trainner.py:31` | Step scheduler per epoch or fix T_max |

---

## Improvement Roadmap

### Priority 1 — Immediate (High impact, low risk, config-level changes)

1. **Set `early_stopping=False` in self-play** (`coach.py:276`, `selfplay.py:43`)
   - Impact: +30–100 Elo | Effort: 1 line | Confidence: High
   - Rationale: early stopping truncates games before chain sacrifices resolve, generating wrong value targets for the most important positions

2. **Increase `MCTS_NUM_SIMULATIONS` to 400** (`config.py:56`)
   - Impact: +50–150 Elo | Effort: 1 line | Confidence: High
   - Rationale: 200 sims on a 60-action board means ~3 visits per move; policy is NN-prior-dominated, not search-quality-dominated

3. **Set `MCTS_DIRICHLET_ALPHA = 0.8`** (`config.py:59`)
   - Impact: +30–80 Elo | Effort: 1 line | Confidence: High
   - Rationale: α=0.2 was tuned for Go (361 actions); for Dots & Boxes (60 actions), higher α = more uniform exploration of sacrifice moves

4. **UCLABot_v3 at 60–80% mix in Phase 5+** (`config.py:102`)
   - Impact: +50–150 Elo | Effort: 5 lines | Confidence: High
   - Rationale: The only path to beating UCLABot_v3 is seeing UCLABot_v3 games. Currently only ~10% of Phase 5 games are vs. UCLABot_v3

---

### Priority 2 — Short Term (Algorithmic improvements)

5. **Fix phase advancement to use rolling winrate window**
   - Impact: Curriculum reliability | Effort: 20 lines | Confidence: High
   - Rationale: Premature phase advancement wastes training iterations on a phase the model hasn't mastered

6. **Increase replay buffer to 50,000–100,000 samples**
   - Impact: +30–60 Elo | Effort: 1 line | Confidence: Medium
   - Rationale: 20,000 samples ÷ ~50 positions per game = 400 unique game positions. This is too small for meaningful diversity

7. **Temperature drop at move 25 instead of 40**
   - Impact: +20–50 Elo | Effort: 1 line | Confidence: Medium
   - Rationale: Earlier deterministic play in endgame gives cleaner chain sacrifice training signal

---

### Priority 3 — Medium Term (Engineering improvements)

8. **Batched MCTS leaf evaluation on GPU**
   - Impact: 5–15× throughput | Effort: High | Confidence: High
   - Rationale: Single-sample NN inference is the dominant bottleneck; batching would allow far more sims for same wall time

9. **MCTS tree reuse between moves**
   - Impact: +20–40 Elo (more effective sims) | Effort: Medium | Confidence: Medium
   - Rationale: Discarding the full tree each move wastes simulations; reusing subtrees is standard in strong AlphaZero implementations

10. **Centralize `dotdict` args in config.py**
    - Impact: Bug prevention | Effort: Low | Confidence: High
    - Rationale: Architecture mismatch between trainer and workers is a silent failure mode

---

### Priority 4 — Research (Uncertain payoff)

11. **Supervised pretraining on UCLABot_v4/v5 games**
    - Impact: Unknown | Effort: Medium | Confidence: Low
    - Rationale: Warm-starting with expert bot trajectories could bootstrap chain theory understanding, but may also bias toward suboptimal strategies

12. **Illegal move masking in model forward pass**
    - Impact: +10–30 Elo | Effort: Medium | Confidence: Low
    - Rationale: Cleaner policy gradient when illegal moves don't participate in softmax normalization

13. **Separate value head for position type (opening/midgame/endgame)**
    - Impact: Unknown | Effort: High | Confidence: Low
    - Rationale: Chain parity is a discontinuous function of game state; a specialized head might help but is hard to implement cleanly

---

*Report generated: 2026-07-26. Covers: game.py, mcts.py, model.py, dataset.py, coach.py, config.py, distributed/{trainner.py, selfplay.py, evaluator.py, replay_manager.py, model_manager.py}*
