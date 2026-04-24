# Implementation Plan (2026-04-24)

Status: Item 3 ✓ COMPLETE. Item 2 ✓ COMPLETE. Item 1 IN PROGRESS.

---

## Item 1 — CFR+ Convergence Stall

**Symptom:** `cfr_plus_mb4_hh` (50k iters, max_bids=4, HH) finished at exploitability 0.4712. Dropped 0.52→0.47 in first 2000 iters, then completely flatlined. Target was ≤0.1.

**What we know:**
- Linear averaging IS correctly implemented in both fast and slow solvers (`strategy_sum += lin_w * reach * strategy`)
- At max_bids=2, HH, 500 iters: fast=0.508, slow=0.699 — fast converges faster, not incorrectly. Fast solver is not obviously broken.

**Hypotheses:**

**A — 0.47 is the true Nash exploitability for this restricted game.**
The restricted bid space (26 bids: HC+Pair only, max_bids=4 cap, HH) may prevent reaching the true Nash of the full 110-bid game. CFR+ converges to Nash of the game it's given — if that game's Nash has non-zero exploitability vs the full game, exploitability plateaus at a non-zero floor.

**B — Numerical precision at large iteration counts.**
After 50k iterations, linear weights cause `strategy_sum` magnitudes to be very large. Float64 precision may cause the average strategy to stop updating meaningfully.

**Investigation steps (in order):**

1. **Read b5j72nxg6 result** (no-HH, 5k iters, fast vs slow):
   ```bash
   # Check if process is done, then read output
   ```
   If both converge to the same plateau → Hypothesis A.

2. **Run slow solver on the production config** (max_bids=4, HH, 5k iters):
   ```python
   from agent.baseline.cfr_1v1 import CFRSolver
   s = CFRSolver(max_bids=4, high_hand=True)
   for t in range(5000):
       s.iterate()
       if t % 500 == 499:
           print(t+1, s.exploitability())
   ```
   If slow solver also stalls at ~0.47 → Hypothesis A confirmed. Move on.

3. **If fast diverges from slow at 5k iters:** Compare `exploitability()` implementation between `cfr_1v1_fast.py` and `cfr_1v1.py`. Most likely bug: average strategy normalization before best-response computation.

4. **If Hypothesis A confirmed:** Accept 0.47 exploitability in the restricted game. Benchmark the agent anyway — restricted-game Nash may still beat random. Alternatively, remove max_bids cap and run on the full 26-bid game.

---

## Item 2 — Fix `eval.py` to Pass `exact_rules`

**Symptom:** All R-NaD Stage A eval metrics (vs_random 0.47–0.57) are invalid. The model trained on exact-rules was evaluated under at-least rules because `new_match` was never passed `exact_rules=True`.

**Exact locations:**

| File | Line | Current | Fix |
|---|---|---|---|
| `agent/rnad/eval.py` | ~61 | `state = new_match(num_players)` | `state = new_match(num_players, exact_rules=exact_rules, high_hand=high_hand)` |
| `agent/rnad/eval.py` | ~113 | `state = new_match(num_players)` | same |
| `agent/rnad/eval.py` | ~305 | `state = new_match(num_players)` | same |

**Signature changes:**

```python
# play_round (~line 46)
def play_round(agents, hand_size=5, exact_rules=False, high_hand=True):

# play_match (~line 100)
def play_match(agents, num_rounds=..., exact_rules=False, high_hand=True):

# evaluate_policy (~line 219)
def evaluate_policy(policy_fn, ..., exact_rules=False, high_hand=True):
```

**Wire through from trainer** in `agent/rnad/trainer.py`, `_log_eval`:
```python
evaluate_policy(..., exact_rules=self.config.exact_rules, high_hand=self.config.high_hand)
```

**Validate fix:**
```bash
python3 -m agent.rnad.eval \
    --checkpoint agent/checkpoints/rnad_final.pt \
    --exact-rules --episodes 1000
```
Expected: vs_random meaningfully above 0.5 (Stage A validation showed 0.546 trend during training).

---

## Item 3 — Check MC Completion

**Command:**
```bash
python3 -m agent.data.compute_extended_conditional_exact_probs --dry-run
```

**If all 70 conditions complete:**
- Cache at `agent/data/extended_conditional_exact_probs.json` is ready
- Re-run benchmark: `python3 benchmark.py --exact`
- Confirm `ExactRulesConditionalAgent` win rates match or exceed 2026-04-23 results (76.5–97%)

**If conditions missing:**
- `--dry-run` will list which hand-rank conditions are absent
- Re-launch; existing cache prevents recomputing completed conditions
- Check `agent/data/extended_conditional_exact_probs.json` to see what's already there

---

## Quick-Start Checklist for Next Session

```bash
# 1. Check MC
python3 -m agent.data.compute_extended_conditional_exact_probs --dry-run

# 2. Fix eval.py (edit 3 call sites + 3 signatures)

# 3. Re-eval Stage A checkpoint under correct rules
python3 -m agent.rnad.eval --checkpoint agent/checkpoints/rnad_final.pt --exact-rules --episodes 1000

# 4. Diagnose CFR+ stall — run slow solver at max_bids=4 HH for 5k iters, compare plateau

# 5. If CFR+ diagnosed: rerun cfr_plus_mb4_hh with fix (or accept plateau and benchmark)
python3 -m agent.baseline.cfr_1v1_fast --run-name cfr_plus_mb4_hh_v2 \
    --total-iters 50000 --max-bids 4 --high-hand --solver fast
```
