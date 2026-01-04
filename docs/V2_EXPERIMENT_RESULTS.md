# V2 Architecture Experiment Results

*Completed: December 2025*

---

## Executive Summary

We tested the V2 architectural fixes with N=50 statistical validation. Results are mixed:

| Experiment | V1 Result | V2 Result | Status |
|------------|-----------|-----------|--------|
| **Exp 19: Disgust V2** | d=0.25 reversed | **d=-1.24** ✓ | **FIXED** |
| **Exp 20: Transfer V2** | d=0.12 | d=-0.20 | Inconclusive |
| **Exp 21: Regulation V2** | d=-0.36 reversed | d=-2.06 reversed | Still broken |
| **Exp 22: CVaR Fear** | (new) | d=0.62 unexpected | Needs tuning |

**One major success**: Disgust V2 completely fixes the reversed effect with a large effect size.

---

## Experiment 19: Disgust V2 (Directional Repulsion)

### Fix Applied
Changed from argmax boost to directional repulsion:
```python
# Before (V1): Boost best action (can be toward contaminant!)
q_values[np.argmax(q_values)] *= (1 + disgust_level)

# After (V2): Boost action AWAY, penalize action TOWARD
away_action = self._get_action_away_from_contaminant(state)
q_values[away_action] += disgust_level * weight * 0.5
toward_action = self._get_action_toward_contaminant(state)
q_values[toward_action] -= disgust_level * weight * 0.3
```

### Results

| Metric | Disgust V2 | Fear (baseline) |
|--------|------------|-----------------|
| Contamination Touches | 115.12 ± 75.76 | 230.32 ± 107.62 |
| Average Reward | 0.219 ± 0.638 | 0.743 ± 0.271 |

**Statistics:**
- t-statistic: -6.127
- p-value: **0.0000** ✓
- Cohen's d: **-1.24** (large effect, correct direction)

### Interpretation
- **Major success**: Disgust V2 has **50% fewer** contamination touches than habituating fear
- The lower reward is expected: disgust avoids short paths through contaminants
- Effect size improved from d=0.25 (reversed) to d=-1.24 (large, correct)

---

## Experiment 20: Transfer V2 (Feature-Based Q)

### Fix Applied
Changed from tabular Q to feature-based linear approximation:
```python
# Features: [bias, threat_proximity, goal_proximity, toward_threat,
#            toward_goal, near_threat, fear_level, fear×approach]
Q(state, action) = W[:, action] · φ(state, context, action)
```

### Results

| Phase | Feature-based | Tabular |
|-------|---------------|---------|
| Training hits (threat at 2,2) | 0.040 ± 0.040 | 0.521 ± 0.068 |
| Test hits (NEW threat at 4,3) | 0.000 ± 0.000 | 0.020 ± 0.140 |

**Statistics:**
- t-statistic: -1.000
- p-value: 0.3198
- Cohen's d: -0.20 (small effect, correct direction)

### Interpretation
- **Inconclusive**: Both agents avoid threats well at test time
- Feature-based agent learned much faster during training (0.04 vs 0.52 hits/ep)
- The test is too easy: threat avoidance generalizes even for tabular in this setup
- **Need harder test**: Try more distant threat, or measure approach tolerance

---

## Experiment 21: Regulation V2 (Bayesian Reappraisal)

### Fix Applied
1. Bayesian belief update: P(safe|observations) using beta-binomial posterior
2. Credit assignment fix: Belief flows to TD target
3. Environment with fake threats: Looks scary but gives +0.4 bonus

### Results

| Metric | Regulated | Unregulated |
|--------|-----------|-------------|
| Average Reward | 0.206 ± 0.777 | 1.338 ± 0.046 |
| Fake Bonuses Collected | 96.20 ± 9.04 | 91.22 ± 4.38 |
| Real Threat Harm | 11.66 ± 3.77 | 10.36 ± 2.34 |

**Statistics:**
- t-statistic: -10.174
- p-value: 0.0000
- Cohen's d: **-2.06** (large effect, REVERSED)

### Interpretation
- **Still broken**: Regulation V2 has WORSE performance
- Interesting: Regulated agent collects MORE fake bonuses (96 vs 91, d=0.70)
- This suggests reappraisal IS working (approaching fake threats more)
- But: Much higher variance (0.777 vs 0.046) indicates learning instability
- **Root cause**: Reappraised value modulation may be interfering with Q-learning convergence

### Possible fixes
1. Reduce modulation magnitude in TD target
2. Use separate learning rates for reappraised vs raw values
3. Delay reappraisal until after initial Q-learning phase

---

## Experiment 22: CVaR Fear (Distributional RL)

### Implementation
- Maintain quantile distribution of returns (21 quantiles)
- Fear level controls CVaR alpha: high fear → low alpha → focus on worst-case
- Asymmetric Huber loss for quantile regression

### Results

| Metric | CVaR Fear | Risk-Neutral |
|--------|-----------|--------------|
| Threat Hits | 11.58 ± 2.89 | 9.94 ± 2.37 |
| Average Reward | 0.634 ± 0.019 | 0.665 ± 0.018 |
| Goal Completions | 97.42 ± 0.87 | 97.30 ± 0.88 |

**Statistics:**
- t-statistic: 3.070
- p-value: 0.0028
- Cohen's d: **0.62** (medium effect, UNEXPECTED direction)

### Interpretation
- **Unexpected**: CVaR Fear has MORE threat hits, not fewer
- This is counterintuitive for risk-averse behavior
- **Possible causes**:
  1. Quantile regression converges slower → less learned policy
  2. CVaR action selection explores more near threats
  3. Fear modulates alpha but doesn't affect initial Q distribution
  4. 21 quantiles may be too few for accurate CVaR estimation

### Possible fixes
1. More training episodes (currently 100)
2. Pre-train with expected value, then switch to CVaR
3. Use more quantiles (51 or 101)
4. Lower learning rate for more stable quantile updates

---

## Summary: V2 Architecture Status

### ✓ WORKING (Fixed)

| Experiment | V1 | V2 | Improvement |
|------------|----|----|-------------|
| **Disgust** | d=0.25 reversed | d=-1.24 correct | **+1.49** |

### ~ PARTIAL (Needs Tuning)

| Experiment | Issue | Suggested Fix |
|------------|-------|---------------|
| Transfer | Test too easy | Harder generalization test |
| CVaR Fear | Slower learning | More training, pre-training |

### ✗ STILL BROKEN

| Experiment | Issue | Suggested Fix |
|------------|-------|---------------|
| Regulation | Learning instability | Reduce modulation magnitude |

---

## Next Steps

1. **Disgust**: ✓ Complete - directional repulsion works
2. **Transfer**: Design harder test (threat at corner, agent must navigate around)
3. **Regulation**: Fix learning instability
   - Option A: Reduce `reappraised_next_value` boost from 0.2 to 0.05
   - Option B: Only apply reappraisal after 50 episodes of Q-learning
4. **CVaR Fear**: Increase training and quantile count
   - Try 200 episodes, 51 quantiles
   - Consider hybrid: expected value for first 50%, CVaR for rest

---

## Key Lessons

### 1. Directional Mechanisms Beat Argmax Modulation
Disgust fix shows that computing direction (away from threat) works much better than boosting the current best action. This principle may apply to other emotions.

### 2. Learning Stability is Critical
Regulation V2 shows the right behavior (more fake bonuses) but destabilizes learning. Any modulation that affects TD targets needs careful tuning.

### 3. Distributional RL Needs More Training
CVaR requires learning an entire distribution, not just a point estimate. The quantile regression may need 2-3x more episodes than standard Q-learning.

### 4. Test Design Matters
Transfer V2 was inconclusive because the test was too easy. Future tests should ensure baselines struggle before comparing.

---

*Results compiled: December 2025*
