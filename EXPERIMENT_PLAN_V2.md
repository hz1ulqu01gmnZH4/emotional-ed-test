# Emotional ED: Additional Experiments Plan (V2)

## Motivation

Critical review identified gaps in the experimental suite. This document plans additional experiments to strengthen the claims.

## Gap Analysis Summary

| Gap | Severity | Experiment |
|-----|----------|------------|
| No statistical rigor | Critical | Exp 12: Statistical validation |
| No reward-shaping ablation | Critical | Exp 13: Reward shaping comparison |
| No positive emotions | High | Exp 14: Joy/curiosity channel |
| No failure modes | Medium | Exp 15: Emotional interference |
| Grief/Regulation too subtle | Low | Defer to richer environments |

---

## Experiment 12: Statistical Validation

### Hypothesis
Previous results are statistically significant and reproducible across random seeds.

### Design
- Re-run key experiments with N=50 seeds
- Report: mean, SD, 95% CI, p-value vs baseline
- Effect size: Cohen's d

### Experiments to Validate
1. Fear avoidance (Exp 1) - distance metric
2. Anger persistence (Exp 2) - wall hits
3. Regret learning (Exp 3) - optimal choice rate
4. Multi-channel integration (Exp 7) - risky goal rate
5. Transfer generalization (Exp 11) - threat hits

### Success Criteria
- p < 0.05 for all key comparisons
- Cohen's d > 0.5 (medium effect) for main claims

---

## Experiment 13: Reward Shaping Ablation

### Hypothesis
Emotional ED produces different behavior than equivalent reward shaping.

### Theoretical Background
The critical alternative hypothesis: "Emotions are just implicit reward shaping."

If fear channel adds signal φ when near threat, is this equivalent to:
- R' = R - φ (reward with penalty)?

ED claims NO: broadcast modulation of learning rates and action selection is computationally distinct from reward modification.

### Design

**Environment**: Same as Exp 1 (5×5 grid, threat at center)

**Agents**:
1. **Standard QL**: No fear, no shaping
2. **Reward Shaping**: R' = R - k × (1/threat_distance) [explicit penalty]
3. **Emotional ED**: Fear channel modulates LR and action selection (no reward change)
4. **Hybrid**: Both reward shaping AND emotional modulation

**Key Manipulations**:
- Match expected value impact: tune k so average reward modification ≈ average ED effect
- Measure: Learning curves, final behavior, sample efficiency

### Predictions
1. Reward shaping and ED produce similar asymptotic behavior
2. ED shows faster learning (direct signal vs slow credit assignment)
3. ED generalizes better to novel threats (feature-based, not state-specific)
4. Hybrid doesn't improve over ED alone (redundant signal)

### Success Criteria
- ED learns threat avoidance in fewer episodes than reward shaping
- ED transfers better to novel threat locations
- Demonstrates ED ≠ reward shaping mechanistically

---

## Experiment 14: Positive Emotion (Joy/Curiosity)

### Hypothesis
Positive emotions (joy, curiosity) drive exploration and approach behavior symmetrically to how fear drives avoidance.

### Theoretical Background
Fredrickson (2001) Broaden-and-Build: Positive emotions broaden attention and encourage exploration.
Silvia (2008): Interest/curiosity as approach-motivated positive affect.

All previous experiments focus on negative emotions. Positive emotions should:
1. Increase exploration (curiosity)
2. Bias toward rewarding states (joy)
3. Enable discovery without explicit reward signal

### Design

**Environment**: 7×7 grid with:
- Hidden reward (not visible until discovered)
- Neutral zones
- Small penalty for steps (encourages efficiency after discovery)

**Agents**:
1. **Standard QL**: Pure exploration via ε-greedy
2. **Curiosity-driven**: Novelty signal increases exploration toward unvisited states
3. **Joy-driven**: Positive experiences boost approach to similar states
4. **Integrated**: Curiosity + Joy

**Joy Module**:
```python
class JoyModule:
    def compute(self, context):
        if context.reward > 0:
            self.joy = min(1.0, self.joy + 0.3)
        else:
            self.joy *= 0.9  # decay
        return self.joy
```

Joy modulates:
- Action selection: bias toward states associated with past joy
- Learning rate: enhanced for positive outcomes

**Curiosity Module**:
```python
class CuriosityModule:
    def compute(self, state, visit_counts):
        novelty = 1.0 / (1.0 + visit_counts[state])
        return novelty * self.curiosity_weight
```

Curiosity modulates:
- Exploration: higher ε for novel states
- Intrinsic reward: bonus for visiting new states

### Predictions
1. Curiosity agent discovers hidden reward faster than standard
2. Joy agent returns to rewarding areas more reliably
3. Integrated agent shows both exploration AND exploitation benefits
4. Demonstrates positive emotions as parallel channels (not just intrinsic reward)

### Success Criteria
- Curiosity: Faster discovery (fewer steps to find hidden reward)
- Joy: Higher revisit rate to positive states
- Effect sizes comparable to fear/anger experiments

---

## Experiment 15: Emotional Interference (Failure Mode)

### Hypothesis
Emotional channels can HURT performance when miscalibrated or in conflict.

### Theoretical Background
Emotions aren't always adaptive:
- Anxiety disorders: Excessive fear prevents approach
- Anger dysregulation: Persistence when should quit
- Addiction: Wanting overrides better judgment

Demonstrating failure modes strengthens the claim that emotions are genuine control systems (not just performance boosters).

### Design

**Environment**: 6×6 grid with:
- Goal requiring path through "scary but safe" zone
- Optimal path goes near harmless visual threat
- Suboptimal safe path available (longer)

**Agents**:
1. **Standard QL**: Finds optimal path
2. **Calibrated Fear**: Learns threat is safe, takes optimal path
3. **Excessive Fear**: High fear weight, can't approach "threat"
4. **Inflexible Anger**: Persists at impossible obstacles
5. **Conflicted**: High fear AND high approach (paralysis)

**Manipulations**:
- Excessive Fear: fear_weight = 2.0 (vs normal 0.5)
- Inflexible Anger: anger_decay = 0.99 (vs normal 0.8)
- Conflicted: fear = 1.0, approach = 1.0 (competing signals)

### Predictions
1. Excessive fear agent avoids optimal path (suboptimal behavior)
2. Inflexible anger agent wastes steps on impossible obstacles
3. Conflicted agent shows timeout/paralysis (high "none" rate)
4. Standard QL outperforms miscalibrated emotional agents

### Success Criteria
- Clear performance degradation with miscalibration
- Demonstrates emotions as double-edged control systems
- Supports claim that emotions are genuine mechanisms (can fail)

---

## Experiment 16: Sample Efficiency Comparison

### Hypothesis
Emotional ED achieves target performance with fewer samples than baselines.

### Theoretical Background
If emotions provide direct supervision signals, they should accelerate learning compared to:
- Standard RL (reward-only)
- Reward shaping (indirect signal)

### Design

**Task**: Fear avoidance (reach goal, avoid threat)

**Agents**:
1. Standard QL
2. Reward Shaping (tuned)
3. Emotional ED (fear channel)

**Metric**: Episodes to reach 90% optimal performance

### Predictions
- ED: ~100 episodes
- Reward Shaping: ~200 episodes
- Standard: ~500 episodes (or never if no penalty)

### Success Criteria
- ED reaches criterion in <50% episodes of reward shaping
- Demonstrates practical advantage of emotional channels

---

## Implementation Order

1. **Exp 12: Statistical Validation** - Quick, strengthens existing results
2. **Exp 13: Reward Shaping Ablation** - Critical for main claim
3. **Exp 14: Joy/Curiosity** - Fills positive emotion gap
4. **Exp 15: Failure Modes** - Strengthens mechanistic claim
5. **Exp 16: Sample Efficiency** - Practical advantage demonstration

---

## Expected Outcomes

If experiments succeed:
- Claims 1-3 supported with statistical rigor
- Alternative hypothesis (reward shaping) falsified
- Positive emotions demonstrate symmetry
- Failure modes prove genuine mechanism
- Sample efficiency shows practical value

If experiments fail:
- Revise ED architecture
- Acknowledge limitations in claims
- Identify boundary conditions

---

---

## Results Summary

| Experiment | Status | Key Finding |
|------------|--------|-------------|
| Exp 12: Statistical | ✓ Complete | 3/5 significant (Fear, Anger, Integration) |
| Exp 13: Reward Shaping | ✓ Complete | RS outperformed ED in this test |
| Exp 14: Joy/Curiosity | ✓ Complete | Environment too easy for differentiation |
| Exp 15: Failure Modes | ✓ Complete | **3/3 failure modes demonstrated** |
| Exp 16: Sample Efficiency | ✓ Complete | No significant speedup (1.07x) |

### Key Takeaways

**Strongest evidence (Exp 15):**
- Excessive fear: 100% → 0% goal rate (complete failure)
- Inflexible anger: 0.8 → 7.5 wall hits (+837%)
- Emotional conflict: 4.3 → 9.3 steps (+116%)

This proves emotions are **genuine control mechanisms** that can malfunction, not just performance boosters.

**Mixed evidence:**
- Statistical validation shows medium-large effects for Fear, Anger, Integration
- Regret and Transfer effects not statistically significant (need larger N or different metrics)
- Reward shaping surprisingly outperformed ED in direct comparison

**Inconclusive:**
- Joy/Curiosity environment was too easy
- Sample efficiency advantage not demonstrated

---

*Plan created: 2024*
*Results updated: 2024*
*Based on critical review feedback*
