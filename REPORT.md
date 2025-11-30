# Emotional Error Diffusion: Experimental Report

## Abstract

This report documents experiments testing the hypothesis that multiple emotional channels broadcasting distinct learning signals produce qualitatively different behavior than single-reward reinforcement learning. Using a minimal tabular grid-world, we demonstrate that fear and anger channels—implemented as modulators of Q-learning—generate behaviors consistent with biological emotion systems: threat avoidance without explicit reward penalty (fear) and persistence at obstacles before rerouting (anger/frustration).

## 1. Introduction

### 1.1 Background

Standard reinforcement learning uses a single scalar reward signal to guide behavior. Biological agents, however, possess multiple emotional systems that modulate learning and action selection in distinct ways. The Error Diffusion (ED) architecture proposes that emotions function as parallel value channels, each broadcasting different learning signals across the network.

### 1.2 Key Claims

1. **Fear** produces threat avoidance even without explicit negative reward
2. **Anger** produces approach-motivated persistence under negative valence (Davidson, 1992)
3. These channels are computationally distinct from reward shaping

### 1.3 Experimental Approach

We implement the simplest possible test: tabular Q-learning in a 5×5 grid-world. This eliminates confounds from neural network architecture, allowing direct observation of how emotional channels affect learning dynamics.

## 2. Experiment 1: Fear Channel

### 2.1 Design

**Environment**: 5×5 grid with threat at center (2,2), goal at corner (4,4).

```
A . . . .
. . . . .
. . X . .
. . . . .
. . . . G
```

**Manipulation**:
- Standard agent receives NO threat penalty in reward
- Emotional agent has fear channel that detects threat proximity

**Key question**: Does fear alone cause threat avoidance?

### 2.2 Fear Module Implementation

```python
class FearModule:
    def compute(self, context):
        if context.threat_distance >= self.safe_distance:
            return 0.0
        return self.max_fear * (1 - context.threat_distance / self.safe_distance)
```

Fear signal increases linearly as threat approaches, modulating:
1. **Learning rate** for negative outcomes (heightened danger awareness)
2. **Action selection** bias toward higher-Q actions (risk aversion)

### 2.3 Results

| Metric | Standard | Emotional ED |
|--------|----------|--------------|
| Mean min threat distance | **0.00** | **1.00** |
| Steps to goal | 8 | 8 |
| Success rate | 100% | 100% |

**Paths taken:**

Standard agent (goes THROUGH threat):
```
0 1 . . .
. 2 3 . .
. . X 5 6
. . . . 7
. . . . G
```

Emotional agent (goes AROUND threat):
```
0 . . . .
1 2 . . .
. 3 X . .
. 4 5 6 7
. . . . G
```

### 2.4 Interpretation

The fear channel produces threat avoidance **without any reward penalty**. The standard agent, blind to threat, takes the direct path through position (2,2). The emotional agent, receiving fear signal near threat, learns to route around despite identical reward structure.

This demonstrates that emotional channels can guide behavior independently of explicit reward shaping.

## 3. Experiment 2: Anger/Frustration Channel

### 3.1 Theoretical Background

Davidson (1992) identified anger as **approach-motivated negative affect**—unlike fear (withdrawal), anger increases approach vigor. Berkowitz (1989) showed frustration increases with goal proximity when blocked.

**Prediction**: Frustrated agents should persist longer at obstacles before rerouting, unlike standard RL which immediately seeks alternatives.

### 3.2 Design

**Environment**: 5×5 grid with wall blocking direct path.

```
A . . . .
. . . . .
. . # # .
. . # . .
. . . . G
```

**Manipulation**:
- Standard agent: Blocked → negative TD error → avoid
- Frustrated agent: Blocked → frustration builds → persist → eventually reroute

### 3.3 Anger Module Implementation

```python
class AngerModule:
    def compute(self, context):
        if context.was_blocked:
            proximity_factor = 1.0 / (1.0 + context.goal_distance)
            consecutive_factor = 1.0 + 0.2 * min(context.consecutive_blocks, 5)
            increment = self.buildup * proximity_factor * consecutive_factor
            self.frustration = min(1.0, self.frustration + increment)
        else:
            self.frustration *= self.decay
        return self.frustration
```

Frustration modulates learning by:
1. **Slowing negative learning** for blocked actions (don't give up immediately)
2. **Boosting recently-blocked actions** in selection (persistence)

### 3.4 Results

**Early Learning (first 20 episodes):**

| Episode | Standard Hits | Frustrated Hits |
|---------|---------------|-----------------|
| 1 | 16 | 15 |
| 2 | 10 | 14 |
| 3 | 11 | **30** |
| 4 | 17 | 5 |
| 5 | 10 | 21 |
| 6 | 10 | **33** |
| ... | ... | ... |
| **Total** | **195** | **217** |

**Difference: +22 wall hits (+11% more persistence)**

**Converged Behavior (after 300 episodes):**

| Metric | Standard | Frustrated ED |
|--------|----------|---------------|
| Success rate | 100% | 100% |
| Mean steps | 8 | 8 |
| Mean wall hits | 0 | 0 |

Both agents converge to identical optimal paths, but learning trajectories differ.

### 3.5 Interpretation

The frustrated agent shows **11% more wall-hitting behavior** during early learning—exactly as predicted by approach-motivated negative affect. This isn't maladaptive in all contexts: persistence at obstacles can be beneficial when obstacles are temporary or surmountable.

The anger channel produces qualitatively different learning dynamics while preserving asymptotic optimality.

## 4. Discussion

### 4.1 Summary of Findings

| Channel | Hypothesis | Confirmed? | Effect Size |
|---------|-----------|------------|-------------|
| Fear | Threat avoidance without reward | ✓ Yes | +1.0 distance units |
| Anger | Persistence at obstacles | ✓ Yes | +11% wall hits |

### 4.2 What These Results Show

1. **Emotional channels are not reward shaping**: Fear produces avoidance without negative reward; anger produces persistence without positive reward for wall-hitting.

2. **Channels affect learning dynamics, not just outcomes**: Both agents converge to optimal behavior, but take different paths through learning space.

3. **Minimal implementation suffices**: Tabular Q-learning + simple emotional modules demonstrate the core phenomenon. No neural networks required.

### 4.3 Limitations

1. **Grid-world simplicity**: Real emotional systems operate in high-dimensional continuous spaces.

2. **Hand-tuned parameters**: Fear weight, anger decay, etc. were set manually. Biological systems presumably tune these through development/evolution.

3. **No cross-emotion interaction**: Current tests isolate single channels. Real emotions interact (fear vs. anger competition, emotion regulation).

4. **No temporal dynamics**: Moods (tonic) vs. emotions (phasic) distinction not tested.

### 4.4 Relation to Architecture Document

These experiments test a subset of the full emotional ED architecture:

| Module | Tested? | Status |
|--------|---------|--------|
| Fear/Threat | ✓ | Confirmed |
| Anger/Frustration | ✓ | Confirmed |
| Disgust | ✗ | Not implemented |
| Wanting/Liking | ✗ | Not implemented |
| Regret (counterfactual) | ✗ | Planned |
| Grief (attachment) | ✗ | Planned |
| Emotion Regulation | ✗ | Planned |
| Cross-emotion Arbitration | ✗ | Planned |

## 5. Future Experiments

### 5.1 Regret Test (Two-Door Paradigm)

Following Coricelli et al. (2005):
- Agent chooses door A or B
- After choice, sees reward behind BOTH doors
- Measure: Does counterfactual information affect future choices?

### 5.2 Grief/Loss Test

- Agent bonds to renewable resource
- Resource disappears permanently
- Measure: Duration of "seeking" behavior before adaptation

### 5.3 Anger vs. Fear Interaction

- Valuable reward placed near threat
- Measure: How do competing approach (anger) and avoidance (fear) channels resolve?

### 5.4 Emotion Regulation Test

- Same threatening stimulus, different "framing"
- Measure: Can learned reappraisal reduce fear response? (V(f(s)) ≠ V(s))

## 6. Conclusion

These minimal experiments provide initial evidence that:

1. Emotional channels produce qualitatively different behavior from single-reward RL
2. Fear and anger channels function as predicted by affective neuroscience literature
3. The emotional ED architecture is computationally tractable and testable

The hypothesis that emotions function as parallel value systems—not just reward modifiers—remains viable and warrants further investigation.

## References

- Berkowitz, L. (1989). Frustration-aggression hypothesis: Examination and reformulation. *Psychological Bulletin*, 106(1), 59-73.
- Coricelli, G., et al. (2005). Regret and its avoidance: A neuroimaging study of choice behavior. *Nature Neuroscience*, 8(9), 1255-1262.
- Davidson, R. J. (1992). Anterior cerebral asymmetry and the nature of emotion. *Brain and Cognition*, 20(1), 125-151.
- Kaneko, I. (2024). Error Diffusion method for neural network training.

## Appendix: Running the Experiments

```bash
# Clone repository
git clone https://github.com/hz1ulqu01gmnZH4/emotional-ed-test.git
cd emotional-ed-test

# Run fear test
python test_fear.py

# Run anger test
python test_anger.py
```

No dependencies beyond NumPy.

---

*Report generated: 2024*
*Repository: https://github.com/hz1ulqu01gmnZH4/emotional-ed-test*
