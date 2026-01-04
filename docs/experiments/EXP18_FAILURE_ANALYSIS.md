# Experiment 18 Failure Analysis: Fear + DQN in Stochastic Cliff

**Date:** 2026-01-04
**Status:** FAILED (p=0.236, d=-0.31)

## Summary

Exp 18 tested fear-modulated DQN in a stochastic CliffWalking environment (20% slip chance). Despite the hypothesis that fear should encourage safer paths, Fear-ED performed WORSE than standard DQN (not significantly) and had 3x higher reward variance.

## Results Recap

| Metric | Standard DQN | Fear-ED | Direction |
|--------|--------------|---------|-----------|
| Survival Rate | 88.8% | 82.3% | WORSE |
| Mean Reward | -18.4 | -25.0 | WORSE |
| Reward Variance | 207.3 | 647.4 | 3x WORSE |

## Root Cause Analysis

### Issue 1: Stale Fear in Replay Buffer

```python
# Line 404: Fear computed from STORED context
fears = torch.tensor([self._compute_fear(ctx) for ctx in contexts], ...)
```

The fear signal is computed from contexts stored in the replay buffer, which can be thousands of steps old. This creates a temporal mismatch:
- Agent is learning from past experiences
- Fear signal reflects past proximity, not current danger
- Results in learning incorrect fear associations

**Fix:** Use current state's proximity to compute fear during update, not stored context.

### Issue 2: Double Fear Influence

Fear affects BOTH:
1. **State encoding** (line 357): `tensor[-1] = fear`
2. **Loss weighting** (line 426): `fear_weights = 1 + fears * fear_weight`

This creates unstable gradients:
- Fear augments state representation
- Fear also weights the loss function
- Network receives conflicting signals

**Fix:** Choose ONE mechanism, not both. Ablation studies show LR modulation alone doesn't work, but combining them requires careful balancing.

### Issue 3: Hardcoded Fear Action Bias

```python
# Line 375: Fear forces UP action
if fear > 0.5 and random.random() < fear:
    return 0  # Move up (away from cliff row)
```

During exploration, high fear FORCES the "up" action. This:
- Reduces exploration of safe paths along the top
- Creates bias in experience replay
- May not be the safest action in all states

**Fix:** Bias toward AWAY from threat, not hardcoded direction.

### Issue 4: Massive Q-Value Manipulation

```python
# Line 385: Fear adds huge bonus
q_values[0, 0] += fear * self.fear_weight * 10  # = 5.0 when fear=1.0
```

Fear adds up to 5.0 to the Q-value of action 0. In a network with typical Q-values in [-100, +10] range, this:
- Creates discontinuous Q-value landscape
- Interferes with TD learning targets
- May cause gradient explosion/vanishing

**Fix:** Scale fear bonus relative to current Q-value range.

### Issue 5: Tonic Fear Decay During Batch Update

```python
# Line 350: Fear decays during computation
self.fear_level *= self.fear_decay
```

The `_compute_fear` method modifies `self.fear_level` as a side effect. When called during batch update (line 404), this:
- Decays fear 64 times per batch (batch_size=64)
- Results in near-zero tonic fear
- Makes fear effectively phasic-only

**Fix:** Separate fear computation from state mutation.

## Why Tabular Fear Works but DQN Fear Fails

### Tabular Q-Learning (Works)

In tabular agents (Exp 1-7), fear modulation:
- Directly modifies Q[state, action] entries
- No function approximation = no gradient interference
- Discrete states = stable fear associations
- No replay buffer = fresh fear computation each step

### DQN (Fails)

In neural network agents:
- Gradients flow through both fear and Q-value computation
- Function approximation smooths fear across similar states
- Replay buffer introduces temporal lag in fear signals
- Batch updates create inconsistent fear levels

## Comparison with Working Experiments

| Experiment | Agent Type | Fear Mechanism | Result |
|------------|------------|----------------|--------|
| Exp 1: Fear | Tabular | Direct Q modulation | SUCCESS (d=1.09) |
| Exp 18: Slippery Cliff | DQN | State augmentation + loss weighting | FAILED (d=-0.31) |
| Exp 21: Visual Transfer | Feature-linear | Feature-based fear (no NN) | SUCCESS (d=1.68) |

## Recommendations

### Short-term Fixes

1. **Remove loss weighting**: Use fear only for state augmentation OR loss weighting, not both
2. **Fresh fear computation**: Compute fear from current state during update, not stored context
3. **Relative Q-value adjustment**: Scale fear bonus by Q-value range
4. **Separate fear state**: Don't mutate fear level during batch computation

### Long-term Architecture Changes

1. **Gradient blocking**: Use `fear.detach()` to prevent gradient flow through fear computation
2. **Separate fear network**: Train a fear prediction network independently
3. **Advantage-based modulation**: Modulate advantage function instead of Q-values
4. **Risk-sensitive TD**: Use CVaR or other risk measures instead of additive fear

## Proposed Fix: FearEDAgent_v2

```python
class FearEDAgentV2:
    """Fixed fear-modulated DQN."""

    def _compute_fear_from_state(self, state_idx: int) -> float:
        """Compute fear from state index, not context."""
        row, col = state_idx // self.width, state_idx % self.width
        cliff_dist = min(abs(row - 3) + abs(col - c) for c in range(1, 11))
        return max(0, 1 - cliff_dist / 3)  # Fear within 3 cells of cliff

    def update(self):
        # ... batch sampling ...

        # Compute fresh fear from CURRENT states (not stored contexts)
        fears = torch.tensor([self._compute_fear_from_state(s) for s in states])

        # Use fear ONLY for state augmentation (not loss weighting)
        state_batch = torch.stack([
            self._state_to_tensor(s, f.item()) for s, f in zip(states, fears)
        ])

        # Standard MSE loss (no fear weighting)
        loss = nn.functional.mse_loss(current_q.squeeze(), target_q)
```

## Conclusion

Exp 18 failed due to multiple interacting issues in how fear was integrated with DQN:
1. Stale fear from replay buffer
2. Double influence (state + loss)
3. Hardcoded action bias
4. Massive Q-value manipulation
5. Side-effect state mutation

The core insight is that **emotional modulation requires different architecture for function approximation** than for tabular methods. Simply adding fear signals to a standard DQN destabilizes learning rather than improving safety.

Future work should explore gradient-blocked architectures, separate emotion networks, or risk-sensitive TD methods.
