# Emotional ED: Fair Test Environments Plan (V3)

## Motivation

V2 experiments showed:
- **Tabular Q-learning**: Emotional channels help (Fear p=0.013, Anger p=0.001)
- **Neural Network DQN**: Emotional modulation HURTS (destabilizes gradients)

**Problem**: Test environments too simple. Standard DQN solves 5×5 gridworlds easily.
Dense emotional signals become noise in dense reward environments.

**Solution**: Design environments where emotional signals provide information advantage.

---

## Key Insight from Gemini Consultation

> "Dense emotional signals are noise in dense reward environments, but they become signal in sparse/harsh environments."

Standard DQN easily solves simple tasks. Emotional channels need environments where:
1. Rewards are sparse (emotional signals fill the gap)
2. Environment is non-stationary (emotional memory helps adaptation)
3. Safety matters (fear prevents catastrophic failures)
4. Transfer is required (feature-based emotions generalize)

---

## Proposed Experiments

### Exp 17: "Pitch Black" Key Search (Sparse Reward / Curiosity)

**Environment**: Large MiniGrid-KeyCorridor (16×16+)
- Agent must explore dead-end rooms to find key
- Then return to open door
- Reward = 0 until door opened (maximally sparse)

**Why Emotional Channels Help**:
- Standard DQN relies on ε-greedy luck
- Curiosity channel broadcasts positive signal for unvisited states
- Agent systematically searches instead of random walking

**Metrics**:
- Steps to first solve
- Total distinct states visited
- Success rate within N episodes

**Complexity**: Low (2D grid, very fast)

**Expected Result**: EED finds solution in significantly fewer episodes

**Implementation**:
```python
class CuriosityModule:
    def compute(self, state, visit_counts):
        novelty = 1.0 / (1.0 + visit_counts[state])
        return novelty * self.curiosity_weight
```

---

### Exp 18: "Slippery Cliff" (Safety / Fear)

**Environment**: Modified CliffWalking
- Path from Start to Goal flanked by cliff
- 20% chance agent moves in random direction (stochastic)
- Falling = -100 reward, episode terminates

**Why Emotional Channels Help**:
- Standard DQN averages Q-values, hugs cliff edge for optimality
- High stochasticity causes frequent deaths
- Fear channel spikes on negative reward / cliff proximity
- Acts as "safety bias" forcing safer sub-optimal path

**Metrics**:
- Survival rate (episodes without falling)
- Variance of return
- Time to convergence on safe policy

**Complexity**: Very Low

**Expected Result**: EED converges to safer policy faster; DQN oscillates or requires massive training

**Implementation**:
```python
class SafetyFearModule:
    def compute(self, context):
        if context.near_cliff:
            return self.max_fear
        if context.reward < -10:  # Pain signal
            self.fear = min(1.0, self.fear + 0.5)
        return self.fear
```

---

### Exp 19: "Changing Seasons" (Non-Stationary / Surprise)

**Environment**: Grid with two food sources
- Red Berries and Blue Berries
- Every N episodes (500), preference flips:
  - Season A: Red = +1, Blue = -1
  - Season B: Red = -1, Blue = +1

**Why Emotional Channels Help**:
- Standard DQN suffers catastrophic forgetting / slow adaptation
- Surprise/Anger channel detects prediction error
- High surprise → increase learning rate locally
- Enables rapid adaptation to regime change

**Metrics**:
- Recovery time (steps to positive reward after season flip)
- Cumulative reward across season changes
- Number of negative rewards post-flip

**Complexity**: Low

**Expected Result**: EED adapts almost immediately; DQN has long negative reward period

**Implementation**:
```python
class SurpriseModule:
    def compute(self, expected_reward, actual_reward):
        prediction_error = abs(expected_reward - actual_reward)
        if prediction_error > self.threshold:
            self.surprise = min(1.0, prediction_error / self.max_error)
            return self.surprise  # Modulates learning rate
        return 0.0
```

---

### Exp 20: "Bottleneck Trap" (Persistence / Anger)

**Environment**: MiniGrid-DoorKey variant
- Door is "jammed" - requires 5 consecutive toggle actions
- Standard navigation gives small negative rewards (time penalty)

**Why Emotional Channels Help**:
- Standard DQN treats door as wall after failed attempts
- Anger/Frustration accumulates when blocked
- High anger → increase action randomness or prioritize forceful actions
- Enables "pushing through" local minima

**Metrics**:
- Success rate in opening door
- Episodes to first success
- Time spent at door before giving up (DQN) vs persisting (EED)

**Complexity**: Low

**Expected Result**: DQN gets stuck wandering; EED "rages" through bottleneck

---

### Exp 21: Visual Hazard Transfer (Transfer / Fear)

**Environment**: Visual input (CNN-based)
- Task A: Avoid RED squares (Lava) in maze
- Task B: Avoid RED triangles (Enemies) in open field

**Protocol**:
1. Train on Task A
2. Freeze Emotional Network layers
3. Train on Task B

**Why Emotional Channels Help**:
- Standard DQN: "Lava" and "Enemies" are different Q-values
- Fear channel associates RED color with danger
- In Task B, Fear immediately outputs "Danger" on RED triangle
- Zero-shot fear transfer before Q-value learned

**Metrics**:
- Zero-shot survival time in Task B
- Steps to convergence on Task B
- Initial avoidance behavior (before any B training)

**Complexity**: Medium (requires CNN encoder)

**Expected Result**: EED shows "Zero-shot fear" - avoids new threat immediately

---

### Exp 22: "Predator-Prey" (Adversarial / Anxiety)

**Environment**: PettingZoo Simple Tag or custom 2D continuous
- Agent is slower than Predator but more agile
- Must eat food pellets while being chased
- Getting caught = -100

**Why Emotional Channels Help**:
- Standard DQN struggles to balance eat vs. run
- Anxiety channel triggers on predator proximity
- Modulates policy: prioritize evasion over food dynamically
- Enables "fight or flight" switching

**Metrics**:
- Average lifespan
- Food collected per episode
- Evasion success rate when predator approaches

**Complexity**: Low/Medium

**Expected Result**: EED exhibits context-dependent behavior; DQN ignores predator until too late

---

### Exp 23: "Battery Run" (Long-horizon / Tonic Mood)

**Environment**: LunarLander or drone navigation
- Agent has Fuel bar (moving costs fuel)
- Landing pads refuel
- Goal: visit N waypoints
- Empty fuel = crash

**Why Emotional Channels Help**:
- Standard DQN gets greedy for next waypoint
- Tonic Anxiety rises as fuel drops
- Modulates discount factor γ (more short-sighted when low fuel)
- Biases toward "nearest fuel" regardless of mission

**Metrics**:
- Waypoints visited before failure
- Fuel management efficiency
- Abort-to-refuel decisions

**Complexity**: Low (Box2D physics)

**Expected Result**: EED aborts mission to refuel; DQN runs out of gas

---

## Implementation Architecture Fix

**Problem from V2**: Emotional gradients destabilized neural network training.

**Solution**: Gradient Blocking / Input Modulation

```
┌─────────────────────────────────────────────────────────────┐
│                      CNN Feature Trunk                       │
│              (trained ONLY by Q-loss, or 90/10)             │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Emotional Net   │     │    Q-Network    │
│ (parallel head) │     │  (main policy)  │
└────────┬────────┘     └────────┬────────┘
         │ stop_gradient         │
         │                       │
         ▼                       │
    Fear, Anger,                 │
    Curiosity scalars            │
         │                       │
         └───────┬───────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Q-Network Dense │
        │ (emotion as     │
        │  input context) │
        └─────────────────┘
```

**Key Changes**:
1. Emotional outputs don't backprop through CNN trunk
2. Emotions provided as INPUT to Q-network dense layers
3. State augmentation, not gradient interference

---

## Priority Order

| Priority | Experiment | Rationale |
|----------|------------|-----------|
| 1 | Exp 18: Slippery Cliff | Simplest, tests core fear → safety claim |
| 2 | Exp 17: Key Search | Tests curiosity, sparse reward |
| 3 | Exp 19: Changing Seasons | Tests adaptation, surprise |
| 4 | Exp 20: Bottleneck | Tests anger/persistence |
| 5 | Exp 21: Visual Transfer | Tests generalization (medium complexity) |
| 6 | Exp 22: Predator-Prey | Tests dynamic switching |
| 7 | Exp 23: Battery Run | Tests tonic/mood effects |

---

## Dependencies

```bash
# Required packages
uv pip install gymnasium minigrid pettingzoo torch numpy
```

---

## Success Criteria

For each experiment, EED should demonstrate:
1. **Statistically significant** improvement (p < 0.05)
2. **Meaningful effect size** (Cohen's d > 0.5)
3. **Qualitative behavioral difference** (not just faster convergence)

If EED fails in these environments, the architecture may need fundamental revision.

---

*Plan created: 2024*
*Based on Gemini consultation identifying fair test scenarios*
*Addresses V2 negative result: simple environments favor standard DQN*
