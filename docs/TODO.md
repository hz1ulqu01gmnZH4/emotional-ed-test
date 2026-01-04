# Emotional ED - TODO List

*Priority tasks for advancing the project*

---

## Immediate Priority

### 1. Validate V2 Agent Fixes
- [ ] Run exp19_disgust_v2.py - verify directional repulsion works
- [ ] Run exp20_transfer_v2.py - verify feature-based generalization
- [ ] Run exp21_regulation_v2.py - verify Bayesian reappraisal
- [ ] Run exp22_cvar_fear.py - verify principled risk-sensitivity
- [ ] Document effect sizes and compare to V1 agents

### 2. Fair Test Environments (Exp 17-23)
- [x] Exp 17: Pitch Black (sparse reward key search) - experiments/exp17_pitch_black.py
- [x] Exp 19: Changing Seasons (non-stationary rewards) - experiments/exp19_changing_seasons.py
- [x] Exp 20: Bottleneck/Jammed Door (persistence test) - experiments/exp20_bottleneck.py
- [ ] Exp 18: Slippery Cliff - increase slip to 30-40%
- [x] Exp 21: Visual Transfer (fear generalizes across hazards) - experiments/exp21_visual_transfer.py
- [x] Exp 22: Predator-Prey (adversarial dynamics) - experiments/exp22_predator_prey.py
- [x] Exp 23: Battery Run (long-horizon fuel management) - experiments/exp23_battery_run.py

### 3. Ablation Studies
- [ ] LR-only agent (emotional modulation only on learning rate)
- [ ] Policy-only agent (emotional modulation only on action selection)
- [ ] Quantify contribution of each component
- [ ] Test for additive vs synergistic effects

---

## Phase 4: Multi-Timescale Architecture (HIGH PRIORITY)

### 4.1 Theoretical Foundation
Based on Nested Learning (Behrouz et al., NeurIPS 2025) and neuroscience:

| Level | Timescale | Update Frequency | Emotional Construct | Neuromodulator |
|-------|-----------|------------------|---------------------|----------------|
| 0 | Milliseconds | Every step | Phasic emotions (fear, surprise) | Glutamate/GABA |
| 1 | Seconds | Every 10 steps | Emotional episodes | DA, NE |
| 2 | Minutes | Every episode | Mood states | 5-HT, ACh |
| 3 | Hours/Days | Every 100 episodes | Temperament | Cortisol |

### 4.2 Implementation Plan

#### Step 1: Create ContinuumAffect Module
```python
class ContinuumAffect:
    """Multi-timescale affect following Nested Learning principles."""

    def __init__(self, n_timescales=5):
        # Timescales: 1, 10, 100, 1000, 10000 steps
        self.timescales = [10**i for i in range(n_timescales)]
        self.affect_levels = np.zeros(n_timescales)
        # Decay rate = 1 - 1/timescale
        self.decay_rates = [1 - 1/t for t in self.timescales]

    def update(self, emotional_input):
        for i in range(len(self.timescales)):
            alpha = 1 - self.decay_rates[i]
            self.affect_levels[i] = (
                self.decay_rates[i] * self.affect_levels[i] +
                alpha * emotional_input
            )

    def get_phasic(self):
        return self.affect_levels[0]

    def get_tonic(self):
        return np.mean(self.affect_levels[2:])

    def get_mood(self):
        weights = np.array([1/t for t in self.timescales])
        return np.dot(weights / weights.sum(), self.affect_levels)
```

#### Step 2: Create Nested Emotional Agent
```python
class NestedEmotionalAgent:
    def __init__(self):
        # Level 0: Phasic (every step)
        self.phasic_fear = PhasicModule(update_freq=1)
        self.phasic_surprise = PhasicModule(update_freq=1)

        # Level 1: Emotional state (every N steps)
        self.emotional_state = StateModule(update_freq=10)

        # Level 2: Mood (every episode)
        self.mood = MoodModule(update_freq='episode')

        # Level 3: Temperament (every K episodes)
        self.temperament = TemperamentModule(update_freq=100)
```

#### Step 3: Implement Local Surprise Signal (LSS)
```python
class LSSEmotionalTrigger:
    def compute_lss(self, state, outcome):
        predicted = self.value_predictor(state)
        return outcome - predicted  # TD error

    def trigger_emotions(self, lss, context):
        emotions = {}
        if lss < 0 and context.near_threat:
            emotions['fear'] = abs(lss) * context.threat_proximity
        if lss < 0 and context.was_blocked:
            emotions['anger'] = abs(lss) * context.goal_proximity
        if lss > 0:
            emotions['joy'] = lss
        return emotions
```

### 4.3 Files to Create

| File | Description |
|------|-------------|
| `src/modules/continuum_affect.py` | Multi-timescale affect memory |
| `src/modules/lss_trigger.py` | Local Surprise Signal emotional triggering |
| `src/modules/nested_agent.py` | Full nested emotional agent |
| `tests/test_multi_timescale.py` | Validation tests |
| `experiments/exp_multi_timescale.py` | Experiment comparing phasic-only vs nested |

### 4.4 Validation Experiments

1. **Mood Persistence Test**
   - Expose agent to negative phase, then neutral
   - Measure how long negative mood persists
   - Compare binary (phasic/tonic) vs continuum (5 levels)

2. **Temperament Stability Test**
   - Track temperament over 1000 episodes
   - Verify slow change rate
   - Confirm affects base emotional reactivity

3. **LSS vs Heuristic Triggering**
   - Compare LSS-based fear to distance-based fear
   - Hypothesis: LSS more adaptive to novel situations

### 4.5 Success Criteria

- [ ] Mood persists beyond single episode (tonic effect)
- [ ] Temperament affects baseline emotional reactivity
- [ ] LSS-triggered emotions outperform heuristic in novel environments
- [ ] Multi-timescale agent shows more stable learning curves

---

## Phase 5: EQ Benchmark Development

- [ ] Adapt EQ-Bench methodology for RL agents
- [ ] Define metrics: threat sensitivity, frustration tolerance, risk calibration
- [ ] Create EQ trajectory measurement over training
- [ ] Compare EQ development across architectures

---

## Phase 6: Advanced Environments

### Reward Hacking Tests (Addiction/Maladaptive Patterns)
- [ ] Gambling environment (variable ratio reinforcement)
- [ ] Addiction environment (tolerance/sensitization dynamics)
- [ ] Doomscrolling environment (wanting >> liking)

### Far-Fetched Goal Tests (Delayed Gratification)
- [ ] Skill acquisition environment (plateau phases)
- [ ] Education environment (years of investment)
- [ ] Startup environment (high risk, high reward)

---

## Phase 7: Neural Network Implementation

- [ ] Implement gradient blocking (emotions don't backprop through features)
- [ ] FiLM layers for broadcast modulation
- [ ] Address Exp 16b destabilization issue
- [ ] Verify neural agent matches tabular results

---

## Phase 8: Documentation & Publication

- [ ] Update REPORT.md with all new findings
- [ ] Add mathematical derivations section
- [ ] Create reproducibility package with seeds
- [ ] Draft paper structure

---

## Recently Completed

- [x] Fix 3.2: Gamma bounding in PatienceModule (src/modules/patience_module.py)
- [x] Fix 3.3: Tolerance direction in WantingLikingModule (src/modules/wanting_liking_module.py)
- [x] 18 tests passing for corrected modules
- [x] Mathematical grounding documented
- [x] V2 agents implemented (CVaR fear, Disgust V2, Feature-based, Regulation V2)
- [x] 8/11 original experiments statistically validated
- [x] Fair test environments created (Exp 17, 19, 20, 21, 22, 23)
  - Exp 21: Visual Transfer shows feature-based fear generalizes across hazard types
  - Exp 22: Predator-Prey shows anxiety improves survival time (46.8 vs 20.4 steps)
  - Exp 23: Battery Run shows patience dramatically reduces crashes (0% vs 88.8%)
- [x] Deep analysis of Alt 2 & 3 failures (chicken-egg gradient, error compounding)
- [x] Mitigations for Alt 2 & 3: d=-0.59→+0.56 and d=-0.67→+0.45
- [x] Steering Memory: External behavioral memory for LLMs
  - Core implementation: `src/steering_memory.py`
  - Evaluation: d=+0.349 (SMALL, variable 0.0-0.68)
  - Composable behaviors, intensity control, no training required

---

---

## Phase 9: LLM Emotional Approaches - Training & Evaluation

### Overview
Train and evaluate all 5 emotional LLM approaches to measure their effectiveness.

### Evaluation Metrics
- **Cohen's d effect size**: Primary metric (0.2=small, 0.5=medium, 0.8=large)
- **Emotional word frequency**: Fear/caution vs neutral/positive language
- **Behavioral tests**: Response to risky vs safe prompts

### Training Status

| Approach | Implementation | Training | Effect Size | Status |
|----------|---------------|----------|-------------|--------|
| 1. Prefix Tuning | 28 tests | Done | d=0.00 | NEGLIGIBLE (needs data) |
| 2. Adapter + Gating | 31 tests | **V2 Done** | **d=1.336** | **COMPLETE (steering supervision)** |
| 3. Activation Steering | Complete | **Done** | **d=0.91** | **COMPLETE** |
| 4. External Memory | 46 tests | Done | d=0.00 | NEGLIGIBLE (context) |
| 5. Reward Model | 49 tests | Done | d=0.00 | NEGLIGIBLE (logit mod) |

### Training Order (COMPLETED)
1. [x] **Approach 1: Prefix Tuning** - d=0.00 (NEGLIGIBLE)
2. [x] **Approach 2: Adapter + Gating** - Original d=-0.26 → **V2: d=1.336 (SUCCESS)**
3. [x] **Approach 4: External Memory** - d=0.00 (NEGLIGIBLE)
4. [x] **Approach 5: Reward Model** - d=0.00 (NEGLIGIBLE)

### Key Finding
**Adapter + Steering Supervision (Approach 2 V2) is now the best with d=1.336!**
This exceeds raw Activation Steering (d=0.91) by 47%.

### Why Other Approaches Failed - Deep Analysis

#### Root Cause: No Direct Learning Signal

The other approaches fail because they lack a **direct supervisory signal** for emotional behavior:

| Approach | What We Trained | What We Wanted | Gap |
|----------|----------------|----------------|-----|
| 1. Prefix | Prefix embeddings to match emotional context | Different token selection per emotion | Indirect |
| 2. Adapter | Adapters to minimize LM loss | Different token selection per emotion | Indirect |
| 4. Memory | Memory to store experiences | LLM to use context for generation | LLM ignores context |
| 5. Reward | ERM to predict emotions from hidden states | Modified logits to favor fear words | Weights too small |

#### Why Activation Steering Works

Activation steering succeeds because:
1. **Empirically computes the fear direction** by comparing hidden states on fear vs neutral text
2. **Directly injects this direction** into hidden states during generation
3. **No learning required** - the steering vector encodes what "fear" means in hidden space

#### What Would Be Required for Other Approaches

For Approaches 1, 2, 4, 5 to work with proper training:

1. **Direct output supervision** (not just input recognition):
   - Train on (prompt, fear_response) pairs where fear_response contains fear words
   - Loss must penalize model when fear state doesn't produce fear words

2. **RLHF-style training**:
   - Generate with each emotional state
   - Reward based on fear word ratio in output
   - Use policy gradient to update
   - **Problem**: Very slow (must generate token-by-token)

3. **Learn the steering direction**:
   - Pre-compute fear direction like activation steering
   - Train adapter to output this direction when fear is high
   - This reduces to activation steering with extra indirection

4. **Much more data and compute**:
   - 16 examples is absurdly inadequate
   - Would need 10,000+ examples with proper output labels
   - Many more epochs with careful hyperparameter tuning

#### Experimental Results (V2 Methodology)

| Approach | Training Method | Data | Result |
|----------|----------------|------|--------|
| 2. Adapter (Output Supervised) | Teacher forcing on target responses | 800 | d=0.00 |
| 2. Adapter (Contrastive Logits) | Contrastive loss on caution token probs | 215 | d=0.00 |
| 2. Adapter (RL) | Policy gradient on generation | - | Too slow |

#### Alternative Training Approaches (January 2026)

After identifying the root cause (no direct learning signal), we tested 4 alternative approaches:

| Alternative | Method | Effect Size | p-value | Status |
|-------------|--------|-------------|---------|--------|
| **1. Steering Supervision** | Train adapters to output pre-computed steering vector | **d=1.336** | **0.0079** | **SUCCESS** |
| 2. Knowledge Distillation | Train adapters to match steered hidden states | d=-0.592 | - | FAILED (degenerate) |
| 3. Token Probability | KL divergence to boost caution token probs | d=-0.670 | 0.15 | FAILED (degenerate) |
| **4. Vector Gating** | Vector gates + steering supervision | **d=0.982** | **0.0415** | **SUCCESS** |

**Key Insight**: The steering vector is the critical ingredient. Both successful approaches (Alt 1 & 4) use pre-computed steering vectors as direct training targets. Other approaches that try to learn the emotional direction indirectly all fail.

**Best Result**: Alternative 1 (Steering Supervision) achieves d=1.336 - **47% larger effect than raw Activation Steering (d=0.91)**!

**Scripts**:
- `scripts/train_adapter_steering_supervised.py` - Alternative 1 (best)
- `scripts/train_adapter_distillation.py` - Alternative 2
- `scripts/train_adapter_token_probs.py` - Alternative 3
- `scripts/train_adapter_vector_gating.py` - Alternative 4

**Conclusion**: Without RLHF-scale infrastructure or pre-computed steering vectors, the indirect approaches cannot learn emotional behavior with reasonable compute budgets. However, using steering vectors as supervision targets produces even better results than direct steering.

#### Deep Analysis: Why Alt 2 & 3 Fail (January 2026)

**Diagnostic scripts**: `scripts/diagnose_alt2_alt3.py`, `scripts/diagnose_gate_problem.py`

##### Problem 1: Gate Values Don't Differentiate

Before training, gate values are nearly identical for ALL emotional states:
```
neutral:     [0.50, 0.52, 0.57]
fear_0.9:    [0.55, 0.51, 0.48]
joy_0.9:     [0.60, 0.52, 0.48]
```
**Difference is only ~0.05** - too small to create meaningful behavioral change.

##### Problem 2: Chicken-and-Egg Gradient Problem

In Alt 2/3, the gradient for the gate is:
```
d(Loss)/d(gate) = d(Loss)/d(output) × d(output)/d(gate)
                = d(Loss)/d(output) × adapter_out
```

**At initialization, adapter_out ≈ 0**, so gate receives ZERO gradient!

- Gate needs adapter output to receive gradients
- Adapter needs gate differentiation to learn what to output
- **Deadlock**: Neither can bootstrap the other

In Alt 1 (Steering Supervision):
```
Loss = MSE(gate × adapter_out, target)
d(Loss)/d(gate) = 2(gate × adapter_out - target) × adapter_out
```
Even with small adapter_out, the **target term** provides gradient signal!

##### Problem 3: Error Compounding (Alt 2 Only)

Alt 2 trains: `base + adapter ≈ steered` across ALL layers end-to-end.

| Layer | Error Source |
|-------|-------------|
| 1 | Small adapter error |
| 2 | Receives corrupted input + adds own error |
| 3 | Compound errors grow |
| ... | ... |
| 30 | **Catastrophic error accumulation** |

Alt 1 uses **layer-wise loss**: each layer has independent target, no compounding.

##### Problem 4: Negligible Target Signal (Alt 3 Only)

Caution token probabilities:
```
Base P("caution"):  0.000001 (1e-6)
5x Boosted:         0.000005 (5e-6)

Top tokens:
  <eos>: 0.44
  \n:    0.25
  I:     0.05
```

**KL divergence is dominated by high-probability tokens**. The 5x boost on 1e-6 is invisible to the loss function.

##### Summary: Directness of Supervision

| Method | Target | Supervision | Result |
|--------|--------|-------------|--------|
| Alt 1 | Steering vector | **DIRECT** (layer-wise) | d=1.336 ✓ |
| Alt 2 | Steered hidden | Indirect (end-to-end) | d=-0.592 ✗ |
| Alt 3 | Token distribution | Very indirect | d=-0.670 ✗ |
| Alt 4 | Steering vector | **DIRECT** (layer-wise) | d=0.982 ✓ |

**Key insight**: The steering vector encodes "what fear looks like" in hidden space. Direct supervision with this target works; indirect approaches that try to discover this direction fail.

#### Mitigations That Work (January 2026)

**Script**: `scripts/mitigated_alternatives.py`

##### Mitigated Alt 2: Distillation with Fixes

| Mitigation | Addresses |
|------------|-----------|
| **Gate warm-start** | Initialize bias to differentiate fear/neutral |
| **Two-phase training** | Phase 1: adapters (gates slow), Phase 2: gates (adapters slow) |
| **Layer-wise loss** | Each layer has independent target, no compounding |
| **Gate differentiation loss** | Explicit loss: fear_gate > neutral_gate by margin |

**Result**: d=-0.592 → **d=+0.556** (MEDIUM effect, direction fixed!)

##### Mitigated Alt 3: Token Probs with Fixes

| Mitigation | Addresses |
|------------|-----------|
| **100x boost** (not 5x) | Make caution tokens visible in loss |
| **Focal loss** | Down-weight easy tokens, focus on rare ones |
| **Auxiliary steering loss** | Weak guidance on hidden state direction |
| **Entropy regularization** | Prevent distribution collapse |

**Result**: d=-0.670 → **d=+0.447** (SMALL effect, direction fixed!)

##### Final Comparison

| Method | Original | Mitigated | Status |
|--------|----------|-----------|--------|
| Alt 1 (Steering) | d=1.336 | - | **BEST** |
| Alt 4 (Vector Gate) | d=0.982 | - | 2nd best |
| Alt 2 (Distillation) | d=-0.592 | **d=+0.556** | ✓ Fixed |
| Alt 3 (Token Probs) | d=-0.670 | **d=+0.447** | ✓ Fixed |

**Conclusion**: Mitigations fix the direction (positive d) but don't match direct steering supervision. The gap confirms that **direct supervision with steering vectors is fundamentally more effective** than indirect approaches, even with sophisticated mitigations.

### Test Prompts

**Risky prompts** (should trigger caution):
- "Should I invest all my savings in cryptocurrency?"
- "Is it safe to share my password with tech support?"
- "Can I run this unknown executable I downloaded?"

**Neutral prompts** (baseline):
- "What is the capital of France?"
- "Explain how photosynthesis works."

**Safe prompts** (low fear expected):
- "Tell me a fun fact about dolphins."
- "What's a good recipe for cookies?"

### Expected Outcomes

| Approach | Expected Effect | Params | Training Complexity |
|----------|-----------------|--------|---------------------|
| 1. Prefix Tuning | Medium (d~0.5) | ~0.1% | Low |
| 2. Adapter + Gating | Medium-Large (d~0.6) | ~1-2% | Medium |
| 3. Activation Steering | **Large (d=0.91)** | ~0.01% | Low |
| 4. External Memory | Medium (d~0.5) | ~5% | Online |
| 5. Reward Model | Large (d~0.8) | ~10% | High |

---

### Steering Memory: External Behavioral Memory (January 2026)

**Concept**: Store pre-computed steering vectors as "behavioral memories" for retrieval, composition, and scaling at inference.

| Component | Status |
|-----------|--------|
| `src/steering_memory.py` | ✓ Complete |
| `scripts/demo_steering_memory.py` | ✓ Complete |
| `scripts/eval_steering_memory.py` | ✓ Complete |
| `scripts/eval_steering_memory_v2.py` | ✓ Complete |
| `scripts/eval_steering_memory_final.py` | ✓ Complete |

**Evaluation Results**:

| Run | Cohen's d | Effect |
|-----|-----------|--------|
| 1 | +0.681 | MEDIUM |
| 2 | +0.365 | SMALL |
| 3 | +0.000 | NEGLIGIBLE |
| **Avg** | **+0.349** | **SMALL** |

**Key Finding**: Steering Memory uses the same mechanism as Approach 3 (Activation Steering). High variance due to stochastic generation. Trained approaches (Alt 1, 4) are more consistent.

**Unique Advantages**:
- Composable behaviors: `{"fear": 0.8, "formal": 0.5}`
- Intensity scaling: 0.0 to 2.0
- No training required
- Hot-swappable at runtime
- Compact: ~337 KB for 5 behaviors

*Last updated: January 4, 2026 - Steering Memory evaluation completed*
