# Emotional ED Architecture V2: Implementation Results

*Completed: December 2025*

---

## Executive Summary

Following multi-model consultation (GPT-5, Gemini, Grok-4) and deep analysis of 11 experiments, we implemented architectural fixes for the three failing experiments (Transfer, Regulation, Disgust) plus a principled risk-sensitivity module (CVaR Fear).

**All validation tests pass (6/6).**

---

## 1. Problem Diagnosis

### Original Failures

| Experiment | Effect Size | Issue |
|------------|-------------|-------|
| Transfer | d=0.12 | Tabular Q cannot generalize; fear in new states has no policy |
| Regulation | d=-0.36 (reversed) | Credit assignment failure + all threats were real |
| Disgust | d=0.25 (reversed) | Argmax boost can boost action TOWARD contaminant |

### Root Causes Identified

1. **Tabular Q is one-hot**: `Q[state, action]` cannot generalize across states
2. **Argmax boost is directionless**: During exploration, best action may be wrong direction
3. **Belief update disconnected from TD**: Reappraisal didn't flow to value estimation
4. **Environment design**: If all threats are real, reducing fear *should* hurt

---

## 2. Implemented Fixes

### 2.1 Disgust V2: Directional Repulsion

**File**: `agents_v2/agents_disgust_v2.py`

**Before (broken)**:
```python
q_values[np.argmax(q_values)] *= (1 + disgust_level)  # Boosts current best
```

**After (fixed)**:
```python
# Boost action AWAY from contaminant
away_action = self._get_action_away_from_contaminant(state)
q_values[away_action] += disgust_level * weight * 0.5

# Penalize action TOWARD contaminant
toward_action = self._get_action_toward_contaminant(state)
q_values[toward_action] -= disgust_level * weight * 0.3
```

**Test Result**: Away actions selected 1000/1000 times vs 0/1000 toward when disgust is high.

---

### 2.2 Feature-Based Q: Transfer Generalization

**File**: `agents_v2/agents_feature_based.py`

**Key Innovation**: Q-value computed from position-invariant features:

```python
def _compute_features(self, state_pos, context, action) -> np.ndarray:
    return np.array([
        1.0,                                          # Bias
        1.0 / (1.0 + context.threat_distance),        # Threat proximity
        1.0 / (1.0 + context.goal_distance),          # Goal proximity
        toward_threat,                                # Action-threat alignment
        toward_goal,                                  # Action-goal alignment
        1.0 if context.near_threat else 0.0,          # Near threat binary
        self.fear_level,                              # Emotional feature
        self.fear_level * max(0, toward_threat)       # Fear × approach interaction
    ])

def Q(self, state_pos, context, action) -> float:
    φ = self._compute_features(state_pos, context, action)
    return np.dot(self.W[:, action], φ)  # Linear function approximation
```

**Test Result**: Q-value of -0.5 for "toward threat" action generalizes to completely new state positions.

---

### 2.3 Regulation V2: Bayesian Reappraisal

**File**: `agents_v2/agents_regulation_v2.py`

**Key Components**:

1. **Bayesian Belief Update**:
```python
def update_belief(self, threat_type: str, was_harmed: bool):
    # Beta-binomial posterior
    safe_count = self.safe_experiences[threat_type]
    dangerous_count = self.dangerous_experiences[threat_type]
    alpha, beta = 1.0, 2.0  # Weak prior favoring dangerous
    posterior = (safe_count + alpha) / (total + alpha + beta)
    self.safety_beliefs[threat_type] = posterior
```

2. **Credit Assignment Fix** - Belief flows to TD target:
```python
def update(self, state, action, reward, next_state, done, context):
    # KEY FIX: TD target uses reappraised value
    safety = self.reappraisal.get_safety_belief(context.threat_type)
    reappraised_next_value = next_q_max * (1 + safety * 0.2)
    target = reward + gamma * reappraised_next_value
```

3. **Environment with Fake Threats**:
```python
class RegulationGridWorldV2:
    """
    - Real threat (X): Actually dangerous, avoid
    - Fake threat (F): Looks scary but safe, gives bonus
    """
    # Fake threat gives +0.4 bonus when approached
    if fake_dist < 1.0 and not self.fake_bonus_collected:
        reward += 0.4
```

**Test Results**:
- P(safe) for fake threat: 0.30 → 0.85 after 10 safe experiences
- P(safe) for real threat: 0.30 → 0.13 after 5 harmful experiences
- Reappraised fear: fake=0.12, real=0.70 (same base fear of 0.8)

---

### 2.4 CVaR Fear: Principled Risk-Sensitivity

**File**: `agents_v2/agents_cvar_fear.py`

**Key Innovation**: Fear controls CVaR alpha (risk level), not just reward shaping:

```python
# CVaR = E[Z | Z ≤ VaR_α(Z)] = mean of worst α fraction

def _fear_to_alpha(self, fear: float) -> float:
    """High fear → Low alpha → More risk-averse"""
    return self.base_alpha - fear * (self.base_alpha - self.min_alpha)

def compute_cvar(self, quantiles: np.ndarray, alpha: float) -> float:
    cutoff_idx = max(1, int(alpha * self.n_quantiles))
    worst_quantiles = quantiles[:cutoff_idx]
    return np.mean(worst_quantiles)
```

**Theoretical Advantage** (from GPT-5):
- Shaping: γΦ(s') - Φ(s) → effects vanish at convergence
- Objective (CVaR): Changes optimal policy → persistent effects

**Test Results**:
- Risk-neutral agent: Chooses high expected value action (ignores variance)
- CVaR agent with fear: Chooses low-variance action (better worst-case)
- Alpha modulation: Fear 0→0.5, Fear 0.5→0.3, Fear 1.0→0.1

---

## 3. Validation Test Summary

```
============================================================
VALIDATION TESTS FOR V2 AGENTS
============================================================

=== Test: Disgust Directional Repulsion ===
✓ Directional calculations correct
  Away (left): 1000, Toward (right): 0
✓ Action selection prefers away direction
PASSED

=== Test: Feature-Based Generalization ===
  Q-values in new state: [0.0, 0.0, 0.0, -0.5]
✓ Feature-based Q generalizes threat avoidance to new state
PASSED

=== Test: Bayesian Reappraisal ===
  Initial belief P(safe|fake_threat): 0.300
  After 10 safe experiences: 0.846
  After 5 harmful experiences (real): 0.125
  Reappraised fear: fake=0.123, real=0.700
PASSED

=== Test: Regulation Environment ===
  Fake threat correctly identified: threat_type='fake'
PASSED

=== Test: CVaR Risk Sensitivity ===
  E[action 0] = 0.50, E[action 1] = 0.50
  CVaR_0.3[action 0] = -1.38, CVaR_0.3[action 1] = 0.35
  Risk-neutral: action 0, CVaR with fear: action 1
PASSED

=== Test: Fear → Alpha Modulation ===
  Fear=0.0 → alpha=0.50
  Fear=1.0 → alpha=0.10
PASSED

============================================================
RESULTS: 6 passed, 0 failed
============================================================
```

---

## 4. File Structure

```
agents_v2/
├── __init__.py              # Module exports
├── agents_disgust_v2.py     # Directional repulsion fix
├── agents_feature_based.py  # Feature-based Q for Transfer
├── agents_regulation_v2.py  # Bayesian reappraisal fix
└── agents_cvar_fear.py      # CVaR distributional RL

tests/
└── test_agents_v2.py        # Validation tests (6/6 pass)
```

---

## 5. Theoretical Framework Updates

### 5.1 Shaping vs Objective Distinction (GPT-5)

| Type | Definition | Effect at Convergence |
|------|------------|----------------------|
| Shaping | γΦ(s') - Φ(s) form | Vanishes (optimal policy unchanged) |
| Objective | CVaR, constraints, beliefs | Persists (optimal policy changed) |

**Implication**: If emotions should matter at convergence, they must change the objective.

### 5.2 Representation Modulation Principle (All Models)

| Approach | Mechanism | Generalization |
|----------|-----------|----------------|
| Current (broken) | Emotions modulate Q[state, action] | ❌ State-specific |
| Fixed (v2) | Emotions modulate φ(state) before Q | ✓ Feature-based |

### 5.3 Belief-State Requirement (GPT-5, Grok-4)

For Regulation and Disgust:
- Agent needs latent variable tracking (is this safe? is this contaminated?)
- Bayesian belief updates required
- Beliefs must flow into BOTH policy AND value estimation

---

## 6. Next Steps

### Immediate: Run Full Experiments
1. Re-run Transfer experiment with `FeatureBasedTransferAgent`
2. Re-run Regulation experiment with `RegulatedFearAgentV2` + `RegulationGridWorldV2`
3. Re-run Disgust experiment with `DisgustOnlyAgentV2`
4. Add CVaR Fear experiment for comparison

### Medium-Term: Neural Network Implementation
1. DQN with emotion as input feature
2. FiLM layers for emotion→representation gating: `φ_e(s) = g(e) ⊙ φ(s) + h(e)`
3. Distributional DQN with CVaR objective

### Long-Term: Full Error Diffusion
1. Broadcast modulation across network layers
2. Opponent processes (Solomon & Corbit) for emotional dynamics
3. Multi-objective Bellman with Pareto arbitration

---

## 7. Model Feedback Summary

### Consensus Points (GPT-5, Gemini, Grok-4)

| Topic | Agreement |
|-------|-----------|
| Mid-learning insight | ✓ Theoretically sound, biologically aligned |
| Transfer failure | ✓ Need representation modulation, not state-specific Q |
| Move to neural networks | ✓ Necessary for proper Error Diffusion testing |
| Disgust reversal | ✓ Mechanism flaw - argmax boost can boost wrong action |
| Regulation reversal | ✓ Credit assignment + environment design |

### Key Quotes

**Gemini on Grief**:
> "Grief acts as a patience mechanism, effectively saying 'I believe this resource is still valuable despite current evidence to the contrary.'"

**GPT-5 on CVaR**:
> "If you want emotions to matter at convergence, they must change the objective, not just accelerate learning."

**Grok-4 on Representation**:
> "Fear should bias feature extraction or prototypical representations. Amygdala modulates perceptual grouping, making threats 'sticky' in latent space."

---

*Implementation completed. Ready for experimental validation.*
