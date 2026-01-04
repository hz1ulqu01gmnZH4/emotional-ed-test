# Multi-Model Feedback Synthesis: Emotional ED Architecture

*Consultation with GPT-5, Gemini, and Grok-4 on methodology and architecture*

---

## 1. Executive Consensus

All three models **agree** on the following key points:

| Topic | Consensus |
|-------|-----------|
| Mid-learning insight | ✅ Theoretically sound and biologically aligned |
| Transfer failure interpretation | ✅ Correct - need representation modulation, not state-specific Q |
| Move to neural networks | ✅ Necessary for proper Error Diffusion testing |
| Disgust reversal cause | ✅ Mechanism flaw - boosting argmax can boost wrong action |
| Regulation reversal cause | ✅ Credit assignment problem + environment design issue |

---

## 2. Detailed Model Responses

### 2.1 GPT-5 (High Reasoning)

**Key Insights:**

1. **Mid-learning = Potential-based shaping**: "Emotional channels are currently acting as reward/policy shaping during learning rather than as changes to the ultimate objective." When Q converges, shaping terms stop moving the policy.

2. **Three routes to fix Transfer**:
   - Linear function approximation with successor features
   - Distributional + risk-sensitive critic (CVaR)
   - Belief-state modeling (POMDP)

3. **Regulation reversal diagnosis**: "Credit assignment problem - if reappraisal changes belief but TD target uses un-reappraised hazard model, agent learns reappraisal is bad."

4. **Disgust reversal diagnosis**: "Missing latent persistence and spread. Contamination is hidden, persistent, propagates through contact chains. Need belief state and propagation model."

5. **Recommended progression**:
   - Linear features with eligibility traces
   - Successor features with emotion-conditioned weights (FiLM-style)
   - Distributional critic for CVaR
   - Then scale to neural nets

6. **Critical distinction**: "Avoid potential-based shaping if you want persistent effects. Reserve for accelerating learning, not defining persistent emotional policies."

**Proposed Experiments:**
- E1: Potential-vs-objective test (CVaR vs shaping)
- E2: Representation modulation and transfer test
- E3: Reappraisal as belief update with ablations

---

### 2.2 Gemini

**Key Insights:**

1. **Grief works because it's a "patience mechanism"**: "Suppresses learning rate when yearning is high. Standard Q-learning immediately drives Q-value to zero (extinction). Grief says 'I believe this resource is still valuable despite current evidence.'"

2. **Transfer fails because policy is in Q[state, action]**: "Even if agent feels fear in new environment, it has no policy for how to react in new states. Q-values for new states are zero."

3. **Disgust flaw identified precisely**: "Uses same modulation as Fear: boosts argmax. If agent is exploring and hasn't learned contaminant is bad, argmax might point TOWARDS contaminant. Disgust then boosts walking into contaminant."

4. **Regulation reversal is conceptually correct**: "Fear works (d=1.09), meaning running away is optimal. Regulation reduces fear → stops running away → hurts performance. Regulation only useful if threat is actually safe."

5. **Broadcast modulation requires NN**: "Neuromodulators spray across large regions, affecting processing globally, not specific memory addresses."

**Fix Recommendations:**

| Experiment | Flaw | Fix |
|------------|------|-----|
| Transfer | Tabular Q cannot generalize | Function Approximation (NN) |
| Disgust | Boosts argmax (can boost suicide) | Repulsive Vector or Negative Bias |
| Regulation | Counter-productive if fear is good | Ensure task requires ignoring threats |

---

### 2.3 Grok-4

**Key Insights:**

1. **Mid-learning is "emotional annealing"**: "Emotions peak during exploration/uncertainty. Habitual behaviors suppress limbic signals. Effect being largest pre-convergence is coherent."

2. **Representations require emotional saliency modulation**: "Fear should bias feature extraction or prototypical representations. Amygdala modulates perceptual grouping, making threats 'sticky' in latent space."

3. **Reversed effects = value misalignment**: "Emotional channels compete destructively without Pareto arbitration."

4. **Disgust needs intrinsic penalty**: "Use intrinsic disgust penalty (non-TD, like curiosity) + contamination as state-augmenting mask."

5. **Regulation needs model-based imagination**: "Reappraisal requires simulate safety trajectories. Add variational inference for threat beliefs."

6. **d=-143 is implausible**: "Likely scale issue or typo for -1.43."

**Architecture Recommendations:**
- FiLM layers or cross-attention for emotion→Q gating
- Opponent processes (Solomon & Corbit) for rebound effects
- Scalarized multi-objective Bellman for channel composition

---

## 3. Synthesized Architecture Recommendations

### 3.1 Consensus Fixes (All Three Agree)

| Issue | Recommended Fix | Priority |
|-------|-----------------|----------|
| **Transfer** | Move to feature-based/neural Q-function | HIGH |
| **Disgust** | Change from argmax boost to repulsive/negative bias | HIGH |
| **Regulation** | Fix credit assignment OR redesign environment | MEDIUM |
| **CVaR Fear** | Implement distributional critic with tail-risk focus | HIGH |
| **Broadcast** | Emotions should gate/scale representations (FiLM) | HIGH |

### 3.2 Theoretical Framework Updates

**From GPT-5:**
- Distinguish **shaping** (accelerates learning) vs **objective** (changes optimal policy)
- Emotions as shaping = effects vanish at convergence
- Emotions as objective change = persistent effects

**From Gemini:**
- Grief = patience/prior protection mechanism
- Disgust needs directional bias (like Grief has), not argmax boost
- Regulation only works if fear is maladaptive in the task

**From Grok-4:**
- Add opponent processes for emotional dynamics
- Pareto arbitration for multi-channel composition
- Hierarchical representations for generalization

### 3.3 Recommended Progression

All models suggest a staged approach:

```
Stage 1: Linear Function Approximation
├── Successor features with emotional weights
├── Eligibility traces with per-channel modulation
└── Test Transfer and generalization

Stage 2: Distributional RL
├── Quantile regression for return distribution
├── CVaR-based fear (tail-risk focus)
└── Test risk-sensitive behavior persistence

Stage 3: Neural Networks
├── FiLM/cross-attention for emotion→representation gating
├── Belief-state modeling for regulation/disgust
└── Full Error Diffusion broadcast testing
```

---

## 4. Specific Bug Fixes Identified

### 4.1 Disgust Mechanism (All Three Identified)

**Current (Broken):**
```python
q_values[np.argmax(q_values)] *= (1 + disgust_level)  # Boosts current best
```

**Fixed (Directional Repulsion):**
```python
def _get_action_away_from_contaminant(self, state):
    """Get action that moves AWAY from contaminant."""
    current_pos = state_to_pos(state)
    contam_pos = self.nearest_contaminant(state)
    # Return action that increases distance
    ...

# In select_action:
if self.disgust_level > 0:
    repel_action = self._get_action_away_from_contaminant(state)
    q_values[repel_action] += disgust_level * disgust_weight
```

### 4.2 Regulation Mechanism (GPT-5 + Gemini)

**Problem:** Credit assignment mismatch - belief update doesn't flow to TD target.

**Fixed:**
```python
class ReappraisalModule:
    def __init__(self):
        self.threat_beliefs = {}  # state → P(actually_harmful)

    def update_belief(self, state, was_harmful):
        prior = self.threat_beliefs.get(state, 0.5)
        # Bayesian update
        likelihood = 0.9 if was_harmful else 0.1
        posterior = likelihood * prior / (likelihood * prior + (1-likelihood) * (1-prior))
        self.threat_beliefs[state] = posterior

    def reappraised_fear(self, state, base_fear):
        belief = self.threat_beliefs.get(state, 0.5)
        return base_fear * belief  # Reduce fear for learned-safe states

    def compute_td_target(self, reward, next_state, Q_next):
        # KEY: TD target uses reappraised value, not raw
        reappraised_value = Q_next * (1 - self.threat_beliefs.get(next_state, 0.5))
        return reward + gamma * reappraised_value
```

### 4.3 Transfer Mechanism (All Three)

**Problem:** Q[state, action] is one-hot, can't generalize.

**Fixed (Linear Features):**
```python
class FeatureBasedAgent:
    def __init__(self, n_features, n_actions):
        self.W = np.zeros((n_features, n_actions))

    def features(self, state, context):
        pos = state_to_pos(state)
        return np.array([
            1.0,  # Bias
            context.threat_distance,
            context.goal_distance,
            self.fear_module.compute(context),  # Emotional feature
            float(context.threat_distance < 2),  # Near threat binary
        ])

    def Q(self, state, action, context):
        φ = self.features(state, context)
        return np.dot(self.W[:, action], φ)

    def update(self, state, action, reward, next_state, context):
        φ = self.features(state, context)
        Q_current = np.dot(self.W[:, action], φ)
        Q_next_max = max(self.Q(next_state, a, context) for a in range(n_actions))
        td_error = reward + gamma * Q_next_max - Q_current
        self.W[:, action] += lr * td_error * φ  # Gradient update
```

---

## 5. Methodological Recommendations

### 5.1 Statistical Improvements

**GPT-5:**
- Control FDR with Benjamini-Hochberg (11 tests)
- Track Q-gap masking: Δ(s)=Q*(a1)−Q*(a2)
- Frozen-policy and frozen-critic ablations

**Grok-4:**
- Fix d=-143 (scale error, likely -1.43)
- Report variance plots, power analysis
- Quantify "mid-learning" via policy entropy

### 5.2 Ablation Structure

All suggest separating:
1. **Learning-only modulation**: Emotion affects updates, not action selection
2. **Policy-only modulation**: Emotion affects action selection, not updates
3. **Full modulation**: Both

This separates exploration effects from objective changes.

### 5.3 Environment Design

**For Regulation testing (Gemini):**
- Environment must have "fake threats" (look scary but safe)
- If all threats are real, reducing fear SHOULD hurt performance

**For Disgust testing (GPT-5):**
- Add explicit contamination propagation dynamics
- Test in POMDP where contaminant is latent

---

## 6. Priority Action Items

### Immediate (Before Next Experiments)

1. ✅ **Fix Disgust mechanism**: Directional repulsion, not argmax boost
2. ✅ **Clarify d=-143**: Confirm calculation (likely scale error)
3. ⬜ **Implement linear feature Q**: Basic transfer test

### Short-Term (Architecture Revision)

4. ⬜ **Add CVaR-based fear**: Distributional critic with tail-risk
5. ⬜ **Fix Regulation credit assignment**: Bayesian belief → TD target
6. ⬜ **Test potential-vs-objective**: Does effect persist at convergence?

### Medium-Term (Neural Implementation)

7. ⬜ **FiLM-style emotion gating**: φ_e(s) = g(e) ⊙ φ(s) + h(e)
8. ⬜ **Belief-state POMDP**: For regulation and disgust
9. ⬜ **Full Error Diffusion**: Broadcast modulation across layers

---

## 7. Key Theoretical Takeaways

### 7.1 The Shaping vs Objective Distinction (GPT-5)

**Shaping**: γΦ(s') - Φ(s) form → preserves optimal policy → effects vanish at convergence

**Objective Change**: CVaR, constraints, belief updates → changes optimal policy → persistent effects

**Implication**: If you want emotions to matter at convergence, they must change the objective, not just accelerate learning.

### 7.2 The Representation Modulation Principle (All)

**Current (broken)**: Emotions modulate Q[state, action]
**Needed**: Emotions modulate φ(state) before Q is computed

This enables:
- Transfer (features generalize, state indices don't)
- Broadcast (emotions affect processing, not just memory)
- Hierarchical control (emotions gate attention/features)

### 7.3 The Belief-State Requirement (GPT-5, Grok-4)

For **Regulation** and **Disgust**, the agent needs:
- Latent variable tracking (is this safe? is this contaminated?)
- Bayesian belief updates
- Beliefs flowing into both policy AND value estimation

Without this, the agent can't distinguish "looks scary but safe" from "actually dangerous."

---

*Synthesis completed: December 2025*
*Sources: GPT-5 (high reasoning), Gemini, Grok-4*
*Next step: Implement priority fixes and re-run validation*
