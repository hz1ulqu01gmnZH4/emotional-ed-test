# Mathematical Grounding for Emotional Error Diffusion

*Phase 1: Neuroscience Foundations and Risk-Sensitive RL*

---

## 1. Neuromodulator-to-Emotion Mapping

Based on computational neuroscience literature, we establish principled mappings between neuromodulatory systems and emotional channels.

### 1.1 The Four Major Neuromodulators

| Neuromodulator | Abbreviation | Primary Function | ED Channel Mapping |
|----------------|--------------|------------------|-------------------|
| **Dopamine** | DA | Reward prediction error, incentive salience | Wanting, Approach motivation |
| **Norepinephrine** | NA/NE | Unexpected uncertainty, arousal | Fear, Anxiety, Alertness |
| **Serotonin** | 5-HT | Patience, temporal discounting, aversive processing | Mood regulation, Frustration tolerance |
| **Acetylcholine** | ACh | Expected uncertainty, attention | Learning rate modulation |

### 1.2 Mathematical Formulations from Neuroscience

#### 1.2.1 Yu & Dayan (2005): Uncertainty and Neuromodulation

**Source**: [Uncertainty, Neuromodulation, and Attention](https://pubmed.ncbi.nlm.nih.gov/15944135/)

Two types of uncertainty are distinguished:

**Expected Uncertainty (Acetylcholine)**:
```
ACh ∝ H[P(s'|s,a)]  # Entropy of known transition uncertainty
```
- Signals unreliability of predictive cues within a known context
- Modulates attention and learning rate for expected variability

**Unexpected Uncertainty (Norepinephrine)**:
```
NE ∝ D_KL[P(o|context_new) || P(o|context_old)]
```
- Signals context switches producing strongly unexpected observations
- Triggers "interrupt" for belief updating

**Application to ED**:
```python
class UncertaintyModulation:
    def compute_ach(self, expected_variance):
        """Expected uncertainty → learning rate modulation"""
        return self.ach_gain * np.sqrt(expected_variance)

    def compute_ne(self, prediction_error, threshold):
        """Unexpected uncertainty → fear/alertness"""
        surprise = abs(prediction_error)
        if surprise > threshold:
            return self.ne_gain * (surprise - threshold)
        return 0.0
```

#### 1.2.2 Serotonin and Temporal Discounting

**Sources**:
- [Low-Serotonin Levels Increase Delayed Reward Discounting](https://www.jneurosci.org/content/28/17/4528)
- [Serotonin neurons and patience](https://pmc.ncbi.nlm.nih.gov/articles/PMC5984631/)

**Key Finding**: Serotonin modulates the discount factor γ in TD learning.

**Standard TD**:
```
V(s) = E[r + γ V(s')]
```

**Serotonin-Modulated TD**:
```
γ_effective = γ_base + k_5HT × [5-HT]

Where:
- γ_base: Baseline discount factor
- k_5HT: Serotonin sensitivity parameter
- [5-HT]: Current serotonin level
```

**Low serotonin** → Lower γ → More impulsive (discount future heavily)
**High serotonin** → Higher γ → More patient (value future rewards)

**Application to ED (Frustration Tolerance)**:
```python
class SerotoninModule:
    def __init__(self, gamma_base=0.95, k_5ht=0.05):
        self.gamma_base = gamma_base
        self.k_5ht = k_5ht
        self.serotonin_level = 1.0  # Normalized

    def effective_gamma(self):
        """Patience/temporal discounting modulated by serotonin"""
        return min(0.99, self.gamma_base + self.k_5ht * self.serotonin_level)

    def update_serotonin(self, reward_history):
        """Serotonin builds with consistent rewards, depletes with frustration"""
        recent_rewards = np.mean(reward_history[-10:])
        if recent_rewards > 0:
            self.serotonin_level = min(2.0, self.serotonin_level + 0.1)
        else:
            self.serotonin_level = max(0.1, self.serotonin_level - 0.1)
```

#### 1.2.3 Dopamine: Wanting vs Liking (Berridge)

**Sources**:
- [Incentive Salience Theory](https://sites.lsa.umich.edu/berridge-lab/)
- [Neural Computational Model of Incentive Salience](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000437)

**Key Distinction**:
- **Wanting** (Dopamine, mesolimbic): Incentive salience, motivational pull
- **Liking** (Opioids, nucleus accumbens): Hedonic impact, pleasure

**Mathematical Formulation**:
```
Reward_total = Wanting × Liking

Where:
- Wanting = DA_baseline × Cue_salience × Sensitization_factor
- Liking = Opioid_baseline × Consumption_pleasure × Tolerance_factor
```

**Addiction Dynamics (Incentive Sensitization)**:
```
Sensitization(t+1) = Sensitization(t) + k_sens × Usage(t)  # Wanting increases
Tolerance(t+1) = Tolerance(t) × decay + k_tol × Usage(t)   # Liking decreases

Over time: Wanting ↑↑ while Liking ↓↓
```

**Application to ED**:
```python
class WantingLikingModule:
    def __init__(self):
        self.wanting_baseline = 1.0
        self.liking_baseline = 1.0
        self.sensitization = 1.0
        self.tolerance = 1.0

    def compute_reward_value(self, cue_salience, consumption_pleasure):
        wanting = self.wanting_baseline * cue_salience * self.sensitization
        liking = self.liking_baseline * consumption_pleasure * self.tolerance
        return wanting, liking, wanting * liking

    def update_after_consumption(self, dose):
        """Addiction dynamics: sensitization and tolerance"""
        self.sensitization *= (1 + 0.05 * dose)  # Wanting sensitizes
        self.tolerance *= (1 - 0.03 * dose)       # Liking tolerates
```

---

## 2. Risk-Sensitive Reinforcement Learning

### 2.1 CVaR (Conditional Value-at-Risk)

**Sources**:
- [Near-Minimax-Optimal CVaR RL](https://arxiv.org/abs/2302.03201)
- [CVaR Algorithms GitHub](https://github.com/Silvicek/cvar-algorithms)

**Definition**: CVaR_τ is the expected value of the worst τ-fraction of outcomes.

```
CVaR_τ(X) = E[X | X ≤ VaR_τ(X)]

Where VaR_τ(X) = inf{x : P(X ≤ x) ≥ τ}
```

**Risk-Sensitive Bellman Equation**:
```
Q_τ(s,a) = r(s,a) + γ × CVaR_τ[V(s')]
```

- τ → 0: Extremely risk-averse (focus on worst outcomes)
- τ = 0.5: Median-focused
- τ → 1: Risk-neutral (standard expectation)

### 2.2 Fear as Dynamic Risk Aversion

**Proposed Formulation**:

Fear dynamically adjusts the risk tolerance parameter τ:

```
τ_effective(s) = τ_base × (1 - fear(s))

Where:
- τ_base ∈ (0, 1]: Baseline risk tolerance
- fear(s) ∈ [0, 1]: Current fear level
- τ_effective → 0 as fear → 1 (extreme risk aversion)
```

**CVaR Q-Learning Update with Fear**:
```python
class CVaRFearAgent:
    def __init__(self, tau_base=0.5, fear_sensitivity=0.8):
        self.tau_base = tau_base
        self.fear_sensitivity = fear_sensitivity
        self.Q_distribution = {}  # Store return distributions

    def compute_tau(self, fear_level):
        """Fear reduces risk tolerance"""
        return max(0.01, self.tau_base * (1 - self.fear_sensitivity * fear_level))

    def cvar_backup(self, next_state, tau):
        """CVaR Bellman backup"""
        Q_values = self.Q_distribution[next_state]
        sorted_Q = np.sort(Q_values)
        k = max(1, int(np.ceil(tau * len(sorted_Q))))
        return np.mean(sorted_Q[:k])  # Average of worst τ-fraction

    def update(self, state, action, reward, next_state, fear_level):
        tau = self.compute_tau(fear_level)
        cvar_next = self.cvar_backup(next_state, tau)
        target = reward + self.gamma * cvar_next
        # Update Q-distribution...
```

### 2.3 Anger as Optimistic Bias

**Proposed Formulation**:

Anger biases toward upper quantiles (optimistic about overcoming obstacles):

```
Q_anger(s,a) = r(s,a) + γ × VaR_{1-τ_anger}[V(s')]

Where τ_anger = base_tau × (1 + anger_level)
```

**Interpretation**: High anger → focus on best possible outcomes → persistence despite current failure.

```python
class OptimisticAngerAgent:
    def __init__(self, tau_base=0.5):
        self.tau_base = tau_base

    def compute_optimistic_tau(self, anger_level):
        """Anger increases optimism (upper quantile focus)"""
        return min(0.99, self.tau_base * (1 + anger_level))

    def optimistic_backup(self, next_state, tau):
        """Upper quantile backup"""
        Q_values = self.Q_distribution[next_state]
        sorted_Q = np.sort(Q_values)[::-1]  # Descending
        k = max(1, int(np.ceil((1-tau) * len(sorted_Q))))
        return np.mean(sorted_Q[:k])  # Average of best (1-τ)-fraction
```

---

## 3. Multi-Timescale Dynamics (Nested Learning)

### 3.1 Timescale Hierarchy

Based on the Nested Learning framework and neuroscience:

| Level | Timescale | Neuromodulator | Emotional Construct | Update Frequency |
|-------|-----------|----------------|---------------------|------------------|
| 0 | Milliseconds | Glutamate/GABA | Phasic reactions | Every step |
| 1 | Seconds | DA, NE | Emotional episodes | Every 10 steps |
| 2 | Minutes | 5-HT, ACh | Mood states | Every episode |
| 3 | Hours/Days | Cortisol, hormones | Temperament | Every 100 episodes |

### 3.2 Mathematical Formulation

**Nested Optimization (from NL paper)**:

Each level optimizes its own objective at its own timescale:

```
Level 0 (Phasic):
  fear_t = argmin_f L_threat(f, context) + λ₀‖f - f_{t-1}‖²

Level 1 (Emotional State):
  emotion_t = argmin_e L_integrate(e, fear_history) + λ₁‖e - e_{t-1}‖²

Level 2 (Mood):
  mood_t = argmin_m L_mood(m, emotion_history) + λ₂‖m - m_{t-1}‖²

Level 3 (Temperament):
  temp_t = argmin_T L_personality(T, mood_history) + λ₃‖T - T_{t-1}‖²
```

**Regularization strength** λ increases with timescale (slower levels change less).

### 3.3 Continuum Memory Implementation

```python
class ContinuumAffect:
    """
    Multi-timescale affect following Nested Learning principles.
    Each timescale is an exponential moving average with different decay.
    """
    def __init__(self, n_timescales=5):
        # Timescales: 1, 10, 100, 1000, 10000 steps
        self.timescales = [10**i for i in range(n_timescales)]
        self.affect_levels = np.zeros(n_timescales)
        # Decay rate = 1 - 1/timescale (larger timescale = slower decay)
        self.decay_rates = [1 - 1/t for t in self.timescales]

    def update(self, emotional_input):
        """Update all timescales with new emotional input"""
        for i in range(len(self.timescales)):
            α = 1 - self.decay_rates[i]  # Learning rate for this timescale
            self.affect_levels[i] = (
                self.decay_rates[i] * self.affect_levels[i] +
                α * emotional_input
            )

    def get_mood(self):
        """Weighted combination emphasizing slower timescales for mood"""
        weights = np.array([1/t for t in self.timescales])
        weights = weights / weights.sum()
        return np.dot(weights, self.affect_levels)

    def get_phasic(self):
        """Fast timescale = phasic emotional response"""
        return self.affect_levels[0]

    def get_tonic(self):
        """Slow timescales = tonic mood"""
        return np.mean(self.affect_levels[2:])
```

---

## 4. Local Surprise Signal (LSS)

### 4.1 Definition (from Nested Learning)

The Local Surprise Signal is the prediction error that triggers emotional responses:

```
LSS_t = ∇_y L(W; x_t) = y_predicted - y_actual
```

**Interpretation**: LSS measures the mismatch between expectation and reality.

### 4.2 Emotional Triggering Based on LSS

```python
class LSSEmotionalTrigger:
    """
    Emotions are triggered by prediction errors (LSS) in specific contexts.
    """
    def __init__(self):
        self.value_predictor = ValueEstimator()
        self.threat_detector = ThreatDetector()

    def compute_lss(self, state, actual_outcome):
        predicted = self.value_predictor.predict(state)
        return actual_outcome - predicted

    def trigger_emotions(self, state, lss, context):
        emotions = {}

        # Negative surprise near threat → Fear (NE activation)
        if lss < 0 and context.threat_nearby:
            emotions['fear'] = abs(lss) * context.threat_proximity
            emotions['ne_level'] = emotions['fear']

        # Negative surprise when blocked → Anger (approach motivation)
        if lss < 0 and context.was_blocked:
            goal_salience = 1.0 / (1.0 + context.goal_distance)
            emotions['anger'] = abs(lss) * goal_salience
            emotions['persistence'] = 1 - np.exp(-emotions['anger'])

        # Positive surprise → Joy (DA activation)
        if lss > 0:
            emotions['joy'] = lss
            emotions['da_level'] = lss

        # Counterfactual regret
        if context.counterfactual_available:
            cf_lss = context.foregone_outcome - actual_outcome
            if cf_lss > 0:  # Could have done better
                emotions['regret'] = cf_lss

        # Large absolute surprise → ACh (attention/learning rate boost)
        if abs(lss) > self.surprise_threshold:
            emotions['ach_level'] = min(1.0, abs(lss) / self.max_surprise)

        return emotions
```

---

## 5. Principled Emotional Module Equations

### 5.1 Fear Module (Risk-Sensitive)

**Replace linear ramp with sigmoid + CVaR**:

```python
class PrincipledFearModule:
    def __init__(self, tau_base=0.5, fear_sensitivity=0.8, threshold=2.0, steepness=2.0):
        self.tau_base = tau_base
        self.fear_sensitivity = fear_sensitivity
        self.threshold = threshold  # Distance at 50% fear
        self.steepness = steepness

    def compute_fear(self, threat_distance):
        """Sigmoid fear response (saturating, not linear)"""
        # Sigmoid: fear high when close, low when far
        fear = 1.0 / (1.0 + np.exp(self.steepness * (threat_distance - self.threshold)))
        return fear

    def compute_risk_tolerance(self, fear):
        """Fear reduces risk tolerance (CVaR parameter)"""
        tau = self.tau_base * (1 - self.fear_sensitivity * fear)
        return max(0.01, tau)

    def risk_distorted_value(self, Q_distribution, fear):
        """CVaR backup with fear-adjusted risk tolerance"""
        tau = self.compute_risk_tolerance(fear)
        sorted_Q = np.sort(Q_distribution)
        k = max(1, int(np.ceil(tau * len(sorted_Q))))
        return np.mean(sorted_Q[:k])
```

### 5.2 Anger Module (Frustration-Persistence)

**Based on Amsel's Frustration Theory**:

```python
class PrincipledAngerModule:
    def __init__(self, buildup_rate=0.3, decay_rate=0.9, persistence_k=2.0):
        self.frustration = 0.0
        self.buildup_rate = buildup_rate
        self.decay_rate = decay_rate
        self.persistence_k = persistence_k

    def compute(self, context):
        """Frustration = goal_expectation × blocking_signal"""
        if context.was_blocked:
            # Frustration increases more when closer to goal (Amsel)
            goal_proximity = 1.0 / (1.0 + context.goal_distance)
            self.frustration = min(1.0, self.frustration +
                                    self.buildup_rate * goal_proximity)
        else:
            self.frustration *= self.decay_rate

        return self.frustration

    def persistence_probability(self):
        """Probability of persisting despite negative feedback"""
        return 1 - np.exp(-self.persistence_k * self.frustration)

    def optimistic_bias(self):
        """Anger biases toward optimistic outcomes"""
        return self.frustration * 0.5  # Add to upper quantile weight
```

### 5.3 Serotonin/Patience Module

```python
class PatienceModule:
    def __init__(self, gamma_base=0.95, k_5ht=0.04, recovery_rate=0.05, depletion_rate=0.1):
        self.gamma_base = gamma_base
        self.k_5ht = k_5ht
        self.serotonin = 1.0  # Normalized level
        self.recovery_rate = recovery_rate
        self.depletion_rate = depletion_rate

    def update(self, reward, was_waiting):
        """Serotonin dynamics: recovers with success, depletes with frustration"""
        if reward > 0:
            # Successful wait → serotonin recovery
            self.serotonin = min(2.0, self.serotonin + self.recovery_rate)
        elif was_waiting and reward <= 0:
            # Waiting without reward → serotonin depletion
            self.serotonin = max(0.1, self.serotonin - self.depletion_rate)

    def effective_gamma(self):
        """Patience (discount factor) modulated by serotonin"""
        gamma = self.gamma_base + self.k_5ht * (self.serotonin - 1.0)
        return np.clip(gamma, 0.5, 0.99)

    def impulsivity(self):
        """Low serotonin → high impulsivity"""
        return max(0, 1.0 - self.serotonin)
```

### 5.4 Wanting/Liking Dissociation Module

```python
class WantingLikingModule:
    def __init__(self, sensitization_rate=0.05, tolerance_rate=0.03):
        self.wanting_baseline = 1.0
        self.liking_baseline = 1.0
        self.sensitization = 1.0  # Wanting multiplier (increases)
        self.tolerance = 1.0      # Liking multiplier (decreases)
        self.sensitization_rate = sensitization_rate
        self.tolerance_rate = tolerance_rate

    def compute_value(self, cue_salience, hedonic_value):
        """Separate wanting (DA) and liking (opioid) computations"""
        wanting = self.wanting_baseline * cue_salience * self.sensitization
        liking = self.liking_baseline * hedonic_value * self.tolerance
        return {
            'wanting': wanting,
            'liking': liking,
            'total': wanting * liking,
            'addiction_index': wanting / max(0.01, liking)  # High = addicted
        }

    def consume(self, dose):
        """Update after consumption (addiction dynamics)"""
        # Sensitization: wanting increases with repeated exposure
        self.sensitization *= (1 + self.sensitization_rate * dose)
        # Tolerance: liking decreases with repeated exposure
        self.tolerance *= (1 - self.tolerance_rate * dose)
        self.tolerance = max(0.01, self.tolerance)

    def abstain(self, duration):
        """Partial recovery during abstinence"""
        recovery = 0.01 * duration
        self.sensitization = max(1.0, self.sensitization - recovery * 0.5)
        self.tolerance = min(1.0, self.tolerance + recovery)
```

---

## 6. Integrated Emotional ED Agent

```python
class PrincipledEmotionalEDAgent:
    """
    Emotional ED agent with mathematically grounded modules.
    """
    def __init__(self, n_states, n_actions, config=None):
        self.Q = np.zeros((n_states, n_actions))
        self.Q_var = np.ones((n_states, n_actions))  # For distributional

        # Principled emotional modules
        self.fear = PrincipledFearModule()
        self.anger = PrincipledAngerModule()
        self.patience = PatienceModule()
        self.wanting_liking = WantingLikingModule()

        # Multi-timescale affect
        self.affect = ContinuumAffect(n_timescales=4)

        # LSS-based triggering
        self.lss_trigger = LSSEmotionalTrigger()

        # Base parameters
        self.lr = 0.1
        self.epsilon = 0.1

    def select_action(self, state, context):
        """Action selection with emotional modulation"""
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.Q[state]))

        # Get emotional state
        fear_level = self.fear.compute_fear(context.threat_distance)

        # Risk-adjusted Q-values
        adjusted_Q = self.Q[state].copy()

        # Fear: reduce value of risky actions (lower quantile focus)
        if fear_level > 0.1:
            risk_penalty = fear_level * self.Q_var[state]
            adjusted_Q -= risk_penalty

        # Anger: boost recently blocked actions (persistence)
        if self.anger.frustration > 0.1:
            persistence = self.anger.persistence_probability()
            if context.last_action is not None:
                adjusted_Q[context.last_action] += persistence * 0.5

        return np.argmax(adjusted_Q)

    def update(self, state, action, reward, next_state, done, context):
        """Update with emotional modulation of learning"""
        # Compute LSS
        predicted_value = self.Q[state, action]
        actual_value = reward + (0 if done else self.patience.effective_gamma() * np.max(self.Q[next_state]))
        lss = actual_value - predicted_value

        # Trigger emotions based on LSS
        emotions = self.lss_trigger.trigger_emotions(state, lss, context)

        # Update emotional modules
        fear_level = self.fear.compute_fear(context.threat_distance)
        anger_level = self.anger.compute(context)
        self.patience.update(reward, context.was_waiting)
        self.affect.update(lss)  # Multi-timescale affect

        # Modulated learning rate
        effective_lr = self.lr

        # Fear increases learning from negative outcomes
        if lss < 0 and fear_level > 0:
            effective_lr *= (1 + fear_level)

        # Anger decreases learning from negative outcomes (persistence)
        if lss < 0 and anger_level > 0:
            effective_lr *= (1 - 0.5 * anger_level)

        # ACh (surprise) increases overall learning rate
        if 'ach_level' in emotions:
            effective_lr *= (1 + emotions['ach_level'])

        # TD update
        self.Q[state, action] += effective_lr * lss

        # Update variance estimate (for distributional)
        self.Q_var[state, action] = 0.9 * self.Q_var[state, action] + 0.1 * lss**2

    def get_emotional_state(self):
        """Return current emotional state for logging"""
        return {
            'fear': self.fear.compute_fear(float('inf')),  # Base fear
            'anger': self.anger.frustration,
            'patience': self.patience.serotonin,
            'mood': self.affect.get_mood(),
            'phasic': self.affect.get_phasic(),
            'tonic': self.affect.get_tonic()
        }
```

---

## 7. Summary: Neuroscience-to-ED Mapping

| Neuroscience Concept | Mathematical Formulation | ED Implementation |
|---------------------|-------------------------|-------------------|
| NE → Unexpected uncertainty | KL divergence from expected | Fear trigger on large LSS |
| ACh → Expected uncertainty | Entropy of transitions | Learning rate modulation |
| DA → Incentive salience | Cue × Sensitization | Wanting computation |
| Opioid → Hedonic impact | Value × Tolerance | Liking computation |
| 5-HT → Temporal discounting | γ modulation | Patience/frustration tolerance |
| CVaR → Risk aversion | Lower quantile average | Fear as τ reduction |
| Upper quantile → Optimism | Upper quantile average | Anger as persistence |
| Multi-timescale | Nested optimization | Continuum affect memory |
| Prediction error | LSS = actual - predicted | Emotional triggering |

---

## References

1. Yu, A.J. & Dayan, P. (2005). [Uncertainty, Neuromodulation, and Attention](https://pubmed.ncbi.nlm.nih.gov/15944135/). Neuron, 46, 681-692.

2. Berridge, K.C. (2007). [The debate over dopamine's role in reward](https://sites.lsa.umich.edu/berridge-lab/). Psychopharmacology.

3. Zhang, J. et al. (2009). [A Neural Computational Model of Incentive Salience](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000437). PLoS Computational Biology.

4. Schweighofer, N. et al. (2008). [Low-Serotonin Levels Increase Delayed Reward Discounting](https://www.jneurosci.org/content/28/17/4528). Journal of Neuroscience.

5. Miyazaki, K. et al. (2018). [Reward probability and timing uncertainty alter serotonin effects on patience](https://pmc.ncbi.nlm.nih.gov/articles/PMC5984631/). Nature Communications.

6. Wang, R. et al. (2023). [Near-Minimax-Optimal Risk-Sensitive RL with CVaR](https://arxiv.org/abs/2302.03201). ICML.

7. Mei, J. et al. (2025). [Multi-neuromodulatory dynamics for adaptive learning](https://arxiv.org/abs/2501.06762). arXiv.

8. Behrouz, A. et al. (2025). Nested Learning: The Illusion of Deep Learning Architectures. NeurIPS.

---

*Document created: December 2025*
*Phase 1.1-1.2 of Implementation Roadmap*
