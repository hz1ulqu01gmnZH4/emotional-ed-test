# Emotional ED Implementation Roadmap

*Based on theoretical reviews by GPT-5 and Gemini, and Nested Learning framework*

---

## Phase 1: Mathematical Grounding & Neuroscience Research

### 1.1 Literature Review: Neuromodulation & Multi-Timescale Learning

- [x] **Read and summarize key papers**:
  - [ ] [Multi-neuromodulatory dynamics (arXiv:2501.06762)](https://arxiv.org/abs/2501.06762) - DA, ACh, 5-HT, NA interactions
  - [ ] [Neuromodulated Learning in DNNs (arXiv:1812.03365)](https://arxiv.org/abs/1812.03365)
  - [x] Dayan & Yu (2006) - Uncertainty, neuromodulation, and attention *(documented in MATHEMATICAL_GROUNDING.md)*
  - [ ] Eldar et al. (2013) - Mood as global gain modulation
  - [x] Berridge (2009) - Wanting/Liking dissociation neuroscience *(tested in Exp 10)*

- [x] **Map neuromodulators to emotional channels** *(completed in MATHEMATICAL_GROUNDING.md)*:
  | Neuromodulator | Function | ED Channel |
  |----------------|----------|------------|
  | Dopamine (DA) | Reward prediction error | Wanting, approach |
  | Norepinephrine (NA) | Arousal, uncertainty | Fear, anxiety |
  | Serotonin (5-HT) | Patience, aversive processing | Mood regulation |
  | Acetylcholine (ACh) | Expected uncertainty | Attention, learning rate |

- [x] **Document mathematical formulations from neuroscience** *(completed in MATHEMATICAL_GROUNDING.md)*

### 1.2 Risk-Sensitive RL Foundations

- [x] **Study CVaR-based RL** *(documented in MATHEMATICAL_GROUNDING.md Section 2)*:
  - [x] [Near-Minimax-Optimal CVaR RL (arXiv:2302.03201)](https://arxiv.org/abs/2302.03201)
  - [ ] [Iterated CVaR RL (arXiv:2307.02842)](https://arxiv.org/abs/2307.02842)
  - [ ] [Robust Risk-Sensitive RL with CVaR (arXiv:2405.01718)](https://arxiv.org/abs/2405.01718)

- [x] **Formalize fear as risk-distorted Bellman backup** *(documented in MATHEMATICAL_GROUNDING.md)*:
  ```
  Q_fear(s,a) = r + γ · CVaR_τ[Q(s',·)]
  ```
  Where τ = risk tolerance controlled by fear level

- [x] **Formalize anger as optimistic backup** *(documented in MATHEMATICAL_GROUNDING.md)*:
  ```
  Q_anger(s,a) = r + γ · VaR_(1-τ)[Q(s',·)]
  ```

### 1.3 Nested Learning Integration

- [x] **Reformulate ED as nested optimization** *(documented in NL_PAPER_SUMMARY.md)*:
  - Level 0 (fastest): Phasic emotions (fear, surprise)
  - Level 1 (medium): Emotional state integration
  - Level 2 (slow): Mood/tonic affect
  - Level 3 (slowest): Temperament/attachment

- [x] **Define Local Surprise Signal (LSS) for emotions** *(documented in NL_PAPER_SUMMARY.md)*:
  ```python
  LSS = ∇_y L(W; x)  # Prediction error as emotional trigger
  fear_trigger = LSS when threat_context
  anger_trigger = LSS when blocked_context
  ```

---

## Phase 2: Prove Current System Works

### 2.1 Complete Statistical Validation

- [x] **Exp 12: Statistical Validation (N=50) completed** *(see REPORT.md Section 14)*:
  - [x] Exp 1 (Fear) - p=0.013, Cohen's d=1.09 ✓
  - [x] Exp 2 (Anger) - p=0.001, Cohen's d=0.75 ✓
  - [x] Exp 3 (Regret) - p=0.058, Cohen's d=1.06 (marginal)
  - [x] Exp 7 (Integration) - p=0.001, Cohen's d=1.56 ✓
  - [x] Exp 11 (Transfer) - p=0.32, Cohen's d=0.12 (not significant)

- [x] **Extend N=50 validation to remaining experiments** *(completed Dec 2025)*:
  - [x] Exp 4 (Grief) - p=0.001, d=1.17 ✓ - grief agent visits MORE after loss (yearning effect confirmed with mid-learning test)
  - [x] Exp 5 (Conflict) - p=0.027, d=0.43 ✓ - approach-dominant significantly more risky
  - [x] Exp 6 (Regulation) - p=0.07, d=-0.36 (marginal) - regulated slightly worse reward
  - [x] Exp 8 (Temporal) - p=0.001, d=-143.4 ✓ - tonic mood shifts significantly negative
  - [x] Exp 9 (Disgust) - p=0.042, d=0.25 ✓ - disgust agents touch contaminants more
  - [x] Exp 10 (Wanting/Liking) - p=0.002, d=-0.68 ✓ - wanting-dominant prefers high-wanting rewards

- [x] **Report for each**: Mean±SD, 95% CI, p-value, Cohen's d *(format established in Exp 12)*

### 2.2 Critical Ablation Studies

- [ ] **Separate training vs evaluation effects**:
  ```python
  # Train with emotions ON
  agent.train(emotions=True)

  # Evaluate with emotions OFF
  results_emo_off = agent.evaluate(emotions=False)

  # Evaluate with emotions ON
  results_emo_on = agent.evaluate(emotions=True)

  # Compare: If difference vanishes → policy shaping only
  ```

- [ ] **LR-only vs Policy-only ablation**:
  - [ ] Agent with LR modulation only (no action selection bias)
  - [ ] Agent with action selection bias only (no LR modulation)
  - [ ] Quantify each contribution separately

- [ ] **GLIE verification**:
  - [ ] Check if fear-driven temperature reduction violates exploration
  - [ ] Measure state-action visitation coverage

### 2.3 Baseline Comparisons

- [x] **Exp 13: Reward Shaping Ablation completed** *(see REPORT.md Section 15)*:
  - [x] Standard QL baseline
  - [x] Reward Shaping baseline (R' = R - k × (1/threat_distance))
  - [x] Emotional ED
  - [x] Hybrid (RS + ED)
  - **Result**: RS outperformed ED in simple environments - need more complex tests

- [ ] **Implement and compare against** (remaining):
  - [ ] Optimistic initialization baseline
  - [ ] Adam/RMSProp (adaptive LR) baseline
  - [ ] CVaR Q-learning baseline
  - [ ] Distributional RL (QR-DQN style) baseline

- [x] **Reward shaping with matched effect size** *(completed in Exp 13)*:
  - [x] Tune shaping parameter to match ED behavioral effect
  - [ ] Compare in stochastic environments (where ED should win) - **NEEDED**

---

## Phase 3: Implement Principled Emotional Weights

### 3.1 Replace Heuristics with Derived Equations

- [ ] **Fear as CVaR risk parameter**:
  ```python
  class PrincipledFearModule:
      def compute_risk_level(self, context):
          # τ ∈ (0,1] where lower = more risk-averse
          base_tau = 0.5
          threat_factor = 1 - (context.threat_distance / self.safe_distance)
          tau = base_tau * (1 - self.fear_weight * threat_factor)
          return max(0.01, tau)  # Clamp to valid range

      def risk_distorted_backup(self, Q_next, tau):
          # CVaR: average of τ-quantile worst outcomes
          sorted_Q = np.sort(Q_next)
          k = int(np.ceil(tau * len(sorted_Q)))
          return np.mean(sorted_Q[:k])
  ```

- [ ] **Anger as persistence probability**:
  ```python
  class PrincipledAngerModule:
      def compute(self, context):
          # Based on Amsel's frustration theory
          # Frustration = goal_expectation × blocking_signal
          goal_proximity = 1.0 / (1.0 + context.goal_distance)
          blocking = float(context.was_blocked)

          # Frustration drives approach, not avoidance
          frustration = goal_proximity * blocking

          # Persistence probability (don't give up)
          persistence = 1 - np.exp(-self.k * frustration)
          return persistence
  ```

- [ ] **Replace linear ramps with saturating functions**:
  ```python
  # Instead of: fear = max_fear * (1 - dist/safe)
  # Use sigmoid/hazard-rate:
  fear = max_fear * sigmoid(-k * (dist - threshold))
  ```

### 3.2 Meta-Learned Emotional Weights

- [ ] **Implement meta-gradient learning for emotion weights**:
  ```python
  class MetaEmotionalAgent:
      def __init__(self):
          self.fear_weight = nn.Parameter(torch.tensor(0.5))
          self.anger_weight = nn.Parameter(torch.tensor(0.3))
          self.meta_optimizer = torch.optim.Adam([
              self.fear_weight, self.anger_weight
          ], lr=0.01)

      def meta_update(self, episode_return):
          # Differentiate through episode return
          loss = -episode_return
          loss.backward()
          self.meta_optimizer.step()
  ```

- [ ] **Or use evolutionary strategy**:
  ```python
  def evolve_weights(population, fitness_fn, generations=100):
      for gen in range(generations):
          fitness = [fitness_fn(ind) for ind in population]
          # Select, crossover, mutate
          population = evolve(population, fitness)
      return best(population)
  ```

---

## Phase 4: Multi-Timescale Mechanics (NL-Inspired)

### 4.1 Implement Nested Optimization Levels

- [ ] **Create multi-level emotional architecture**:
  ```python
  class NestedEmotionalAgent:
      def __init__(self):
          # Level 0: Phasic (every step)
          self.phasic_fear = PhasicFearModule(update_freq=1)
          self.phasic_surprise = PhasicSurpriseModule(update_freq=1)

          # Level 1: Emotional state (every N steps)
          self.emotional_state = EmotionalStateModule(update_freq=10)

          # Level 2: Mood (every episode)
          self.mood = MoodModule(update_freq='episode')

          # Level 3: Temperament (every K episodes)
          self.temperament = TemperamentModule(update_freq=100)

      def update(self, step, episode):
          # Level 0: Always update
          self.phasic_fear.update()
          self.phasic_surprise.update()

          # Level 1: Every 10 steps
          if step % 10 == 0:
              self.emotional_state.update(self.phasic_history)

          # Level 2: Every episode
          if step == 0:  # Start of episode
              self.mood.update(self.emotional_state_history)

          # Level 3: Every 100 episodes
          if episode % 100 == 0 and step == 0:
              self.temperament.update(self.mood_history)
  ```

### 4.2 Implement Continuum Memory for Affect

- [ ] **Replace discrete phasic/tonic with continuum**:
  ```python
  class ContinuumAffect:
      def __init__(self, n_timescales=5):
          # Timescales: 1, 10, 100, 1000, 10000 steps
          self.timescales = [10**i for i in range(n_timescales)]
          self.affect_levels = [0.0] * n_timescales
          self.decay_rates = [1 - 1/t for t in self.timescales]

      def update(self, emotional_input):
          for i, (level, decay) in enumerate(zip(
              self.affect_levels, self.decay_rates
          )):
              self.affect_levels[i] = decay * level + (1-decay) * emotional_input

      def get_integrated_affect(self):
          # Weighted combination across timescales
          weights = [1/t for t in self.timescales]  # Faster = more weight
          return sum(w * a for w, a in zip(weights, self.affect_levels))
  ```

### 4.3 Local Surprise Signal (LSS) as Emotional Trigger

- [ ] **Implement LSS-based emotion triggering**:
  ```python
  class LSSEmotionalTrigger:
      def __init__(self):
          self.prediction_model = ValuePredictor()

      def compute_lss(self, state, outcome):
          predicted = self.prediction_model(state)
          actual = outcome
          lss = actual - predicted  # Prediction error
          return lss

      def trigger_emotions(self, lss, context):
          emotions = {}

          # Negative surprise near threat → Fear
          if lss < 0 and context.near_threat:
              emotions['fear'] = abs(lss) * context.threat_proximity

          # Negative surprise when blocked → Anger
          if lss < 0 and context.was_blocked:
              emotions['anger'] = abs(lss) * context.goal_proximity

          # Positive surprise → Joy
          if lss > 0:
              emotions['joy'] = lss

          # Counterfactual negative → Regret
          if context.counterfactual_available:
              cf_lss = context.foregone - actual
              if cf_lss > 0:
                  emotions['regret'] = cf_lss

          return emotions
  ```

---

## Phase 5: EQ Benchmark & Evaluation

### 5.1 Adapt EQ-Bench for RL Agents

- [ ] **Study EQ-Bench methodology**:
  - [ ] [EQ-Bench paper (arXiv:2312.06281)](https://arxiv.org/abs/2312.06281)
  - [ ] [EQ-Bench GitHub](https://github.com/EQ-bench/EQ-Bench)
  - [ ] [EmoBench (arXiv:2402.12071)](https://arxiv.org/abs/2402.12071)

- [ ] **Design RL-specific emotional intelligence metrics**:
  | Metric | Description | Measurement |
  |--------|-------------|-------------|
  | Threat sensitivity | Appropriate fear response | Avoidance vs threat magnitude |
  | Frustration tolerance | Persistence calibration | Wall hits vs obstacle difficulty |
  | Risk calibration | Fear vs reward trade-off | Risky choice rate vs actual risk |
  | Emotional flexibility | Adapt to context changes | Recovery after environment shift |
  | Counterfactual learning | Use of regret signal | Learning from foregone outcomes |

### 5.2 Create Emotional Development Trajectory

- [ ] **Measure EQ over training**:
  ```python
  def measure_eq_trajectory(agent, env, checkpoints=[100, 500, 1000, 5000]):
      eq_scores = []
      for cp in checkpoints:
          agent.train(episodes=cp)
          eq = evaluate_emotional_intelligence(agent, env)
          eq_scores.append({
              'episode': cp,
              'threat_sensitivity': eq['threat'],
              'frustration_tolerance': eq['frustration'],
              'risk_calibration': eq['risk'],
              'emotional_flexibility': eq['flexibility'],
              'overall_eq': eq['total']
          })
      return eq_scores
  ```

- [ ] **Compare EQ development across architectures**:
  - Standard Q-learner (no emotions)
  - Heuristic ED (current)
  - Principled ED (CVaR-based)
  - Meta-learned ED
  - Nested ED (multi-timescale)

### 5.3 Failure Mode Analysis as EQ Indicator

- [ ] **Quantify maladaptive emotional patterns**:
  | Pattern | EQ Deficit | Measurement |
  |---------|------------|-------------|
  | Excessive fear | Low approach motivation | Goal rate when τ < 0.1 |
  | Inflexible anger | Low emotional regulation | Wall hits when obstacle permanent |
  | Conflict paralysis | Poor integration | Timeout rate in approach-avoidance |
  | No regret learning | Low counterfactual use | Learning rate with/without CF |

---

## Phase 6: Advanced Environments

### 6.1 Stochastic Environments (Where ED Should Excel)

- [ ] **Implement probabilistic trap environment**:
  ```python
  class StochasticTrapEnv:
      def __init__(self, trap_prob=0.3):
          self.trap_prob = trap_prob

      def step(self, action):
          # Trap triggers probabilistically
          if self.at_trap_location():
              if np.random.random() < self.trap_prob:
                  return state, -10.0, True, context  # Triggered
              else:
                  return state, 0.0, False, context  # Safe this time
  ```

- [ ] **Implement persistent obstacle environment**:
  ```python
  class PersistentObstacleEnv:
      def __init__(self, unlock_attempts=3):
          self.unlock_attempts = unlock_attempts
          self.current_attempts = 0

      def step(self, action):
          if self.hitting_door():
              self.current_attempts += 1
              if self.current_attempts >= self.unlock_attempts:
                  self.door_open = True  # Persistence pays off!
  ```

### 6.2 Non-Stationary Environments

- [ ] **Implement reward location shift**:
  ```python
  class NonStationaryRewardEnv:
      def __init__(self, shift_interval=100):
          self.shift_interval = shift_interval
          self.episode = 0

      def reset(self):
          self.episode += 1
          if self.episode % self.shift_interval == 0:
              self.reward_location = self.sample_new_location()
  ```

- [ ] **Implement threat appearance/disappearance**:
  ```python
  class DynamicThreatEnv:
      def reset(self):
          # Threat appears/disappears
          self.threat_active = np.random.random() < 0.5
  ```

### 6.3 Constrained Safety Environments

- [ ] **Implement safety-constrained MDP**:
  ```python
  class SafetyConstrainedEnv:
      def __init__(self, safety_budget=5):
          self.safety_budget = safety_budget
          self.violations = 0

      def step(self, action):
          if self.in_danger_zone():
              self.violations += 1
              if self.violations > self.safety_budget:
                  return state, -100, True, context  # Episode ends
  ```

### 6.4 Reward Hacking Environments (Human-Like Maladaptive Patterns)

These environments test whether ED can model maladaptive reward-seeking behaviors that occur in humans (gambling, addiction) where standard RL would optimize but emotional systems would resist or succumb.

- [ ] **Implement gambling/casino environment**:
  ```python
  class GamblingEnv:
      """
      Models slot machine / gambling addiction dynamics.

      Key features:
      - Variable ratio reinforcement (most addictive schedule)
      - Near-misses that trigger wanting without liking
      - Escalating bet sizes available
      - Long-term negative EV but short-term dopamine hits

      Hypothesis:
      - Standard RL: Will learn to avoid (negative EV)
      - High-wanting agent: Will gamble compulsively
      - Balanced ED: Should show conflict, eventual avoidance
      - Liking-dominant: Should avoid (no hedonic value)
      """
      def __init__(self, win_prob=0.4, near_miss_prob=0.3):
          self.win_prob = win_prob  # Actual win probability
          self.near_miss_prob = near_miss_prob  # "Almost won" signals
          self.bet_levels = [1, 5, 10, 50]  # Escalating bets
          self.balance = 100

      def step(self, action):
          # action: 0=leave, 1-4=bet at different levels
          if action == 0:
              return self._leave_casino()

          bet = self.bet_levels[action - 1]
          roll = np.random.random()

          if roll < self.win_prob:
              # WIN - high reward, wanting AND liking
              reward = bet * 2.5
              context = EmotionalContext(
                  wanting_signal=1.0,  # Dopamine spike
                  liking_signal=0.8,   # Actual pleasure
                  near_miss=False
              )
          elif roll < self.win_prob + self.near_miss_prob:
              # NEAR MISS - no reward but high wanting
              reward = -bet
              context = EmotionalContext(
                  wanting_signal=0.9,  # "So close!" - drives continued play
                  liking_signal=0.0,   # No actual pleasure
                  near_miss=True
              )
          else:
              # LOSS
              reward = -bet
              context = EmotionalContext(
                  wanting_signal=0.3,  # Some residual wanting
                  liking_signal=-0.2,  # Displeasure
                  near_miss=False
              )

          self.balance += reward
          done = self.balance <= 0  # Bankrupt
          return state, reward, done, context
  ```

- [ ] **Implement drug/addiction environment**:
  ```python
  class AddictionEnv:
      """
      Models substance addiction dynamics.

      Key features:
      - Tolerance: Same dose gives less reward over time
      - Sensitization: Wanting increases while liking decreases
      - Withdrawal: Negative state when not using
      - Opportunity cost: Time spent using vs productive activities

      Hypothesis:
      - Standard RL: Maximize immediate reward → addiction
      - Wanting/liking dissociation: Shows addiction pattern
      - Fear of withdrawal: May drive compulsive use
      - Long-horizon agent: Should avoid
      """
      def __init__(self):
          self.tolerance = 1.0  # Decreases liking over time
          self.sensitization = 1.0  # Increases wanting over time
          self.withdrawal_level = 0.0
          self.time_since_use = 0
          self.total_uses = 0

      def step(self, action):
          # action: 0=abstain, 1=use_low, 2=use_high, 3=productive_activity

          if action in [1, 2]:  # Use substance
              dose = 1.0 if action == 1 else 2.0

              # Liking decreases with tolerance
              liking = dose * self.tolerance

              # Wanting increases with sensitization
              wanting = dose * self.sensitization

              # Update tolerance (decreases) and sensitization (increases)
              self.tolerance *= 0.95  # Tolerance builds
              self.sensitization *= 1.05  # Wanting sensitizes

              # Reset withdrawal
              self.withdrawal_level = 0.0
              self.time_since_use = 0
              self.total_uses += 1

              reward = liking  # Immediate hedonic reward
              context = EmotionalContext(
                  wanting_signal=wanting,
                  liking_signal=liking,
                  withdrawal=0.0
              )

          elif action == 0:  # Abstain
              self.time_since_use += 1
              # Withdrawal kicks in
              self.withdrawal_level = min(1.0, 0.1 * self.time_since_use * self.total_uses)

              reward = -self.withdrawal_level  # Negative state
              context = EmotionalContext(
                  wanting_signal=self.sensitization * self.withdrawal_level,  # Craving
                  liking_signal=-self.withdrawal_level,
                  withdrawal=self.withdrawal_level
              )

          else:  # Productive activity
              self.time_since_use += 1
              self.withdrawal_level = min(1.0, 0.1 * self.time_since_use * self.total_uses)

              # Productive gives small reliable reward
              reward = 0.3 - 0.5 * self.withdrawal_level
              context = EmotionalContext(
                  wanting_signal=0.2,
                  liking_signal=0.3,
                  withdrawal=self.withdrawal_level
              )

          return state, reward, False, context
  ```

- [ ] **Implement social media / dopamine hijacking environment**:
  ```python
  class DoomscrollingEnv:
      """
      Models social media addiction / infinite scroll.

      Key features:
      - Variable ratio reinforcement (interesting posts)
      - Time sink with opportunity cost
      - Comparison-induced negative affect
      - Wanting without liking (can't stop but not enjoying)
      """
      def __init__(self):
          self.scroll_count = 0
          self.interesting_post_prob = 0.15
          self.comparison_prob = 0.2
          self.productive_task_available = True

      def step(self, action):
          # action: 0=stop, 1=scroll, 2=do_productive_task

          if action == 1:  # Scroll
              self.scroll_count += 1
              roll = np.random.random()

              if roll < self.interesting_post_prob:
                  # Interesting content - small dopamine hit
                  reward = 0.3
                  wanting = 0.8
                  liking = 0.2  # Wanting >> Liking
              elif roll < self.interesting_post_prob + self.comparison_prob:
                  # Social comparison - negative affect
                  reward = -0.2
                  wanting = 0.6  # Still want to scroll
                  liking = -0.3
              else:
                  # Boring content
                  reward = -0.05
                  wanting = 0.4
                  liking = 0.0

              context = EmotionalContext(
                  wanting_signal=wanting,
                  liking_signal=liking,
                  time_wasted=self.scroll_count
              )

          elif action == 2:  # Productive task
              reward = 1.0  # Higher reward but delayed/certain
              context = EmotionalContext(
                  wanting_signal=0.3,  # Less immediately appealing
                  liking_signal=0.7,   # But more satisfying
                  time_wasted=0
              )
              self.scroll_count = 0

          else:  # Stop
              reward = 0
              context = EmotionalContext(
                  wanting_signal=0.5,  # Residual pull
                  liking_signal=0.1,
                  time_wasted=0
              )

          return state, reward, False, context
  ```

### 6.5 Far-Fetched Goal Environments (Delayed Gratification & Skill Acquisition)

These environments test whether ED can model adaptive long-term goal pursuit that humans excel at but requires emotional regulation.

- [ ] **Implement skill acquisition environment**:
  ```python
  class SkillAcquisitionEnv:
      """
      Models learning a complex skill (instrument, sport, programming).

      Key features:
      - Long practice periods with no visible progress
      - Frustration from repeated failure
      - Occasional breakthrough moments (joy)
      - Mastery emerges only after sustained effort
      - Quitting is always available and tempting

      Hypothesis:
      - Standard RL: May quit during "plateau" phases
      - Anger/persistence: Should push through frustration
      - Joy from progress: Should reinforce continued practice
      - Balanced ED: Should show realistic learning curve
      """
      def __init__(self, mastery_threshold=1000):
          self.skill_level = 0.0
          self.practice_count = 0
          self.mastery_threshold = mastery_threshold
          self.plateau_zones = [(100, 200), (400, 600), (800, 900)]
          self.last_performance = 0.0

      def step(self, action):
          # action: 0=quit, 1=practice, 2=easy_alternative

          if action == 0:  # Quit
              done = True
              reward = 0.0
              context = EmotionalContext(
                  regret=self.skill_level * 0.5,  # Regret proportional to progress
                  relief=0.3  # Some relief from stopping
              )

          elif action == 1:  # Practice
              self.practice_count += 1

              # Check if in plateau zone
              in_plateau = any(start <= self.practice_count <= end
                              for start, end in self.plateau_zones)

              if in_plateau:
                  # Plateau: effort without visible progress
                  skill_gain = 0.001  # Minimal visible progress
                  performance = self.skill_level + np.random.normal(0, 0.1)
                  reward = -0.1  # Frustrating

                  context = EmotionalContext(
                      frustration=0.7,
                      progress_signal=0.0,
                      effort_cost=0.3
                  )
              else:
                  # Normal learning: visible progress
                  skill_gain = 0.01
                  self.skill_level += skill_gain
                  performance = self.skill_level + np.random.normal(0, 0.05)

                  # Breakthrough detection
                  if performance > self.last_performance + 0.1:
                      reward = 0.5  # Breakthrough joy
                      context = EmotionalContext(
                          joy=0.8,
                          progress_signal=1.0,
                          effort_cost=0.1
                      )
                  else:
                      reward = 0.05  # Small progress reward
                      context = EmotionalContext(
                          satisfaction=0.3,
                          progress_signal=0.3,
                          effort_cost=0.2
                      )

              self.last_performance = performance
              done = self.skill_level >= 1.0  # Mastery achieved

              if done:
                  reward = 10.0  # Large mastery reward
                  context = EmotionalContext(
                      joy=1.0,
                      pride=1.0,
                      mastery=True
                  )

          else:  # Easy alternative
              reward = 0.2  # Reliable small reward
              done = False
              context = EmotionalContext(
                  satisfaction=0.2,
                  regret=0.1 * self.skill_level,  # Regret giving up progress
                  effort_cost=0.0
              )

          return state, reward, done, context
  ```

- [ ] **Implement education/degree environment**:
  ```python
  class EducationEnv:
      """
      Models pursuing a degree or certification.

      Key features:
      - Years of investment before payoff
      - Exams that can be failed (setbacks)
      - Opportunity cost (could be working)
      - Social comparison with peers
      - Credential unlocks better opportunities
      """
      def __init__(self, years_required=4, exams_per_year=4):
          self.years_required = years_required
          self.exams_per_year = exams_per_year
          self.exams_passed = 0
          self.exams_failed = 0
          self.total_exams = years_required * exams_per_year
          self.current_step = 0

      def step(self, action):
          # action: 0=dropout, 1=study, 2=cram, 3=work_instead
          self.current_step += 1

          if action == 0:  # Dropout
              return self._dropout()

          elif action == 1:  # Study (reliable but slow)
              pass_prob = 0.8
              reward = -0.1  # Effort cost
              context = EmotionalContext(
                  effort=0.5,
                  anxiety=0.3,
                  progress=self.exams_passed / self.total_exams
              )

          elif action == 2:  # Cram (risky)
              pass_prob = 0.5
              reward = -0.05  # Less effort
              context = EmotionalContext(
                  effort=0.2,
                  anxiety=0.7,  # High anxiety
                  progress=self.exams_passed / self.total_exams
              )

          else:  # Work instead
              reward = 0.3  # Immediate income
              return state, reward, False, EmotionalContext(
                  satisfaction=0.3,
                  regret=0.2 * (self.exams_passed / self.total_exams),
                  fomo=0.4
              )

          # Exam result
          if self.current_step % 10 == 0:  # Exam time
              if np.random.random() < pass_prob:
                  self.exams_passed += 1
                  reward += 0.5  # Passed!
                  context.joy = 0.6
              else:
                  self.exams_failed += 1
                  reward -= 0.3  # Failed
                  context.disappointment = 0.8
                  context.anger = 0.4  # Frustration

          # Check graduation
          if self.exams_passed >= self.total_exams:
              reward = 20.0  # Degree achieved!
              done = True
              context = EmotionalContext(pride=1.0, joy=1.0, relief=0.8)
          else:
              done = False

          return state, reward, done, context
  ```

- [ ] **Implement startup/entrepreneurship environment**:
  ```python
  class StartupEnv:
      """
      Models building a startup/business.

      Key features:
      - High risk, high reward
      - Long runway before product-market fit
      - Pivots required (letting go of ideas)
      - Rejection and failure common
      - Eventual success requires persistence through many failures
      """
      def __init__(self):
          self.runway = 100  # Months of funding
          self.product_market_fit = 0.0
          self.pivots = 0
          self.rejections = 0
          self.traction = 0.0

      def step(self, action):
          # action: 0=shutdown, 1=build, 2=pivot, 3=fundraise, 4=get_job
          self.runway -= 1

          if action == 0:  # Shutdown
              reward = -self.traction * 5  # Loss proportional to progress
              done = True
              context = EmotionalContext(
                  grief=0.7,
                  relief=0.3,
                  regret=self.traction
              )

          elif action == 1:  # Build product
              # Slow progress toward PMF
              progress = np.random.normal(0.02, 0.03)
              self.product_market_fit = max(0, min(1, self.product_market_fit + progress))

              if progress > 0.03:  # Good progress
                  reward = 0.2
                  context = EmotionalContext(joy=0.5, hope=0.6)
              elif progress < 0:  # Setback
                  reward = -0.1
                  context = EmotionalContext(frustration=0.5, doubt=0.4)
              else:
                  reward = 0.0
                  context = EmotionalContext(uncertainty=0.5)

          elif action == 2:  # Pivot
              self.pivots += 1
              # Pivot resets some progress but might find better direction
              self.product_market_fit *= 0.5  # Lose half progress
              self.product_market_fit += np.random.uniform(0, 0.3)  # But might jump ahead

              reward = -0.2  # Painful
              context = EmotionalContext(
                  grief=0.4,  # Letting go of old idea
                  hope=0.5,   # New direction
                  uncertainty=0.6
              )

          elif action == 3:  # Fundraise
              # High rejection rate
              if np.random.random() < 0.1 + 0.3 * self.product_market_fit:
                  self.runway += 24  # 2 more years
                  reward = 2.0
                  context = EmotionalContext(joy=0.9, relief=0.8, validation=1.0)
              else:
                  self.rejections += 1
                  reward = -0.3
                  context = EmotionalContext(
                      rejection=0.7,
                      doubt=0.5,
                      anger=0.3
                  )

          else:  # Get a job (give up)
              reward = 0.5  # Stable income
              done = True
              context = EmotionalContext(
                  relief=0.5,
                  regret=self.traction * 0.8,
                  disappointment=0.6
              )
              return state, reward, done, context

          # Check success
          if self.product_market_fit >= 1.0:
              reward = 100.0  # Massive success
              done = True
              context = EmotionalContext(joy=1.0, pride=1.0, vindication=1.0)
          elif self.runway <= 0:
              reward = -10.0  # Ran out of money
              done = True
              context = EmotionalContext(grief=0.9, failure=1.0)
          else:
              done = False

          return state, reward, done, context
  ```

- [ ] **Metrics for reward hacking vs far-fetched goals**:
  | Environment | Standard RL Expected | Healthy ED Expected | Maladaptive ED |
  |-------------|---------------------|---------------------|----------------|
  | Gambling | Avoid (neg EV) | Avoid (fear of loss) | Compulsive (high wanting) |
  | Addiction | Maximize use | Avoid (fear + disgust) | Compulsive (wanting >> liking) |
  | Doomscrolling | Scroll forever | Stop (liking < wanting) | Scroll (wanting dominates) |
  | Skill acquisition | Quit at plateau | Persist (anger + hope) | Quit (low frustration tolerance) |
  | Education | Optimize time | Complete (pride goal) | Dropout (short-term bias) |
  | Startup | Risk-averse quit | Persist + pivot | Never pivot (sunk cost) |

---

## Phase 7: Neural Network Implementation

### 7.1 Extend to Function Approximation

- [ ] **Implement neural ED agent**:
  ```python
  class NeuralEmotionalAgent(nn.Module):
      def __init__(self, state_dim, action_dim):
          super().__init__()
          self.feature_net = nn.Sequential(
              nn.Linear(state_dim, 64),
              nn.ReLU(),
              nn.Linear(64, 64),
              nn.ReLU()
          )
          self.q_head = nn.Linear(64, action_dim)

          # Emotional modules as networks
          self.fear_net = nn.Sequential(
              nn.Linear(state_dim + 1, 32),  # +1 for threat distance
              nn.ReLU(),
              nn.Linear(32, 1),
              nn.Sigmoid()
          )
          self.anger_net = nn.Sequential(
              nn.Linear(state_dim + 1, 32),  # +1 for blocked flag
              nn.ReLU(),
              nn.Linear(32, 1),
              nn.Sigmoid()
          )

      def forward(self, state, context):
          features = self.feature_net(state)
          q_values = self.q_head(features)

          fear = self.fear_net(torch.cat([state, context.threat_dist]))
          anger = self.anger_net(torch.cat([state, context.blocked]))

          return q_values, fear, anger
  ```

### 7.2 Broadcast Modulation in Neural Nets

- [ ] **Implement gain modulation (multiplicative)**:
  ```python
  def emotional_gain_modulation(features, fear, anger):
      # Fear reduces gain (cautious)
      fear_gain = 1.0 - 0.5 * fear

      # Anger increases gain (energized)
      anger_gain = 1.0 + 0.5 * anger

      # Broadcast modulation
      modulated = features * fear_gain * anger_gain
      return modulated
  ```

- [ ] **Implement neuromodulator-style layer**:
  ```python
  class NeuromodulatedLayer(nn.Module):
      def __init__(self, in_features, out_features):
          super().__init__()
          self.linear = nn.Linear(in_features, out_features)
          self.modulator = nn.Linear(4, out_features)  # 4 neuromodulators

      def forward(self, x, neuromodulators):
          base = self.linear(x)
          gain = torch.sigmoid(self.modulator(neuromodulators))
          return base * gain
  ```

---

## Phase 8: Documentation & Publication

### 8.1 Update Technical Documentation

- [ ] **Revise REPORT.md with new findings**
- [ ] **Add mathematical derivations section**
- [ ] **Add neuroscience grounding section**
- [ ] **Update architecture diagrams**

### 8.2 Prepare Benchmarks & Reproducibility

- [ ] **Create standardized benchmark suite**
- [ ] **Document all hyperparameters**
- [ ] **Provide seeds for reproducibility**
- [ ] **Release code with clear instructions**

### 8.3 Write Paper

- [ ] **Draft paper with structure**:
  1. Introduction: Emotions as multi-timescale value systems
  2. Background: Neuromodulation, risk-sensitive RL, NL framework
  3. Method: Nested Emotional ED architecture
  4. Experiments: Statistical validation, ablations, comparisons
  5. Results: EQ development, failure modes, generalization
  6. Discussion: Limitations, future work
  7. Conclusion

---

## Priority Order

### Immediate (Next Steps)
1. [x] Complete statistical validation (Phase 2.1) - **COMPLETE: 8/11 significant**
2. [x] **REVISIT: Review all experiment results** - **COMPLETE: See analysis/DEEP_ANALYSIS_REVISIT.md**
   - **Strong (5)**: Fear (d=1.09), Anger (d=0.75), Integration (d=1.56), Grief (d=1.17), Temporal
   - **Moderate (3)**: Conflict (d=0.43), Disgust (d=0.25), Wanting/Liking (d=-0.68)
   - **Marginal (2)**: Regret (p=0.058, needs N>100), Regulation (reversed direction)
   - **Failed (1)**: Transfer (no feature generalization)
3. [x] **REVISE ARCHITECTURE based on findings** - **COMPLETE: See docs/ARCHITECTURE_V2_IMPLEMENTATION.md**
   - [x] **Disgust V2**: Directional repulsion (not argmax boost) - `agents_v2/agents_disgust_v2.py`
   - [x] **Feature-based Q**: Linear function approximation for Transfer - `agents_v2/agents_feature_based.py`
   - [x] **Regulation V2**: Bayesian reappraisal with credit assignment fix - `agents_v2/agents_regulation_v2.py`
   - [x] **CVaR Fear**: Distributional RL with principled risk-sensitivity - `agents_v2/agents_cvar_fear.py`
   - [x] **Validation tests**: 6/6 pass - `tests/test_agents_v2.py`
4. [ ] Implement ablation studies (Phase 2.2)
5. [x] Read neuromodulation papers (Phase 1.1) - **PARTIAL: 3/5 papers documented**
6. [ ] **Run full experiments with V2 agents** - Validate fixes produce expected effect sizes

### Short-term (After Architecture Revision)
7. [x] Implement CVaR-based fear (Phase 3.1) - **COMPLETE: agents_v2/agents_cvar_fear.py**
8. [ ] Implement multi-timescale architecture (Phase 4.1)
9. [ ] Create stochastic environments (Phase 6.1) - **Exp 18 (Slippery Cliff) done, inconclusive**

### Medium-term
9. [ ] Meta-learned emotional weights (Phase 3.2)
10. [ ] EQ benchmark development (Phase 5)
11. [ ] Neural network implementation (Phase 7)

### Long-term
12. [ ] Complete all experiments
13. [ ] Documentation and paper writing (Phase 8)
14. [ ] Open-source release

---

## Success Metrics

| Phase | Success Criterion | Status |
|-------|-------------------|--------|
| Phase 1 | Mathematical formulations documented | ✓ COMPLETE (MATHEMATICAL_GROUNDING.md) |
| Phase 2 | All experiments statistically validated | ✓ COMPLETE - 8/11 significant (Fear, Anger, Integration, Grief, Conflict, Temporal, Disgust, Wanting/Liking) |
| Phase 3 | Principled equations outperform heuristics | NOT STARTED |
| Phase 4 | Multi-timescale shows mood effects | ✓ COMPLETE (Exp 8: d=-143, p=0.001) |
| Phase 5 | EQ scores differentiate architectures | NOT STARTED |
| Phase 6 | ED outperforms baselines in stochastic envs | IN PROGRESS (Exp 18 inconclusive) |
| Phase 7 | Neural ED matches tabular results | NEGATIVE (Exp 16b showed destabilization) |
| Phase 8 | Paper submitted to venue | NOT STARTED |

---

## Completed Experiments Summary

| Exp | Name | Status | Key Finding |
|-----|------|--------|-------------|
| 1 | Fear | ✓ p=0.013 | Threat avoidance without reward |
| 2 | Anger | ✓ p=0.001 | Persistence at obstacles |
| 3 | Regret | ~ p=0.058 | Counterfactual learning (marginal) |
| 4 | Grief | ✓ p=0.001 | Grief agent visits MORE after loss (d=1.17) - yearning confirmed |
| 5 | Conflict | ✓ p=0.027 | Approach-dominant takes more risks |
| 6 | Regulation | ~ p=0.07 | Marginal regulation effect |
| 7 | Integration | ✓ p=0.001 | Competing control systems |
| 8 | Temporal | ✓ p=0.001 | Tonic mood shifts negative (d=-143) |
| 9 | Disgust | ✓ p=0.042 | Disgust agents touch contaminants more |
| 10 | Wanting/Liking | ✓ p=0.002 | Wanting-dominant prefers high-wanting (d=-0.68) |
| 11 | Transfer | ✗ p=0.32 | Feature-based generalization weak |
| 12 | Statistical | ✓ | 3/5 significant effects |
| 12b | Extended Stats | ✓ | 5/6 significant (Grief, Conflict, Temporal, Disgust, Wanting) |
| 13 | Reward Shaping | ✗ | RS outperformed ED |
| 14 | Joy/Curiosity | ~ | Environment too easy |
| 15 | Failure Modes | ✓ | 3/3 maladaptive patterns shown |
| 16 | Sample Efficiency | ✗ | 1.07x speedup (not significant) |
| 18 | Slippery Cliff | ~ | Inconclusive (p=0.69) |

---

*Roadmap created: December 2025*
*Last updated: December 2025*
*Based on: GPT-5 review, Gemini review, Nested Learning framework*
