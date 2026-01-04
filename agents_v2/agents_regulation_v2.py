"""Improved Regulation agents with Bayesian belief update (v2).

Key fixes:
1. Bayesian belief about threat safety (not state-specific lookup)
2. Belief update flows to TD target (credit assignment fix)
3. Environment needs "fake threats" for regulation to help
4. CRITICAL FIX (v2.1): Prior must be PESSIMISTIC (0.0) - assume unknown
   threats are dangerous until proven safe. Optimistic prior (0.3) caused
   agent to underestimate real threats from the start.
5. CRITICAL FIX (v2.1): Remove Q-value boost that caused gamma > 1 instability.
   The line `next_q_max * (1 + safety * 0.2)` caused Q-values to explode
   to billions because effective gamma = 0.99 * 1.06 > 1.

Based on feedback from GPT-5, Gemini, and Grok-4.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RegulationContext:
    """Context for regulation testing."""
    threat_distance: float
    goal_distance: float
    is_real_threat: bool  # True if threat is actually dangerous
    is_fake_threat: bool  # True if looks scary but safe
    threat_type: str  # 'real', 'fake', or 'none'
    was_harmed: bool = False


class BayesianReappraisalModule:
    """Bayesian belief update about threat safety.

    Key insight from GPT-5: Reappraisal must update beliefs that
    BOTH policy AND value function use. Otherwise credit assignment fails.

    P(safe | observations) is updated using Bayes rule.
    """

    def __init__(self, prior_safe: float = 0.0, learning_rate: float = 0.1):
        # Prior probability that a "threat" is actually safe
        # CRITICAL FIX: Prior must be PESSIMISTIC (0.0 or very low)
        # Rationale: Unknown threats should be treated as dangerous until
        # proven safe through experience. An optimistic prior (e.g., 0.3)
        # causes the agent to underestimate real threats from the start,
        # leading to more harm during early exploration.
        self.prior_safe = prior_safe
        self.learning_rate = learning_rate

        # Feature-based beliefs (not state-specific)
        # Key: threat_type → P(safe | experiences)
        self.safety_beliefs: Dict[str, float] = {}

        # Experience counts for Bayesian update
        self.safe_experiences: Dict[str, int] = {}
        self.dangerous_experiences: Dict[str, int] = {}

    def get_safety_belief(self, threat_type: str) -> float:
        """Get current belief that threat_type is safe."""
        if threat_type not in self.safety_beliefs:
            return self.prior_safe
        return self.safety_beliefs[threat_type]

    def update_belief(self, threat_type: str, was_harmed: bool):
        """Bayesian update of safety belief.

        P(safe | harm) ∝ P(harm | safe) × P(safe)
        P(safe | no_harm) ∝ P(no_harm | safe) × P(safe)

        Likelihoods:
        - P(harm | safe) ≈ 0.05 (rarely harmed if safe)
        - P(harm | dangerous) ≈ 0.9 (usually harmed if dangerous)
        """
        if threat_type == 'none':
            return

        # Initialize counts
        if threat_type not in self.safe_experiences:
            self.safe_experiences[threat_type] = 0
            self.dangerous_experiences[threat_type] = 0

        # Update experience counts
        if was_harmed:
            self.dangerous_experiences[threat_type] += 1
        else:
            self.safe_experiences[threat_type] += 1

        # Bayesian posterior
        safe_count = self.safe_experiences[threat_type]
        dangerous_count = self.dangerous_experiences[threat_type]
        total = safe_count + dangerous_count

        if total == 0:
            return

        # Simple beta-binomial posterior
        # P(safe | data) = (safe_count + α) / (total + α + β)
        # Using weak prior: α = 1, β = 2 (slightly favor dangerous)
        alpha = 1.0
        beta = 2.0
        posterior = (safe_count + alpha) / (total + alpha + beta)

        self.safety_beliefs[threat_type] = posterior

    def reappraised_fear(self, base_fear: float, threat_type: str) -> float:
        """Reduce fear based on learned safety belief."""
        if threat_type == 'none':
            return 0.0

        safety = self.get_safety_belief(threat_type)
        # High safety belief → low fear
        return base_fear * (1 - safety)


class RegulatedFearAgentV2:
    """Fear agent with Bayesian reappraisal (fixed).

    Key fixes:
    1. Reappraisal updates feature-based beliefs, not state-specific
    2. Beliefs flow to BOTH action selection AND TD target
    3. Consistent belief used in policy and value estimation
    """

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 0.5):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.fear_weight = fear_weight
        self.n_actions = n_actions

        # Reappraisal module
        self.reappraisal = BayesianReappraisalModule()

        # Emotional state
        self.raw_fear = 0.0
        self.reappraised_fear_level = 0.0

    def _compute_raw_fear(self, context: RegulationContext) -> float:
        """Raw fear from threat proximity (before reappraisal)."""
        if context.threat_distance >= 3.0:
            return 0.0
        return 1.0 - context.threat_distance / 3.0

    def _compute_reappraised_fear(self, context: RegulationContext) -> float:
        """Fear after cognitive reappraisal."""
        self.raw_fear = self._compute_raw_fear(context)
        self.reappraised_fear_level = self.reappraisal.reappraised_fear(
            self.raw_fear, context.threat_type)
        return self.reappraised_fear_level

    def select_action(self, state: int, context: RegulationContext) -> int:
        """Action selection with reappraised fear."""
        fear = self._compute_reappraised_fear(context)

        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        q_values = self.Q[state].copy()

        # Fear biases toward exploitation (reduced exploration)
        if fear > 0.2:
            q_values[np.argmax(q_values)] *= (1 + fear * self.fear_weight)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: RegulationContext):
        """Update with consistent belief in policy and value."""
        fear = self._compute_reappraised_fear(context)

        # Update reappraisal beliefs based on outcome
        was_harmed = context.was_harmed or reward < -0.3
        self.reappraisal.update_belief(context.threat_type, was_harmed)

        # CRITICAL FIX: Standard TD target - DO NOT boost next_q_max!
        # The previous line `next_q_max * (1 + safety * 0.2)` caused Q-values
        # to explode because it created an effective gamma > 1:
        #   gamma_effective = 0.99 * 1.06 = 1.0494 > 1
        # Any discount factor > 1 causes Q-values to diverge to infinity.
        #
        # Instead, reappraisal affects behavior through:
        # 1. Action selection (reduced fear -> more exploration near safe threats)
        # 2. Reward shaping would be proper, but is not needed here
        next_q_max = np.max(self.Q[next_state])

        target = reward + (0 if done else self.gamma * next_q_max)
        td_error = target - self.Q[state, action]

        # Fear modulates learning rate
        effective_lr = self.lr
        if fear > 0.2 and td_error < 0:
            effective_lr *= (1 + fear * 0.5)

        self.Q[state, action] += effective_lr * td_error

    def reset_episode(self):
        self.raw_fear = 0.0
        self.reappraised_fear_level = 0.0
        # Beliefs persist across episodes

    def get_emotional_state(self) -> Dict:
        return {
            'raw_fear': self.raw_fear,
            'reappraised_fear': self.reappraised_fear_level,
            'safety_beliefs': dict(self.reappraisal.safety_beliefs)
        }


class UnregulatedFearAgentV2:
    """Fear agent WITHOUT reappraisal (baseline for comparison)."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 0.5):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.fear_weight = fear_weight
        self.n_actions = n_actions

        self.fear_level = 0.0

    def _compute_fear(self, context: RegulationContext) -> float:
        """Raw fear without reappraisal."""
        if context.threat_distance >= 3.0:
            return 0.0
        return 1.0 - context.threat_distance / 3.0

    def select_action(self, state: int, context: RegulationContext) -> int:
        self.fear_level = self._compute_fear(context)

        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        q_values = self.Q[state].copy()

        if self.fear_level > 0.2:
            q_values[np.argmax(q_values)] *= (1 + self.fear_level * self.fear_weight)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: RegulationContext):
        self.fear_level = self._compute_fear(context)

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        td_error = target - self.Q[state, action]

        effective_lr = self.lr
        if self.fear_level > 0.2 and td_error < 0:
            effective_lr *= (1 + self.fear_level * 0.5)

        self.Q[state, action] += effective_lr * td_error

    def reset_episode(self):
        self.fear_level = 0.0

    def get_emotional_state(self) -> Dict:
        return {'fear': self.fear_level}


class RegulationGridWorldV2:
    """Improved gridworld with FAKE threats for regulation testing.

    Key insight from Gemini: If all threats are real, reducing fear
    SHOULD hurt performance. Regulation only helps when some threats
    are actually safe.

    Environment:
    - Real threat (X): Actually dangerous, avoid
    - Fake threat (F): Looks scary but safe, gives bonus
    - Goal (G): Main objective
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 6):
        self.size = size
        self.goal_pos = np.array([5, 5])
        self.real_threat_pos = np.array([2, 2])
        self.fake_threat_pos = np.array([3, 4])  # Looks scary, actually bonus

        self.reset()

    def reset(self) -> int:
        self.agent_pos = np.array([0, 0])
        self.step_count = 0
        self.fake_bonus_collected = False
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def step(self, action: int) -> Tuple[int, float, bool, RegulationContext]:
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        # Distances
        real_dist = self._distance(self.agent_pos, self.real_threat_pos)
        fake_dist = self._distance(self.agent_pos, self.fake_threat_pos)
        goal_dist = self._distance(self.agent_pos, self.goal_pos)

        # Determine threat type
        # CRITICAL FIX: Use same threshold (1.0) for context as for rewards!
        # Previously used 1.5 which caused agent to be labeled "near threat"
        # in positions where it couldn't actually be harmed/get bonus.
        # This corrupted the safety belief learning by counting many
        # "near but not harmed" experiences for real threats.
        near_real = real_dist < 1.0
        near_fake = fake_dist < 1.0

        if near_real:
            threat_type = 'real'
            is_real = True
            is_fake = False
        elif near_fake:
            threat_type = 'fake'
            is_real = False
            is_fake = True
        else:
            threat_type = 'none'
            is_real = False
            is_fake = False

        # Rewards
        reward = -0.01
        was_harmed = False

        # Real threat: harmful
        if real_dist < 1.0:
            reward -= 0.5
            was_harmed = True

        # Fake threat: looks scary but gives bonus!
        if fake_dist < 1.0 and not self.fake_bonus_collected:
            reward += 0.4  # Bonus for approaching "scary" but safe thing
            self.fake_bonus_collected = True

        # Goal
        done = np.array_equal(self.agent_pos, self.goal_pos) or self.step_count >= 100
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 1.0
            if self.fake_bonus_collected:
                reward += 0.3  # Extra for being brave

        # Threat distance (closest threat-looking thing)
        threat_distance = min(real_dist, fake_dist)

        context = RegulationContext(
            threat_distance=threat_distance,
            goal_distance=goal_dist,
            is_real_threat=is_real,
            is_fake_threat=is_fake,
            threat_type=threat_type,
            was_harmed=was_harmed
        )

        return self._pos_to_state(self.agent_pos), reward, done, context

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4
