"""Nested Emotional Agent using multi-timescale affect architecture.

Based on:
- Behrouz et al. (2025): Nested Learning framework
- Phase 4 design from EXPERIMENT_PLAN_V3.md

Key Features:
1. Multi-timescale affect: phasic (immediate) to tonic (mood/temperament)
2. LSS-based emotional triggering: emotions from prediction errors
3. Level-specific updates: different emotions update at different frequencies
4. Cross-level modulation: mood affects emotional reactivity

This agent integrates:
- ContinuumAffect: Multi-timescale emotional memory
- LSSEmotionalTrigger: Prediction-error based emotion triggering
- Neuromodulator mappings: DA→reward, NE→fear, 5-HT→patience, ACh→attention
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Import from our modules
import sys
sys.path.insert(0, '/home/ak/emotional-ed-test')
from src.modules.continuum_affect import ContinuumAffect, MultiChannelAffect
from src.modules.lss_trigger import LSSEmotionalTrigger, EmotionalContext, AdaptiveLSSThreshold


@dataclass
class NestedContext:
    """Extended context for nested emotional processing."""
    # Threat-related
    threat_distance: float = float('inf')
    near_threat: bool = False

    # Goal-related
    goal_distance: float = float('inf')
    was_blocked: bool = False
    consecutive_blocks: int = 0

    # Outcome-related
    reward: float = 0.0

    # State info
    is_terminal: bool = False
    step: int = 0


class NestedEmotionalAgent:
    """Q-learning agent with nested multi-timescale emotional architecture.

    Architecture:
    - Level 0 (Phasic): Immediate fear/surprise responses
    - Level 1 (Episode): Short-term emotional episodes
    - Level 2 (Mood): Persistent mood states across episodes
    - Level 3 (Temperament): Stable personality over many episodes

    Emotional channels:
    - Fear: Threat avoidance (NE-like)
    - Anger: Persistence when blocked (DA-like)
    - Joy: Positive reinforcement (DA-like)
    - Curiosity: Exploration drive (ACh-like)

    Modulation mechanisms:
    - LR modulation: Emotions affect learning rate
    - Policy modulation: Emotions bias action selection
    - Attention modulation: Emotions focus on relevant features
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        # Emotional parameters
        fear_weight: float = 0.5,
        anger_weight: float = 0.3,
        joy_weight: float = 0.2,
        # Multi-timescale parameters
        n_timescales: int = 4,
        mood_influence: float = 0.3,
        # LSS parameters
        surprise_threshold: float = 0.1
    ):
        """Initialize nested emotional agent.

        Args:
            n_states: Number of states
            n_actions: Number of actions
            lr: Base learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            fear_weight: Weight for fear modulation
            anger_weight: Weight for anger modulation
            joy_weight: Weight for joy modulation
            n_timescales: Number of timescale levels (default 4)
            mood_influence: How much mood affects emotional reactivity
            surprise_threshold: Minimum LSS to trigger emotions
        """
        # Q-table
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions

        # Emotional weights
        self.fear_weight = fear_weight
        self.anger_weight = anger_weight
        self.joy_weight = joy_weight
        self.mood_influence = mood_influence

        # Multi-channel affect with separate timescales per emotion
        self.affect = MultiChannelAffect(
            channels=['fear', 'anger', 'joy', 'curiosity'],
            n_timescales=n_timescales
        )

        # LSS-based emotional trigger
        self.lss_trigger = LSSEmotionalTrigger(
            surprise_threshold=surprise_threshold,
            fear_sensitivity=1.0,
            anger_sensitivity=1.0,
            joy_sensitivity=1.0
        )

        # Adaptive threshold for LSS
        self.adaptive_threshold = AdaptiveLSSThreshold(
            initial_threshold=surprise_threshold
        )

        # State tracking
        self.step_count = 0
        self.episode_count = 0
        self.consecutive_blocks = 0

        # Current emotional state (for action selection)
        self.current_emotions: Dict[str, float] = {}
        self.current_mood = 0.0

        # Visit counts for curiosity
        self.visit_counts = np.zeros(n_states)

    def _compute_lss(self, state: int, action: int, reward: float,
                     next_state: int, done: bool) -> float:
        """Compute Local Surprise Signal (prediction error).

        LSS = actual - predicted
        Positive: better than expected
        Negative: worse than expected
        """
        predicted = self.Q[state, action]
        actual = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        return actual - predicted

    def _build_emotional_context(self, context: NestedContext,
                                  reward: float) -> EmotionalContext:
        """Convert NestedContext to EmotionalContext for LSS trigger."""
        return EmotionalContext(
            near_threat=context.near_threat,
            threat_distance=context.threat_distance,
            goal_distance=context.goal_distance,
            was_blocked=context.was_blocked,
            consecutive_blocks=context.consecutive_blocks,
            reward=reward,
            is_terminal=context.is_terminal
        )

    def _compute_curiosity(self, state: int) -> float:
        """Compute curiosity signal based on state novelty."""
        visit_count = self.visit_counts[state]
        # Inverse of visit count (novel states are more interesting)
        return 1.0 / (1.0 + visit_count)

    def _get_mood_modulated_reactivity(self) -> float:
        """Get emotional reactivity modulated by current mood.

        Negative mood -> higher reactivity (more sensitive to threats)
        Positive mood -> lower reactivity (more resilient)
        """
        mood = self.affect.get_integrated_mood()
        # Mood in [-1, 1], reactivity in [0.7, 1.3]
        return 1.0 - self.mood_influence * mood

    def select_action(self, state: int, context: Optional[NestedContext] = None) -> int:
        """Select action with emotional modulation.

        Emotions affect action selection:
        - Fear: Bias toward higher-Q (safer) actions
        - Anger: Increase exploitation (reduce exploration)
        - Joy: Slight bias toward positive experiences
        - Curiosity: Bias toward unvisited states
        """
        # Random exploration
        if np.random.random() < self.epsilon:
            # Even random actions are curiosity-biased
            curiosity = self._compute_curiosity(state)
            if curiosity > 0.5:
                # Prefer actions leading to unvisited states
                # This is approximated by Q-value uncertainty
                return np.random.randint(self.n_actions)
            return np.random.randint(self.n_actions)

        # Get Q-values
        q_values = self.Q[state].copy()

        # Fear modulation: bias toward higher-Q actions
        fear = self.current_emotions.get('fear', 0.0)
        if fear > 0.1:
            q_min, q_max = q_values.min(), q_values.max()
            if q_max > q_min:
                normalized = (q_values - q_min) / (q_max - q_min)
                q_values += fear * normalized * self.fear_weight

        # Anger modulation: sharpen distribution (more greedy)
        anger = self.current_emotions.get('anger', 0.0)
        if anger > 0.1:
            # Temperature reduction
            temperature = 1.0 / (1.0 + anger * self.anger_weight)
            q_mean = q_values.mean()
            q_values = q_mean + (q_values - q_mean) / temperature

        # Joy modulation: slight optimism bias
        joy = self.current_emotions.get('joy', 0.0)
        if joy > 0.1:
            q_values += joy * self.joy_weight * 0.1

        return int(np.argmax(q_values))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: NestedContext):
        """Update with nested emotional modulation.

        1. Compute LSS (prediction error)
        2. Trigger emotions based on LSS + context
        3. Update multi-timescale affect
        4. Modulate learning rate by emotions
        5. Standard TD update with modulated LR
        """
        self.step_count += 1

        # Track visits for curiosity
        self.visit_counts[state] += 1

        # Track blocking for anger
        if context.was_blocked:
            self.consecutive_blocks += 1
        else:
            self.consecutive_blocks = 0
        context.consecutive_blocks = self.consecutive_blocks

        # 1. Compute Local Surprise Signal
        lss = self._compute_lss(state, action, reward, next_state, done)

        # Update running prediction in LSS trigger
        self.lss_trigger.update_prediction(reward)

        # Update adaptive threshold
        self.adaptive_threshold.update(lss)

        # 2. Trigger emotions based on LSS and context
        emotional_context = self._build_emotional_context(context, reward)
        emotions = self.lss_trigger.trigger_emotions(lss, emotional_context)

        # Add curiosity (not LSS-based)
        curiosity = self._compute_curiosity(next_state)
        if curiosity > 0.3:
            emotions['curiosity'] = curiosity

        # 3. Update multi-timescale affect
        self.affect.update(emotions, record_history=False)

        # Store current emotions for action selection
        self.current_emotions = emotions
        self.current_mood = self.affect.get_integrated_mood()

        # 4. Compute learning rate modulation
        reactivity = self._get_mood_modulated_reactivity()

        # Fear increases learning from negative outcomes
        fear = emotions.get('fear', 0.0)
        fear_mod = 1.0 + self.fear_weight * fear * reactivity * (1.0 if lss < 0 else 0.5)

        # Anger increases learning rate overall
        anger = emotions.get('anger', 0.0)
        anger_mod = 1.0 + self.anger_weight * anger * reactivity

        # Joy slightly increases learning from positive outcomes
        joy = emotions.get('joy', 0.0)
        joy_mod = 1.0 + self.joy_weight * joy * (1.0 if lss > 0 else 0.3)

        # Curiosity increases learning from novel states
        curiosity = emotions.get('curiosity', 0.0)
        curiosity_mod = 1.0 + 0.3 * curiosity

        # Combined modulation
        effective_lr = self.lr * fear_mod * anger_mod * joy_mod * curiosity_mod
        effective_lr = min(0.5, effective_lr)  # Cap for stability

        # 5. TD update
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]
        self.Q[state, action] += effective_lr * delta

    def on_episode_end(self):
        """Called at end of episode for tonic updates."""
        self.episode_count += 1

        # Reset phasic states but preserve tonic (mood, temperament)
        self.affect.reset(preserve_temperament=True)

        # Reset per-episode tracking
        self.consecutive_blocks = 0
        self.current_emotions = {}

    def reset_episode(self):
        """Reset for new episode (alias for on_episode_end)."""
        self.on_episode_end()

    def get_emotional_state(self) -> Dict:
        """Return complete emotional state for logging."""
        return {
            'current_emotions': self.current_emotions.copy(),
            'mood': self.current_mood,
            'affect_state': self.affect.get_state(),
            'step': self.step_count,
            'episode': self.episode_count
        }


class SimplifiedNestedAgent:
    """Simplified version for quick experiments.

    Uses just 2 timescales (phasic/tonic) and 2 emotions (fear/curiosity).
    Good for ablation studies against full NestedEmotionalAgent.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        fear_weight: float = 0.5,
        curiosity_weight: float = 0.3
    ):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.fear_weight = fear_weight
        self.curiosity_weight = curiosity_weight

        # Simple 2-timescale affect
        self.phasic_fear = 0.0
        self.tonic_fear = 0.0
        self.phasic_decay = 0.5
        self.tonic_decay = 0.95

        # Visit tracking for curiosity
        self.visit_counts = np.zeros(n_states)

    def select_action(self, state: int, context: Optional[NestedContext] = None) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Fear modulation
        total_fear = self.phasic_fear + 0.5 * self.tonic_fear
        if total_fear > 0.1:
            q_min, q_max = q_values.min(), q_values.max()
            if q_max > q_min:
                normalized = (q_values - q_min) / (q_max - q_min)
                q_values += total_fear * normalized * self.fear_weight

        return int(np.argmax(q_values))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context: NestedContext):
        # Update visit count
        self.visit_counts[state] += 1

        # Compute fear from context
        if context.near_threat:
            fear_input = max(0, 1.0 - context.threat_distance / 3.0)
        else:
            fear_input = 0.0

        # Update phasic (fast) and tonic (slow) fear
        self.phasic_fear = self.phasic_decay * self.phasic_fear + (1 - self.phasic_decay) * fear_input
        self.tonic_fear = self.tonic_decay * self.tonic_fear + (1 - self.tonic_decay) * fear_input

        # Curiosity from novelty
        curiosity = 1.0 / (1.0 + self.visit_counts[next_state])

        # Learning rate modulation
        total_fear = self.phasic_fear + 0.5 * self.tonic_fear

        # LSS
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        delta = target - self.Q[state, action]

        # Fear increases learning from negative
        fear_mod = 1.0 + self.fear_weight * total_fear * (1.0 if delta < 0 else 0.5)
        curiosity_mod = 1.0 + self.curiosity_weight * curiosity

        effective_lr = min(0.5, self.lr * fear_mod * curiosity_mod)
        self.Q[state, action] += effective_lr * delta

    def reset_episode(self):
        # Preserve tonic, reset phasic
        self.phasic_fear = 0.0


# Example usage
if __name__ == "__main__":
    print("=== Nested Emotional Agent Demo ===\n")

    # Create agent
    agent = NestedEmotionalAgent(n_states=25, n_actions=4)

    # Simulate some steps
    context = NestedContext(
        threat_distance=2.0,
        near_threat=True,
        goal_distance=5.0,
        was_blocked=False,
        reward=-0.1
    )

    # Simulate episode
    state = 0
    for step in range(10):
        action = agent.select_action(state, context)
        next_state = (state + 1) % 25
        reward = -0.1 if step < 9 else 1.0
        done = step == 9

        # Update context
        context.step = step
        context.reward = reward
        context.threat_distance = max(0.5, 3.0 - step * 0.3)
        context.near_threat = context.threat_distance < 2.0

        agent.update(state, action, reward, next_state, done, context)

        print(f"Step {step}: state={state}, action={action}, reward={reward:.2f}")
        print(f"  Emotions: {agent.current_emotions}")
        print(f"  Mood: {agent.current_mood:.3f}")

        state = next_state

    print("\nFinal emotional state:")
    print(agent.get_emotional_state())
