"""Multi-timescale affect module following Nested Learning principles.

Based on:
- Behrouz et al. (2025): Nested Learning framework
- Neuroscience of multi-timescale emotional processing
- Phasic (fast) vs Tonic (slow) affect distinction

Key insight: Emotions operate at multiple timescales simultaneously:
- Level 0 (phasic): Immediate reactions to stimuli (fear spike)
- Level 1: Short-term emotional episodes (~10 steps)
- Level 2 (tonic): Mood states (~1 episode)
- Level 3: Temperament/personality (~100 episodes)
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class AffectSnapshot:
    """Snapshot of affect state at a point in time."""
    timestamp: int
    phasic: float
    mood: float
    temperament: float
    all_levels: List[float]


class ContinuumAffect:
    """Multi-timescale affect memory using exponential moving averages.

    Each timescale is an EMA with different decay rates:
    - Fast timescales capture transient emotions
    - Slow timescales capture persistent mood/temperament

    This creates a continuum from phasic (immediate) to tonic (lasting).
    """

    def __init__(self, n_timescales: int = 5):
        """Initialize continuum affect.

        Args:
            n_timescales: Number of timescale levels (default 5 covers
                         1, 10, 100, 1000, 10000 step windows)
        """
        # Timescales as powers of 10
        self.n_timescales = n_timescales
        self.timescales = np.array([10**i for i in range(n_timescales)])

        # Affect levels at each timescale
        self.affect_levels = np.zeros(n_timescales)

        # Decay rates: larger timescale = slower decay = more persistence
        # decay = 1 - 1/timescale (approaches 1 for slow timescales)
        self.decay_rates = 1 - 1 / self.timescales

        # Learning rates (complement of decay)
        self.learning_rates = 1 - self.decay_rates

        # History for analysis
        self.history: List[AffectSnapshot] = []
        self.step_count = 0

    def update(self, emotional_input: float, record_history: bool = False):
        """Update all timescales with new emotional input.

        Args:
            emotional_input: Current emotional signal (can be negative)
            record_history: Whether to record snapshot for analysis
        """
        self.step_count += 1

        # Update each timescale with its own EMA
        for i in range(self.n_timescales):
            self.affect_levels[i] = (
                self.decay_rates[i] * self.affect_levels[i] +
                self.learning_rates[i] * emotional_input
            )

        if record_history:
            self.history.append(AffectSnapshot(
                timestamp=self.step_count,
                phasic=self.get_phasic(),
                mood=self.get_mood(),
                temperament=self.get_temperament(),
                all_levels=self.affect_levels.copy().tolist()
            ))

    def get_phasic(self) -> float:
        """Get fast timescale = phasic emotional response.

        This is the immediate, reactive component of affect.
        High variance, responds quickly to stimuli.
        """
        return float(self.affect_levels[0])

    def get_tonic(self) -> float:
        """Get slow timescales = tonic mood/baseline.

        Average of slower timescales (levels 2+).
        Low variance, changes slowly over time.
        """
        if self.n_timescales > 2:
            return float(np.mean(self.affect_levels[2:]))
        return float(self.affect_levels[-1])

    def get_mood(self) -> float:
        """Get weighted combination emphasizing slower timescales.

        Mood = weighted average where slower timescales contribute more.
        This gives a stable but responsive mood signal.
        """
        # Weight by inverse timescale (slow = more weight for mood)
        weights = 1 / self.timescales
        weights = weights / weights.sum()
        return float(np.dot(weights, self.affect_levels))

    def get_temperament(self) -> float:
        """Get slowest timescale = temperament/personality.

        This represents stable, long-term emotional disposition.
        Should change very slowly (over hundreds of episodes).
        """
        return float(self.affect_levels[-1])

    def get_arousal(self) -> float:
        """Get current arousal level (absolute phasic magnitude).

        High arousal = strong emotional reaction (positive or negative).
        """
        return abs(self.get_phasic())

    def get_valence(self) -> float:
        """Get current emotional valence (sign of mood).

        Positive valence = positive mood
        Negative valence = negative mood
        """
        return float(np.sign(self.get_mood()))

    def get_state(self) -> Dict:
        """Return complete affect state for logging."""
        return {
            'phasic': self.get_phasic(),
            'tonic': self.get_tonic(),
            'mood': self.get_mood(),
            'temperament': self.get_temperament(),
            'arousal': self.get_arousal(),
            'valence': self.get_valence(),
            'all_levels': self.affect_levels.tolist()
        }

    def reset(self, preserve_temperament: bool = True):
        """Reset affect state.

        Args:
            preserve_temperament: If True, keep slowest timescale
                                 (personality persists across episodes)
        """
        if preserve_temperament:
            temperament = self.affect_levels[-1]
            self.affect_levels = np.zeros(self.n_timescales)
            self.affect_levels[-1] = temperament
        else:
            self.affect_levels = np.zeros(self.n_timescales)

    def get_history_array(self) -> np.ndarray:
        """Return history as numpy array for analysis."""
        if not self.history:
            return np.array([])
        return np.array([
            [s.timestamp, s.phasic, s.mood, s.temperament]
            for s in self.history
        ])


class MultiChannelAffect:
    """Multiple affect continua for different emotional dimensions.

    Each channel (fear, anger, joy, etc.) has its own multi-timescale
    representation, allowing independent phasic/tonic dynamics.
    """

    def __init__(self, channels: List[str], n_timescales: int = 5):
        """Initialize multi-channel affect.

        Args:
            channels: List of emotion channel names
            n_timescales: Number of timescales per channel
        """
        self.channels = channels
        self.affects = {
            channel: ContinuumAffect(n_timescales)
            for channel in channels
        }

    def update(self, emotional_inputs: Dict[str, float],
               record_history: bool = False):
        """Update all channels with their respective inputs.

        Args:
            emotional_inputs: Dict mapping channel name to input value
            record_history: Whether to record snapshots
        """
        for channel, value in emotional_inputs.items():
            if channel in self.affects:
                self.affects[channel].update(value, record_history)

    def get_channel(self, channel: str) -> ContinuumAffect:
        """Get affect object for specific channel."""
        return self.affects[channel]

    def get_integrated_mood(self) -> float:
        """Get integrated mood across all channels.

        Combines: fear (negative) + anger (mixed) + joy (positive)
        """
        mood = 0.0
        weights = {
            'fear': -1.0,   # Fear contributes negatively
            'anger': -0.3,  # Anger slightly negative
            'joy': 1.0,     # Joy positive
            'grief': -0.8,  # Grief negative
            'disgust': -0.5  # Disgust negative
        }
        for channel, affect in self.affects.items():
            weight = weights.get(channel, 0.0)
            mood += weight * affect.get_mood()
        return mood

    def get_state(self) -> Dict[str, Dict]:
        """Return complete state for all channels."""
        return {
            channel: affect.get_state()
            for channel, affect in self.affects.items()
        }

    def reset(self, preserve_temperament: bool = True):
        """Reset all channels."""
        for affect in self.affects.values():
            affect.reset(preserve_temperament)


class NestedLevelAffect:
    """Affect with explicit level-based updates (Nested Learning style).

    Instead of continuous EMA, updates happen at level-specific frequencies:
    - Level 0: Every step
    - Level 1: Every 10 steps
    - Level 2: Every episode (on reset)
    - Level 3: Every 100 episodes
    """

    def __init__(self):
        """Initialize nested level affect."""
        # Affect at each level
        self.levels = np.zeros(4)

        # Update frequencies
        self.update_freqs = [1, 10, 'episode', 100]

        # Step counters
        self.step_count = 0
        self.episode_count = 0

        # Accumulators for level updates
        self.level_accumulators = [[] for _ in range(4)]

    def update(self, emotional_input: float):
        """Update phasic level and accumulate for slower levels.

        Args:
            emotional_input: Current emotional signal
        """
        self.step_count += 1

        # Level 0: Update every step
        self.levels[0] = 0.7 * self.levels[0] + 0.3 * emotional_input
        self.level_accumulators[0].append(emotional_input)

        # Level 1: Update every 10 steps
        if self.step_count % 10 == 0:
            if self.level_accumulators[0]:
                avg = np.mean(self.level_accumulators[0][-10:])
                self.levels[1] = 0.9 * self.levels[1] + 0.1 * avg
                self.level_accumulators[1].append(avg)

    def on_episode_end(self):
        """Called at episode end to update tonic levels.

        Level 2 (mood) updates every episode.
        Level 3 (temperament) updates every 100 episodes.
        """
        self.episode_count += 1

        # Level 2: Update every episode
        if self.level_accumulators[1]:
            avg = np.mean(self.level_accumulators[1])
            self.levels[2] = 0.95 * self.levels[2] + 0.05 * avg
            self.level_accumulators[2].append(avg)

        # Level 3: Update every 100 episodes
        if self.episode_count % 100 == 0:
            if self.level_accumulators[2]:
                avg = np.mean(self.level_accumulators[2][-100:])
                self.levels[3] = 0.99 * self.levels[3] + 0.01 * avg

        # Clear fast accumulators for new episode
        self.level_accumulators[0] = []
        self.level_accumulators[1] = []

    def get_phasic(self) -> float:
        return float(self.levels[0])

    def get_emotional_state(self) -> float:
        return float(self.levels[1])

    def get_mood(self) -> float:
        return float(self.levels[2])

    def get_temperament(self) -> float:
        return float(self.levels[3])

    def get_state(self) -> Dict:
        return {
            'phasic': self.get_phasic(),
            'emotional_state': self.get_emotional_state(),
            'mood': self.get_mood(),
            'temperament': self.get_temperament(),
            'step': self.step_count,
            'episode': self.episode_count
        }


# Example usage
if __name__ == "__main__":
    print("=== ContinuumAffect Demo ===\n")

    affect = ContinuumAffect(n_timescales=5)

    # Simulate emotional episode
    print("Phase 1: Negative event (steps 0-20)")
    for i in range(20):
        affect.update(-1.0, record_history=True)
        if i % 5 == 0:
            print(f"  Step {i}: phasic={affect.get_phasic():.3f}, "
                  f"mood={affect.get_mood():.3f}")

    print("\nPhase 2: Neutral period (steps 20-50)")
    for i in range(30):
        affect.update(0.0, record_history=True)
        if i % 10 == 0:
            print(f"  Step {20+i}: phasic={affect.get_phasic():.3f}, "
                  f"mood={affect.get_mood():.3f}")

    print("\nPhase 3: Positive event (steps 50-70)")
    for i in range(20):
        affect.update(1.0, record_history=True)
        if i % 5 == 0:
            print(f"  Step {50+i}: phasic={affect.get_phasic():.3f}, "
                  f"mood={affect.get_mood():.3f}")

    print(f"\nFinal state: {affect.get_state()}")
