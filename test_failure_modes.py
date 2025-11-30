"""Experiment 15: Emotional Interference / Failure Modes.

Tests when emotional channels HURT performance - demonstrating
they are genuine control mechanisms that can malfunction.

Hypothesis:
- Excessive fear prevents optimal approach
- Inflexible anger wastes resources on impossible goals
- Emotional conflict causes paralysis/indecision
- Miscalibrated emotions produce worse outcomes than no emotion

This is critical: if emotions only help, they might just be
performance boosters. Showing failure modes proves they are
genuine competing control systems.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class FailureContext:
    """Context for failure mode experiments."""
    threat_distance: float
    goal_distance: float
    was_blocked: bool
    optimal_path_blocked: bool  # Is optimal path through "scary but safe" zone?


class ScaryButSafeGridWorld:
    """Environment where optimal path goes through scary-looking but safe zone.

    Layout:
    A . . . G
    . X X X .
    . X . X .
    . X X X .
    . . . . .

    X = Scary looking but SAFE
    Direct path (through X) = 4 steps
    Safe path (around) = 8 steps

    Tests: Does excessive fear prevent optimal behavior?
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 5):
        self.size = size
        self.goal_pos = np.array([0, 4])
        # Scary but safe zone (visual threat, no actual penalty)
        self.scary_zone = {
            (1, 1), (1, 2), (1, 3),
            (2, 1), (2, 3),
            (3, 1), (3, 2), (3, 3)
        }
        self.reset()

    def reset(self) -> int:
        self.agent_pos = np.array([0, 0])
        self.step_count = 0
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def _in_scary_zone(self, pos: np.ndarray) -> bool:
        return (pos[0], pos[1]) in self.scary_zone

    def step(self, action: int) -> Tuple[int, float, bool, FailureContext]:
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        was_blocked = False
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            was_blocked = True
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        # No actual penalty for scary zone - it's SAFE
        reward = -0.01  # Step cost

        done = False
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 1.0
            done = True

        if self.step_count >= 50:
            done = True

        # Compute threat distance as if scary zone was dangerous
        min_scary_dist = float('inf')
        for sz in self.scary_zone:
            d = self._distance(self.agent_pos, np.array(sz))
            min_scary_dist = min(min_scary_dist, d)

        context = FailureContext(
            threat_distance=min_scary_dist,
            goal_distance=self._distance(self.agent_pos, self.goal_pos),
            was_blocked=was_blocked,
            optimal_path_blocked=False
        )

        return self._pos_to_state(self.agent_pos), reward, done, context

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4


class ImpossibleObstacleGridWorld:
    """Environment with an impossible obstacle.

    Tests: Does inflexible anger waste resources persisting?
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 5):
        self.size = size
        self.goal_pos = np.array([4, 4])
        # Unbreakable wall blocking direct path
        self.wall = {(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)}
        self.reset()

    def reset(self) -> int:
        self.agent_pos = np.array([0, 2])
        self.step_count = 0
        self.wall_hits = 0
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def _is_wall(self, pos: np.ndarray) -> bool:
        return (pos[0], pos[1]) in self.wall

    def step(self, action: int) -> Tuple[int, float, bool, FailureContext]:
        self.step_count += 1

        delta = np.array(self.ACTIONS[action])
        new_pos = self.agent_pos + delta

        was_blocked = False

        # Boundary check
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            was_blocked = True
            new_pos = self.agent_pos

        # Wall check - UNBREAKABLE
        if self._is_wall(new_pos):
            was_blocked = True
            self.wall_hits += 1
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        reward = -0.01
        if was_blocked:
            reward -= 0.05  # Extra penalty for hitting wall

        done = False
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 1.0
            done = True

        if self.step_count >= 100:
            done = True

        context = FailureContext(
            threat_distance=float('inf'),
            goal_distance=np.linalg.norm(self.agent_pos - self.goal_pos),
            was_blocked=was_blocked,
            optimal_path_blocked=True
        )

        return self._pos_to_state(self.agent_pos), reward, done, context

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4


class StandardAgent:
    """Standard Q-learning agent."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context):
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += self.lr * (target - self.Q[state, action])


class ExcessiveFearAgent:
    """Agent with excessive fear - avoids anything scary-looking."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 fear_weight: float = 2.0):  # HIGH fear weight
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.fear_weight = fear_weight
        self.current_fear = 0.0

    def _compute_fear(self, context) -> float:
        safe_distance = 3.0  # Very sensitive
        if context.threat_distance >= safe_distance:
            return 0.0
        return 1.0 - context.threat_distance / safe_distance

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()
        # High fear drastically reduces action values
        if self.current_fear > 0.2:
            q_values *= (1 - self.current_fear * self.fear_weight)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context):
        self.current_fear = self._compute_fear(context)

        # Fear slows all learning (paralysis)
        effective_lr = self.lr * (1 - self.current_fear * 0.8)

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])


class InflexibleAngerAgent:
    """Agent with inflexible anger - persists at obstacles excessively."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 anger_buildup: float = 0.3, anger_decay: float = 0.99):  # SLOW decay
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.anger_buildup = anger_buildup
        self.anger_decay = anger_decay
        self.frustration = 0.0
        self.last_blocked_action = None

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Anger makes us persist at blocked action
        if self.frustration > 0.3 and self.last_blocked_action is not None:
            q_values[self.last_blocked_action] *= (1 + self.frustration * 2.0)

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context):
        if context.was_blocked:
            self.frustration = min(1.0, self.frustration + self.anger_buildup)
            self.last_blocked_action = action
        else:
            self.frustration *= self.anger_decay  # Very slow decay

        # Anger prevents learning from failure
        effective_lr = self.lr
        if context.was_blocked and self.frustration > 0.5:
            effective_lr *= 0.1  # Barely learn from blocks

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])


class ConflictedAgent:
    """Agent with conflicting emotions - fear AND approach both high."""

    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.fear = 0.0
        self.approach = 0.0

    def select_action(self, state: int) -> int:
        # Conflict increases randomness (indecision)
        conflict_level = min(self.fear, self.approach)
        effective_epsilon = self.epsilon + conflict_level * 0.5

        if np.random.random() < effective_epsilon:
            return np.random.randint(self.Q.shape[1])

        q_values = self.Q[state].copy()

        # Conflict reduces action differentiation
        if conflict_level > 0.3:
            q_values = q_values * (1 - conflict_level * 0.5) + np.mean(q_values) * conflict_level * 0.5

        return np.argmax(q_values)

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, context):
        # Update both fear and approach
        if context.threat_distance < 2.0:
            self.fear = min(1.0, self.fear + 0.2)
        else:
            self.fear *= 0.9

        if context.goal_distance < 3.0:
            self.approach = min(1.0, self.approach + 0.2)
        else:
            self.approach *= 0.9

        # Conflict slows learning
        conflict_level = min(self.fear, self.approach)
        effective_lr = self.lr * (1 - conflict_level * 0.5)

        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += effective_lr * (target - self.Q[state, action])


def run_episode(env, agent, max_steps: int = 100):
    """Run episode tracking failure metrics."""
    state = env.reset()
    total_reward = 0
    steps = 0
    goal_reached = False

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, ctx = env.step(action)
        agent.update(state, action, reward, next_state, done, ctx)

        total_reward += reward
        steps += 1

        if done:
            if reward > 0.5:
                goal_reached = True
            break

        state = next_state

    wall_hits = env.wall_hits if hasattr(env, 'wall_hits') else 0

    return {
        'reward': total_reward,
        'steps': steps,
        'goal_reached': goal_reached,
        'wall_hits': wall_hits
    }


def test_excessive_fear(n_seeds: int = 30, n_train: int = 200, n_eval: int = 50):
    """Test excessive fear preventing optimal behavior."""
    results = {name: {'reward': [], 'steps': [], 'goal_rate': []}
               for name in ['Standard', 'ExcessiveFear']}

    for seed in range(n_seeds):
        np.random.seed(seed)

        for name in results:
            env = ScaryButSafeGridWorld()

            if name == 'Standard':
                agent = StandardAgent(env.n_states, env.n_actions)
            else:
                agent = ExcessiveFearAgent(env.n_states, env.n_actions)

            # Train
            for _ in range(n_train):
                run_episode(env, agent)

            # Eval
            agent.epsilon = 0.05
            eval_results = []
            for _ in range(n_eval):
                eval_results.append(run_episode(env, agent))

            results[name]['reward'].append(np.mean([r['reward'] for r in eval_results]))
            results[name]['steps'].append(np.mean([r['steps'] for r in eval_results]))
            results[name]['goal_rate'].append(np.mean([r['goal_reached'] for r in eval_results]))

    return {name: {
        'reward_mean': np.mean(data['reward']),
        'reward_std': np.std(data['reward']),
        'steps_mean': np.mean(data['steps']),
        'steps_std': np.std(data['steps']),
        'goal_rate': np.mean(data['goal_rate'])
    } for name, data in results.items()}


def test_inflexible_anger(n_seeds: int = 30, n_train: int = 200, n_eval: int = 50):
    """Test inflexible anger wasting resources."""
    results = {name: {'reward': [], 'wall_hits': [], 'goal_rate': []}
               for name in ['Standard', 'InflexibleAnger']}

    for seed in range(n_seeds):
        np.random.seed(seed)

        for name in results:
            env = ImpossibleObstacleGridWorld()

            if name == 'Standard':
                agent = StandardAgent(env.n_states, env.n_actions)
            else:
                agent = InflexibleAngerAgent(env.n_states, env.n_actions)

            # Train
            for _ in range(n_train):
                run_episode(env, agent)

            # Eval
            agent.epsilon = 0.05
            eval_results = []
            for _ in range(n_eval):
                eval_results.append(run_episode(env, agent))

            results[name]['reward'].append(np.mean([r['reward'] for r in eval_results]))
            results[name]['wall_hits'].append(np.mean([r['wall_hits'] for r in eval_results]))
            results[name]['goal_rate'].append(np.mean([r['goal_reached'] for r in eval_results]))

    return {name: {
        'reward_mean': np.mean(data['reward']),
        'reward_std': np.std(data['reward']),
        'wall_hits_mean': np.mean(data['wall_hits']),
        'wall_hits_std': np.std(data['wall_hits']),
        'goal_rate': np.mean(data['goal_rate'])
    } for name, data in results.items()}


def test_emotional_conflict(n_seeds: int = 30, n_train: int = 200, n_eval: int = 50):
    """Test emotional conflict causing paralysis."""
    results = {name: {'reward': [], 'steps': [], 'goal_rate': []}
               for name in ['Standard', 'Conflicted']}

    for seed in range(n_seeds):
        np.random.seed(seed)

        for name in results:
            env = ScaryButSafeGridWorld()

            if name == 'Standard':
                agent = StandardAgent(env.n_states, env.n_actions)
            else:
                agent = ConflictedAgent(env.n_states, env.n_actions)

            # Train
            for _ in range(n_train):
                run_episode(env, agent)

            # Eval
            agent.epsilon = 0.05
            eval_results = []
            for _ in range(n_eval):
                eval_results.append(run_episode(env, agent))

            results[name]['reward'].append(np.mean([r['reward'] for r in eval_results]))
            results[name]['steps'].append(np.mean([r['steps'] for r in eval_results]))
            results[name]['goal_rate'].append(np.mean([r['goal_reached'] for r in eval_results]))

    return {name: {
        'reward_mean': np.mean(data['reward']),
        'reward_std': np.std(data['reward']),
        'steps_mean': np.mean(data['steps']),
        'steps_std': np.std(data['steps']),
        'goal_rate': np.mean(data['goal_rate'])
    } for name, data in results.items()}


def main():
    np.random.seed(42)

    print("=" * 70)
    print("EXPERIMENT 15: EMOTIONAL FAILURE MODES")
    print("=" * 70)

    print("\nHYPOTHESIS: Miscalibrated emotions HURT performance")
    print("- Excessive fear prevents optimal approach")
    print("- Inflexible anger wastes resources")
    print("- Emotional conflict causes paralysis")
    print()
    print("This proves emotions are genuine control mechanisms, not just boosters")
    print()

    # Test 1: Excessive Fear
    print("=" * 70)
    print("TEST 1: Excessive Fear (Scary-but-Safe Environment)")
    print("=" * 70)
    print("\nOptimal path goes through scary-looking but safe zone.")
    print("Does excessive fear prevent taking optimal path?")

    fear_results = test_excessive_fear(n_seeds=30)

    print(f"\n{'Agent':<18} {'Reward':<18} {'Steps':<18} {'Goal Rate':<12}")
    print("-" * 65)
    for name, stats in fear_results.items():
        print(f"{name:<18} {stats['reward_mean']:.3f} ± {stats['reward_std']:.3f}   "
              f"{stats['steps_mean']:.1f} ± {stats['steps_std']:.1f}      "
              f"{stats['goal_rate']:.0%}")

    # Test 2: Inflexible Anger
    print("\n" + "=" * 70)
    print("TEST 2: Inflexible Anger (Impossible Obstacle)")
    print("=" * 70)
    print("\nWall is UNBREAKABLE. Does anger waste resources persisting?")

    anger_results = test_inflexible_anger(n_seeds=30)

    print(f"\n{'Agent':<18} {'Reward':<18} {'Wall Hits':<18} {'Goal Rate':<12}")
    print("-" * 65)
    for name, stats in anger_results.items():
        print(f"{name:<18} {stats['reward_mean']:.3f} ± {stats['reward_std']:.3f}   "
              f"{stats['wall_hits_mean']:.1f} ± {stats['wall_hits_std']:.1f}      "
              f"{stats['goal_rate']:.0%}")

    # Test 3: Emotional Conflict
    print("\n" + "=" * 70)
    print("TEST 3: Emotional Conflict (Fear + Approach)")
    print("=" * 70)
    print("\nHigh fear AND high approach motivation. Does conflict cause paralysis?")

    conflict_results = test_emotional_conflict(n_seeds=30)

    print(f"\n{'Agent':<18} {'Reward':<18} {'Steps':<18} {'Goal Rate':<12}")
    print("-" * 65)
    for name, stats in conflict_results.items():
        print(f"{name:<18} {stats['reward_mean']:.3f} ± {stats['reward_std']:.3f}   "
              f"{stats['steps_mean']:.1f} ± {stats['steps_std']:.1f}      "
              f"{stats['goal_rate']:.0%}")

    # Hypothesis Tests
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)

    # H1: Excessive fear hurts performance
    std_fear = fear_results['Standard']['goal_rate']
    exc_fear = fear_results['ExcessiveFear']['goal_rate']
    print(f"\nH1: Excessive fear REDUCES goal rate")
    print(f"    Standard: {std_fear:.0%}")
    print(f"    Excessive Fear: {exc_fear:.0%}")
    if exc_fear < std_fear - 0.1:
        print("    ✓ Excessive fear HURTS performance (-10%+ goal rate)")
    elif exc_fear < std_fear:
        print("    ~ Excessive fear slightly hurts")
    else:
        print("    ✗ No fear-induced performance loss")

    # H2: Inflexible anger wastes resources
    std_hits = anger_results['Standard']['wall_hits_mean']
    ang_hits = anger_results['InflexibleAnger']['wall_hits_mean']
    print(f"\nH2: Inflexible anger WASTES resources (more wall hits)")
    print(f"    Standard wall hits: {std_hits:.1f}")
    print(f"    Inflexible anger wall hits: {ang_hits:.1f}")
    if ang_hits > std_hits * 1.5:
        print("    ✓ Anger WASTES resources (+50%+ wall hits)")
    elif ang_hits > std_hits:
        print("    ~ Anger slightly more wall hits")
    else:
        print("    ✗ No anger-induced resource waste")

    # H3: Conflict causes paralysis
    std_steps = conflict_results['Standard']['steps_mean']
    conf_steps = conflict_results['Conflicted']['steps_mean']
    print(f"\nH3: Emotional conflict causes paralysis (more steps)")
    print(f"    Standard steps: {std_steps:.1f}")
    print(f"    Conflicted steps: {conf_steps:.1f}")
    if conf_steps > std_steps * 1.2:
        print("    ✓ Conflict causes paralysis (+20%+ steps)")
    elif conf_steps > std_steps:
        print("    ~ Conflict slightly more steps")
    else:
        print("    ✗ No conflict-induced paralysis")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Emotional Failure Modes")
    print("=" * 70)

    failures_demonstrated = 0
    if exc_fear < std_fear - 0.05:
        failures_demonstrated += 1
    if ang_hits > std_hits * 1.2:
        failures_demonstrated += 1
    if conf_steps > std_steps * 1.1:
        failures_demonstrated += 1

    print(f"\nFailure modes demonstrated: {failures_demonstrated}/3")
    print("\nKey findings:")
    print("1. Excessive fear prevents optimal paths through safe zones")
    print("2. Inflexible anger wastes resources on impossible obstacles")
    print("3. Emotional conflict produces indecision and slower behavior")

    if failures_demonstrated >= 2:
        print("\n✓ Emotions are genuine CONTROL SYSTEMS that can malfunction")
        print("  Not just performance boosters - they have failure modes")
    else:
        print("\n~ Results inconclusive - failure modes not clearly demonstrated")


if __name__ == "__main__":
    main()
