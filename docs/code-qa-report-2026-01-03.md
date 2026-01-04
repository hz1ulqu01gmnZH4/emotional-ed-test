# Code QA Report

**Date:** 2026-01-03
**Files Checked:** 42+
**Verdict:** FAIL

---

## Summary

| Category | Count |
|----------|-------|
| Fallback Violations | 8 |
| Implementation Compromises | 0 |

---

## Fallback Violations (8)

### HIGH Severity (4)

#### 1. scripts/webui.py:67-128
**Pattern:** Exception swallowing - converts errors to strings

```python
except Exception as e:
    return f"Error loading model: {e}"
```

**Fix:** Re-raise with context or let exceptions propagate

```python
except Exception as e:
    raise RuntimeError(f"Failed to load emotional steering model: {e}") from e
```

---

#### 2. agents_temporal.py:140
**Pattern:** Type confusion masking with isinstance fallback

```python
recent_valence = context.cumulative_reward / max(1, len(context.cumulative_reward) if isinstance(context.cumulative_reward, list) else 20)
```

**Fix:** Assert expected type, remove isinstance guard

```python
assert isinstance(context.cumulative_reward, list), \
    f"BUG: cumulative_reward must be list, got {type(context.cumulative_reward)}"
recent_valence = sum(context.cumulative_reward) / max(1, len(context.cumulative_reward))
```

---

#### 3. scripts/compare_models.py:233
**Pattern:** Log and continue on failure

```python
except Exception as e:
    print(f"\n  Error: {e}")
    # Continues to next iteration
```

**Fix:** Raise RuntimeError with context

```python
except Exception as e:
    raise RuntimeError(f"Model comparison failed for {model_name}: {e}") from e
```

---

#### 4. agents_v2/agents_disgust_v2.py:235
**Pattern:** dict.get() with default on reads (borderline)

```python
exposures = self.exposure_count.get(state, 0)
```

**Fix:** Assert state exists if it should have been tracked

```python
assert state in self.exposure_count, f"BUG: State {state} not tracked but expected"
exposures = self.exposure_count[state]
```

---

### MEDIUM Severity (4)

#### 5. test_temporal.py:94-96
**Pattern:** Empty list silently converted to 0

```python
'phase_moods': {p: np.mean(m) if m else 0 for p, m in phase_moods.items()},
'phase_rewards': {p: np.mean(r) if r else 0 for p, r in phase_rewards.items()},
'mood_history': agent.get_mood_history() if hasattr(agent, 'get_mood_history') else []
```

**Fix:** Assert list not empty before mean

```python
assert all(m for m in phase_moods.values()), \
    f"BUG: Empty mood data in phases: {[p for p,m in phase_moods.items() if not m]}"
'phase_moods': {p: np.mean(m) for p, m in phase_moods.items()},
```

---

#### 6. test_statistical_extended.py:372,399
**Pattern:** Empty data defaults to 0.0

```python
phasic_mood.append(np.mean(mood_readings) if mood_readings else 0.0)
tonic_mood.append(np.mean(mood_readings) if mood_readings else 0.0)
```

**Fix:** Assert mood_readings collected

```python
assert mood_readings, f"BUG: No mood readings collected for seed {seed}"
phasic_mood.append(np.mean(mood_readings))
```

---

#### 7. test_sample_efficiency_gpu.py:560,594-595
**Pattern:** Zero variance fallback in Cohen's d

```python
return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0

threat_speedup = np.mean(std_threat_eps) / np.mean(emo_threat_eps) if np.mean(emo_threat_eps) > 0 else 0
reward_speedup = np.mean(std_reward_eps) / np.mean(emo_reward_eps) if np.mean(emo_reward_eps) > 0 else 0
```

**Fix:** Assert pooled_std > 0, fail on zero variance

```python
assert pooled_std > 0, f"BUG: pooled_std is 0 - data has no variance"
return (np.mean(x) - np.mean(y)) / pooled_std

assert np.mean(emo_threat_eps) > 0, "BUG: Emotional agent never reached threat criterion"
threat_speedup = np.mean(std_threat_eps) / np.mean(emo_threat_eps)
```

---

#### 8. exp18_slippery_cliff.py:626
**Pattern:** Same zero variance fallback

```python
return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
```

**Fix:** Assert pooled_std > 0

```python
assert pooled_std > 0, f"BUG: Zero variance in cohens_d calculation"
return (np.mean(group1) - np.mean(group2)) / pooled_std
```

---

## Implementation Compromises (0)

No implementation compromises detected.

### Verified Acceptable Patterns

| Pattern | Status | Reason |
|---------|--------|--------|
| Empty `reset_episode()` methods | ✓ Acceptable | Intentional for baseline agents (documented) |
| Exception classes with `pass` | ✓ Acceptable | Standard Python exception pattern |
| Baseline agents returning `{}` | ✓ Acceptable | Correct "no emotions" behavior |
| `Optional[int]` returning `None` | ✓ Acceptable | Proper type usage |
| Exception handlers that re-raise | ✓ Acceptable | Correct error propagation |

### Additional Checks Passed

- ✓ No TODO/FIXME/HACK comments found
- ✓ No incomplete implementations
- ✓ No logic shortcuts or dead code
- ✓ No validation bypasses
- ✓ No testing compromises (all tests have real assertions)
- ✓ No hardcoded credentials or magic values

---

## Priority Fixes

1. **scripts/webui.py** - Stop swallowing exceptions in UI handlers
2. **agents_temporal.py:140** - Remove type confusion guard
3. **scripts/compare_models.py** - Fail on comparison errors
4. **Statistical functions** - Assert non-zero variance, don't default to 0

---

## Principle

> Code must FAIL LOUDLY when something goes wrong - NEVER silently continue with corrupted or meaningless data.

Returning `0` when variance is zero produces invalid statistics that look valid. Catching exceptions and returning error strings lets the application continue with partial/wrong data. These patterns hide bugs and make debugging difficult.
