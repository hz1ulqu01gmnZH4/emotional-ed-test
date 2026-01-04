# Experiment 19: Activation Steering for Emotional LLMs

**Date**: 2026-01-02
**Status**: Complete
**Result**: Positive - Emotional steering successfully modulates LLM outputs

## Abstract

This experiment implements **Activation Steering** - a technique for adding learned "emotional direction vectors" to LLM hidden states at inference time. Unlike fine-tuning approaches, the base LLM weights remain frozen; emotional modulation is achieved by adding steering vectors to intermediate layer activations.

Key findings:
- Learned directions achieve 95-100% consistency in separating emotional from neutral responses
- Steering effects are coherent at scale ≤ 0.5 with intensity ≤ 0.6
- Higher steering magnitudes cause output degradation (expected behavior)
- The approach requires ~0.01% additional parameters (direction vectors only)

## Background

### Motivation

Previous experiments (Exp 1-18) demonstrated emotional modulation in RL agents using:
- Gradient blocking to prevent emotional signal interference
- State augmentation (emotions as input features)
- Temporal dynamics (phasic + tonic emotional states)

The question: **Can we integrate these emotional mechanisms into LLMs while keeping weights frozen?**

### Approach Selection

Five integration approaches were documented (see `docs/llm-integration/`):

| Approach | Parameters | Complexity | Selected |
|----------|------------|------------|----------|
| Emotional Prefix Tuning | ~0.1% | Medium | No |
| Adapter Emotional Gating | ~1-5% | High | No |
| **Activation Steering** | **~0.01%** | **Low** | **Yes** |
| External Emotional Memory | Variable | High | No |
| Emotional Reward Model | ~10-50% | Very High | No |

Activation Steering was selected for initial implementation due to:
1. Minimal additional parameters
2. Simple architecture (direction vectors + forward hooks)
3. Clear theoretical foundation (difference-in-means)
4. No training loop required (direct computation)

## Method

### Architecture

```
┌─────────────────────────────────────────┐
│ Emotional Direction Bank                │
│ ├── fear: [n_layers, hidden_dim]        │
│ ├── curiosity: [n_layers, hidden_dim]   │
│ ├── anger: [n_layers, hidden_dim]       │
│ └── joy: [n_layers, hidden_dim]         │
└────────────────┬────────────────────────┘
                 │
                 ▼
    steering = Σ(intensity × direction)
                 │
                 ▼
┌─────────────────────────────────────────┐
│ FROZEN LLM (Qwen2.5-1.5B-Instruct)      │
│ Layer i: hidden_states += steering[i]   │
└─────────────────────────────────────────┘
```

### Direction Learning Algorithm

**Difference-in-Means** method:

```python
direction[layer] = mean(emotional_activations) - mean(neutral_activations)
direction[layer] = normalize(direction[layer])
```

For each emotion:
1. Collect (neutral, emotional) response pairs
2. Extract hidden states at each layer for both responses
3. Compute mean activation for neutral and emotional sets
4. Direction = emotional_mean - neutral_mean
5. Normalize to unit length

### Steering Mechanism

PyTorch forward hooks intercept layer outputs:

```python
def steering_hook(module, input, output):
    hidden_states = output[0]
    steering = direction_bank.get_combined_steering(emotional_state, layer_idx)
    hidden_states = hidden_states + (steering * scale)
    return (hidden_states,) + output[1:]
```

### Dataset Generation

Contrastive pairs generated via parallel subagents:

| Emotion | Pairs | Example Neutral | Example Emotional |
|---------|-------|-----------------|-------------------|
| Fear | 20 | "The cliff is 100m high." | "I'm concerned about safety - the cliff is 100m high. Please be careful." |
| Curiosity | 20 | "Python was released in 1991." | "Interesting! I wonder what inspired the design choices. Could you tell me more?" |
| Anger | 20 | "The query timed out." | "Let me try another approach - there must be a way to fix this." |
| Joy | 20 | "Your code passed tests." | "Excellent! Congratulations on getting everything working!" |

## Implementation

### Package Structure

```
src/llm_emotional/
├── __init__.py
├── steering/
│   ├── direction_bank.py      # Direction storage/persistence
│   ├── direction_learner.py   # Difference-in-means learning
│   ├── steering_hooks.py      # Forward hook implementation
│   └── emotional_llm.py       # Main wrapper class
├── emotions/
│   ├── context_computer.py    # Context → emotional state
│   └── datasets.py            # Dataset loading (no fallback)
```

### Key Design Decisions

1. **No Fallback Policy**: All operations fail loudly on error
   - `DatasetNotFoundError` if dataset missing
   - `DirectionBankError` on unknown emotions
   - `ModelLoadError` if model fails to load

2. **Emotion Validation**: Unknown emotions in `set_emotional_state()` raise errors
   - Catches typos like `feer` instead of `fear`
   - Enforced after code-qa review

3. **Qwen2.5-1.5B-Instruct**: Selected for balance of capability and efficiency
   - 1536 hidden dimensions
   - 28 transformer layers
   - Fits in GPU memory for rapid iteration

## Results

### Direction Quality Metrics

| Emotion | Consistency | Separation | Direction Norm |
|---------|-------------|------------|----------------|
| Fear | 100.0% | 0.0316 | 1.0000 |
| Curiosity | 95.0% | 0.0261 | 1.0000 |
| Anger | 100.0% | 0.0341 | 1.0000 |
| Joy | 100.0% | 0.0370 | 0.9961 |

**Consistency**: Percentage of pairs where emotional projection > neutral projection
**Separation**: Mean difference in projection values (emotional - neutral)

### Steering Effect on Output

**Prompt**: "Tell me about skydiving."

| State | Response Characteristics |
|-------|-------------------------|
| Neutral | Factual description of skydiving mechanics |
| Fearful | Emphasizes "thrilling experience", "high altitudes" |
| Curious | Explores deeper, invites follow-up |
| Determined | Frames as "popular extreme sport", action-oriented |
| Joyful | "Thrilling and exhilarating", positive framing |

### Steering Magnitude Analysis

| Scale | Intensity | Coherence | Notes |
|-------|-----------|-----------|-------|
| 0.5 | 0.6 | ✓ Coherent | Subtle but measurable differences |
| 1.0 | 0.8 | ✓ Coherent | More pronounced emotional tone |
| 2.0 | 0.8 | ~ Degraded | Some grammatical issues |
| 3.0 | 0.8 | ✗ Broken | Output becomes incoherent |

**Finding**: Effective steering requires scale × intensity < 1.0 for coherent output.

## Test Coverage

70 unit tests covering:

| Module | Tests | Coverage |
|--------|-------|----------|
| direction_bank.py | 19 | Initialization, set/get, persistence |
| steering_hooks.py | 14 | Hook behavior, manager coordination |
| datasets.py | 14 | Validation, no-fallback enforcement |
| context_computer.py | 23 | Phasic/tonic emotions, edge cases |

All tests pass:
```
========================= 70 passed in 0.89s =========================
```

## Code Quality

Code-QA analysis (no-fallback-enforcer + compromise-checker):

- **Initial scan**: 1 violation found (silent skip of unknown emotions)
- **After fix**: 0 violations, 0 compromises
- **Verdict**: PASS

## Limitations

1. **Subtle Effects**: Steering produces subtle changes, not dramatic personality shifts
2. **Scale Sensitivity**: High steering magnitudes break coherence
3. **Dataset Dependence**: Direction quality depends on contrastive pair quality
4. **Layer Selection**: Currently steers all layers; selective steering may be more effective

## Future Work

1. **Selective Layer Steering**: Identify which layers are most effective for each emotion
2. **Activation Patching**: Alternative to difference-in-means for direction learning
3. **Larger Models**: Test on Qwen3-4B, Llama-3.2-7B
4. **Context Computer Integration**: Dynamic emotional state from conversation history
5. **Comparison Study**: Benchmark against prompt-based emotional control

## Conclusion

Activation steering successfully modulates LLM outputs with emotional characteristics while keeping base weights frozen. The approach is lightweight (~0.01% parameters), achieves high consistency (95-100%), and integrates cleanly with the existing Emotional-ED framework patterns.

The key insight from RL experiments—using emotions as state augmentation rather than gradient signals—translates directly: emotional directions are additive modifications to hidden states, not changes to the underlying model.

## Artifacts

| File | Description |
|------|-------------|
| `src/llm_emotional/` | Implementation package |
| `data/emotional_pairs.json` | 80 contrastive training pairs |
| `data/direction_bank.json` | Learned direction vectors |
| `scripts/train_directions.py` | Training script |
| `scripts/demo_steering.py` | Demo script |
| `tests/llm_emotional/` | 70 unit tests |

## References

1. Turner et al. (2023). "Activation Addition: Steering Language Models Without Optimization"
2. Li et al. (2024). "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"
3. Zou et al. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency"
4. Project experiments: `EXPERIMENT_PLAN_V3.md`, `exp1-18` results

---

**Experiment conducted by**: Claude Code
**Model**: Qwen2.5-1.5B-Instruct
**Compute**: CUDA GPU
**Training time**: ~3 minutes (direction learning only)
