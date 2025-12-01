# Nested Learning: The Illusion of Deep Learning Architectures

*Summary of Ali Behrouz et al. (NeurIPS 2025)*
*Paper: https://abehrouz.github.io/files/NL.pdf*

---

## Main Thesis

**Nested Learning (NL)** reconceptualizes neural networks not as static stacks of layers, but as **integrated systems of nested, multi-level, and/or parallel optimization problems**, each operating at different timescales with its own "context flow" and gradient flow.

The authors argue that traditional deep learning architectures (including Transformers) are "illusions" of depth—what appears as deep layers are actually flat systems with components updating at different frequencies.

---

## Key Concepts and Definitions

### 1. Learning vs. Memorization (Neuropsychology)

| Term | Definition |
|------|------------|
| **Memory** | A neural update caused by an input |
| **Learning** | The process for acquiring effective and useful memory |

### 2. Associative Memory (Definition 1)

Given keys K ⊆ ℝ^dk and values V ⊆ ℝ^dv, associative memory is an operator M: K → V that minimizes:

```
M* = argmin_M L̃(M(K); V)
```

**Key insight**: All components of neural networks (including optimizers) are associative memory systems that compress their own context flow.

### 3. Update Frequency (Definition 2)

For any component A, its frequency f_A = number of updates per unit time (one unit = one update over one data point).

### 4. Ordering Operator (≻)

- A ≻ B if f_A > f_B
- Or if equal frequency but B's computation requires A's state
- Components organized into "levels" where higher levels = lower update frequencies

---

## Mathematical Formulations

### Example 1: Simple MLP with Gradient Descent

Training a 1-layer MLP reformulated as:

```
W^(t+1) = argmin_W ⟨Wx_(t+1), u_(t+1)⟩ + (1/2η_(t+1))‖W - W^t‖²₂
```

Where **u_(t+1) = ∇_y L(W^t; x_(t+1))** is the **Local Surprise Signal (LSS)**—the mismatch between current output and the structure enforced by the objective.

### Example 2: Gradient Descent with Momentum → 2-Level Nest

```
W^(t+1) = W^t - m^(t+1)

m^(t+1) = argmin_m -⟨m, ∇L(W^t; x_(t+1))⟩ + η_(t+1)‖m - m^t‖²₂
```

**Revelation**: Momentum-based GD is a 2-level optimization process where momentum compresses gradients into its parameters.

### Example 3: Linear Attention as Associative Memory

```
M^(t+1) = argmin_M ⟨Mk_(t+1), v_(t+1)⟩ + ‖M - M^t‖²₂
```

Linear attention is itself an associative memory optimizing with gradient descent, making the complete system multi-level.

---

## Multi-Timescale Learning & Neuromodulation

### Neurophysiological Motivation

The paper draws from neuroscience:

| Mechanism | Description | Timescale |
|-----------|-------------|-----------|
| **Synaptic consolidation** | Rapid stabilization during wakefulness | Fast (online) |
| **Systems consolidation** | Replay during sleep (hippocampal ripples, cortical spindles) | Slow (offline) |
| **Brain waves** | Different frequencies in different regions | Multi-scale |

### NL Formalizes Multi-Timescale Learning

| Level | Update Frequency | Examples |
|-------|------------------|----------|
| Inner loops (high freq) | Every token/step | Attention mechanisms, momentum |
| Outer loops (low freq) | Per batch/epoch | Feedforward weights, architecture |

This creates a hierarchy of abstractions similar to **Hierarchical Temporal Memory**.

---

## Three Core Contributions

### 1. Deep Optimizers

Extensions to make optimizers more expressive:

**a) Preconditioned Momentum (More Expressive Association):**
```
m^(i+1) = α_(i+1)m^i - η_t P_i ∇L(W^i; x_i)
```
Momentum maps gradients to values P_i (e.g., Hessian information).

**b) Delta-Rule (More Expressive Objectives):**
```
m^(i+1) = [α_(i+1)I - ∇L^T ∇L]m^i - η_t P_i ∇L
```
Using ℓ2 regression instead of dot-product similarity.

**c) MLP Momentum (More Expressive Memory):**
Replace linear momentum with MLP to capture non-linear gradient dynamics.

### 2. Self-Modifying Titans

A sequence model that **learns how to modify itself** by learning its own update algorithm. The learning rule itself becomes learnable.

### 3. Continuum Memory System

Generalizes "long-term/short-term memory" into a continuum. Combined with self-modifying models, creates **HOPE** (the learning module).

---

## Key Insights

### Transformers Are Not Truly Deep

| Component | Update Frequency | Behavior |
|-----------|------------------|----------|
| Attention | Every token | High-frequency, dynamic |
| Feedforward | Only during training | Low-frequency, static |

This explains why LLMs are **static after deployment**—a form of "anterograde amnesia."

### Optimizers Are Memories

Well-known optimizers (Adam, SGD+Momentum) are **associative memory modules** that compress gradients using gradient descent.

### In-Context Learning Explained

ICL emerges through compression of context flows at different timescales—attention compresses recent context while feedforward stores pre-training knowledge.

---

## Implications for Neural Network Design

### 1. New Dimension: Levels vs Layers

Instead of stacking layers (traditional depth), add more **levels of optimization**:
- Higher-order in-context learning
- Better continual learning
- More expressive learning algorithms

### 2. Addressing LLM Limitations

NL provides framework to overcome:
- ✗ Catastrophic forgetting
- ✗ Cannot form new long-term memories
- ✗ Limited adaptation beyond context window
- ✗ Static knowledge after deployment

### 3. Connection to Fast Weight Programs

| Weight Type | Loop | Example |
|-------------|------|---------|
| Slow weights | Outer loop | Feedforward layers |
| Fast weights | Inner loop | Attention, momentum |

### 4. Neuroscientifically Plausible

Aligns with biological learning:
- Multi-timescale processing (brain waves)
- Memory consolidation (online/offline)
- Neuroplasticity (continuous adaptation)

---

## Relevance to Emotional ED Architecture

The Nested Learning framework has direct implications for the Emotional Error Diffusion project:

### 1. Emotions as Different Optimization Levels

| Emotion | Timescale | NL Interpretation |
|---------|-----------|-------------------|
| Fear (phasic) | Fast | Inner loop, high-frequency modulation |
| Mood (tonic) | Slow | Outer loop, low-frequency baseline |
| Grief/Yearning | Very slow | Slowest level, attachment dynamics |

### 2. Learning Rate Modulation = Level Frequency

The ED architecture's learning rate modulation maps to NL's update frequency:
```
effective_lr = base_lr × fear_modulation × anger_modulation
```
This is equivalent to changing the **update frequency** of the associative memory.

### 3. Multi-Timescale Affect Implementation

Following NL, implement emotions as nested optimization problems:

```python
# Level 0: Fastest (phasic emotions)
fear_t = update_fear(context)  # Every step

# Level 1: Medium (emotional state)
mood_t = update_mood(fear_history)  # Every N steps

# Level 2: Slowest (temperament/attachment)
temperament = update_temperament(mood_history)  # Every episode
```

### 4. Local Surprise Signal (LSS) as Emotional Trigger

The paper's "Local Surprise Signal" (u = ∇_y L) maps directly to emotional prediction error:
- **Fear**: Triggered by negative LSS near threats
- **Anger**: Triggered by blocked goal LSS
- **Regret**: Counterfactual LSS (what could have been)

### 5. Self-Modifying Emotional Weights

Following "Self-Modifying Titans," emotional channel weights could be learned:
```python
# Instead of fixed weights
fear_weight = 0.5  # Static

# Learn the weights as an inner optimization
fear_weight_t = learn_weight(emotional_context, outcome)
```

---

## Practical Implementation Suggestions

### From NL to Emotional ED

1. **Formalize emotions as nested optimization levels** with explicit frequencies
2. **Use LSS (prediction error) as emotional trigger** instead of ad-hoc distance functions
3. **Implement continuum memory** for affect (not just binary phasic/tonic)
4. **Make emotional weights learnable** via meta-gradient or self-modifying rules
5. **Add systems consolidation** (offline replay weighted by emotional salience)

### Mathematical Reformulation

Current ED:
```
Q^(t+1) = Q^t + α × m_fear × m_anger × δ
```

NL-inspired ED:
```
# Level 0: Value update (every step)
Q^(t+1) = argmin_Q ⟨Qs, δ⟩ + (1/2η_Q)‖Q - Q^t‖²

# Level 1: Emotional modulation (every step, different objective)
m^(t+1) = argmin_m L_emotion(m, LSS) + (1/2η_m)‖m - m^t‖²

# Level 2: Mood/temperament (every N steps)
T^(t+1) = argmin_T L_mood(T, m_history) + (1/2η_T)‖T - T^t‖²
```

---

## References

- Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V. (2025). Nested Learning: The Illusion of Deep Learning Architectures. *NeurIPS 2025*.
- Related: Titans architecture, HOPE module, Fast Weight Programs

---

*Summary generated: December 2025*
