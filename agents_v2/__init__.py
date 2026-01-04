"""Improved agents for Emotional Error Diffusion (v2).

This module contains architectural fixes based on feedback from
GPT-5, Gemini, and Grok-4:

1. agents_disgust_v2: Directional repulsion (not argmax boost)
2. agents_feature_based: Feature-based Q for Transfer generalization
3. agents_regulation_v2: Bayesian reappraisal with credit assignment fix
4. agents_cvar_fear: Distributional RL with CVaR for principled risk-sensitivity

Key insights addressed:
- Mid-learning insight: Effects masked at convergence
- Transfer failure: Tabular Q can't generalize; need features
- Disgust reversal: Boosting argmax can boost wrong action
- Regulation reversal: Credit assignment + environment design
- Shaping vs Objective: CVaR changes objective, not just shaping
"""

from .agents_disgust_v2 import (
    DisgustOnlyAgentV2,
    FearOnlyAgentV2,
    IntegratedFearDisgustAgentV2,
)

from .agents_feature_based import (
    FeatureBasedFearAgent,
    FeatureBasedTransferAgent,
    TabularBaselineAgent,
    FeatureContext,
)

from .agents_regulation_v2 import (
    RegulatedFearAgentV2,
    UnregulatedFearAgentV2,
    BayesianReappraisalModule,
    RegulationGridWorldV2,
    RegulationContext,
)

from .agents_cvar_fear import (
    CVaRFearAgent,
    RiskNeutralAgent,
    AdaptiveCVaRFearAgent,
    HybridFearAgent,
    QuantileDistribution,
    FearContext,
)

__all__ = [
    # Disgust v2
    'DisgustOnlyAgentV2',
    'FearOnlyAgentV2',
    'IntegratedFearDisgustAgentV2',
    # Feature-based
    'FeatureBasedFearAgent',
    'FeatureBasedTransferAgent',
    'TabularBaselineAgent',
    'FeatureContext',
    # Regulation v2
    'RegulatedFearAgentV2',
    'UnregulatedFearAgentV2',
    'BayesianReappraisalModule',
    'RegulationGridWorldV2',
    'RegulationContext',
    # CVaR Fear
    'CVaRFearAgent',
    'RiskNeutralAgent',
    'AdaptiveCVaRFearAgent',
    'HybridFearAgent',
    'QuantileDistribution',
    'FearContext',
]
