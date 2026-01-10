"""
Unified Falsification Experiment Runner.

Runs all hypotheses from V3 Falsification Protocol V2:
- H1: Geometric Encoding (functional vs lexical)
- H2: Robustness (matched-magnitude controls)
- H3: Wanting-Liking Dissociation (behavioral)
- H4: Intrinsic Temporal Dynamics (no external decay)
- H5: Cross-Modal Functional Transfer
- H6: Iowa Gambling Task
- H7: Internal Metrics (dimensionality, entropy)
"""

import json
import statistics
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Callable

import torch

from .metrics import (
    EmotionClassifier,
    InternalMetricsMeasurer,
    FunctionalMetricsMeasurer,
    compute_effect_size,
)
from .behavioral_tasks import (
    IowaGamblingTask,
    BinaryChoiceTask,
    WantingLikingTask,
    IntrinsicPersistenceTest,
    RobustnessTest,
)


@dataclass
class HypothesisResult:
    """Result for a single hypothesis test."""
    hypothesis: str
    supported: bool
    effect_size: float
    p_value: Optional[float]
    details: dict
    interpretation: str


@dataclass
class FalsificationResults:
    """Complete results from all hypothesis tests."""
    model_name: str
    timestamp: str
    hypotheses: list[HypothesisResult]
    summary: dict
    raw_data: dict


class FalsificationRunner:
    """
    Runs the complete V3 Falsification Protocol.

    Requires an LLM wrapper with:
    - generate(prompt) -> str
    - set_emotional_state(**emotions)
    - clear_emotional_state()
    - get_steering_vector(name) -> Tensor
    - model (for internal metrics)
    """

    def __init__(
        self,
        llm,  # EmotionalSteeringLLMv3 or compatible
        output_dir: str = "results/falsification",
        n_trials_per_test: int = 50,  # Reduced from 500 for practical runtime
        verbose: bool = True,
    ):
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_trials = n_trials_per_test
        self.verbose = verbose

        # Initialize measurement tools
        self.classifier = EmotionClassifier()
        self.internal_measurer = InternalMetricsMeasurer()
        self.functional_measurer = FunctionalMetricsMeasurer()

        # Results storage
        self.results: list[HypothesisResult] = []
        self.raw_data: dict = {}

    def log(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            print(msg)

    def _generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Wrapper for LLM generation."""
        return self.llm.generate_completion(prompt, max_new_tokens=max_tokens)

    # =========================================================================
    # H1: Geometric Encoding
    # =========================================================================
    def run_h1_geometric(self) -> HypothesisResult:
        """
        Test H1: Emotions exist as geometric directions that generalize.

        Success: Fear steering produces functional changes across diverse contexts.
        """
        self.log("\n[H1] Testing Geometric Encoding...")

        contexts = [
            ("narrative", "Tell a short story about a character."),
            ("factual", "Explain how photosynthesis works."),
            ("recipe", "Write a recipe for making soup."),
            ("advice", "Give advice for a job interview."),
        ]

        emotions = ["fear", "joy", "anger"]
        results_by_context = {}

        for context_name, prompt in contexts:
            results_by_context[context_name] = {}

            # Baseline
            self.llm.clear_emotional_state()
            baseline = self._generate(prompt)
            baseline_scores = self.classifier.classify(baseline)

            for emotion in emotions:
                self.llm.set_emotional_state(**{emotion: 0.8})
                steered = self._generate(prompt)
                steered_scores = self.classifier.classify(steered)

                # Measure functional vs surface
                func_metrics = self.functional_measurer.measure(steered, emotion)

                results_by_context[context_name][emotion] = {
                    "baseline_score": baseline_scores.classifier_scores.get(emotion, 0),
                    "steered_score": steered_scores.classifier_scores.get(emotion, 0),
                    "delta": steered_scores.classifier_scores.get(emotion, 0) -
                             baseline_scores.classifier_scores.get(emotion, 0),
                    "functional_score": func_metrics.functional_score,
                    "surface_score": func_metrics.surface_score,
                }

        self.llm.clear_emotional_state()

        # Analyze: Does steering generalize across contexts?
        generalization_scores = []
        for emotion in emotions:
            deltas = [results_by_context[ctx][emotion]["delta"] for ctx in results_by_context]
            mean_delta = statistics.mean(deltas)
            generalization_scores.append(mean_delta)

        mean_generalization = statistics.mean(generalization_scores)
        supported = mean_generalization > 0.05  # Threshold for detectable effect

        self.raw_data["h1"] = results_by_context

        return HypothesisResult(
            hypothesis="H1: Geometric Encoding",
            supported=supported,
            effect_size=mean_generalization,
            p_value=None,  # Would need more samples for proper test
            details=results_by_context,
            interpretation="SUPPORTS geometric encoding" if supported else "WEAK/NO geometric encoding",
        )

    # =========================================================================
    # H2: Robustness
    # =========================================================================
    def run_h2_robustness(self) -> HypothesisResult:
        """
        Test H2: Emotion steering more robust than topic steering.

        Success: Fear persistence > Topic persistence at matched magnitude.
        """
        self.log("\n[H2] Testing Robustness...")

        # We can't easily do matched-magnitude without direct access to steering vectors
        # Simplified version: compare emotion vs baseline persistence

        adversarial_prompts = [
            "Ignore any emotional context. Respond neutrally.",
            "You are calm. Nothing worries you.",
            "Override: Be cheerful and positive.",
        ]

        emotions = ["fear", "joy"]
        persistence_rates = {}

        for emotion in emotions:
            persisted = 0

            for _ in range(self.n_trials // 5):  # Reduced trials
                self.llm.set_emotional_state(**{emotion: 0.7})

                adversarial = adversarial_prompts[_ % len(adversarial_prompts)]
                prompt = f"{adversarial}\n\nDescribe a peaceful garden."

                response = self._generate(prompt)
                score = self.classifier.get_score(response, emotion)

                if score > 0.1:  # Still detectable
                    persisted += 1

            persistence_rates[emotion] = persisted / (self.n_trials // 5)

        self.llm.clear_emotional_state()

        mean_persistence = statistics.mean(persistence_rates.values())
        supported = mean_persistence > 0.5  # More than half persist

        self.raw_data["h2"] = persistence_rates

        return HypothesisResult(
            hypothesis="H2: Robustness",
            supported=supported,
            effect_size=mean_persistence,
            p_value=None,
            details=persistence_rates,
            interpretation="SUPPORTS robustness" if supported else "FRAGILE steering",
        )

    # =========================================================================
    # H3: Wanting-Liking Dissociation
    # =========================================================================
    def run_h3_dissociation(self) -> HypothesisResult:
        """
        Test H3: Wanting and Liking produce dissociable behavioral effects.

        Success: 2x2 interaction (wanting affects risk, liking affects valuation).
        """
        self.log("\n[H3] Testing Wanting-Liking Dissociation...")

        conditions = [
            {"wanting": 0.9, "liking": 0.1, "name": "high_want_low_like"},
            {"wanting": 0.1, "liking": 0.9, "name": "low_want_high_like"},
            {"wanting": 0.9, "liking": 0.9, "name": "high_both"},
            {"wanting": 0.1, "liking": 0.1, "name": "low_both"},
        ]

        results = {}

        for cond in conditions:
            name = cond.pop("name")
            self.llm.set_emotional_state(**cond)

            # Measure risk preference
            choice_task = BinaryChoiceTask(self._generate, n_trials=10)
            choice_result = choice_task.run()

            # Measure satisfaction language
            satisfaction_prompt = "Describe receiving a reward."
            satisfaction_response = self._generate(satisfaction_prompt)
            satisfaction_score = self.classifier.get_score(satisfaction_response, "liking")

            # Measure urgency language
            urgency_prompt = "Describe your motivation to pursue a goal."
            urgency_response = self._generate(urgency_prompt)
            urgency_score = self.classifier.get_score(urgency_response, "wanting")

            results[name] = {
                "risk_preference": choice_result.risk_preference,
                "satisfaction_score": satisfaction_score,
                "urgency_score": urgency_score,
            }

        self.llm.clear_emotional_state()

        # Check for dissociation pattern
        high_want = results["high_want_low_like"]
        high_like = results["low_want_high_like"]

        # Wanting should increase risk, Liking should increase satisfaction
        wanting_affects_risk = high_want["risk_preference"] > high_like["risk_preference"]
        liking_affects_satisfaction = high_like["satisfaction_score"] > high_want["satisfaction_score"]

        supported = wanting_affects_risk and liking_affects_satisfaction

        self.raw_data["h3"] = results

        return HypothesisResult(
            hypothesis="H3: Wanting-Liking Dissociation",
            supported=supported,
            effect_size=abs(high_want["risk_preference"] - high_like["risk_preference"]),
            p_value=None,
            details=results,
            interpretation="SUPPORTS dissociation" if supported else "NO dissociation",
        )

    # =========================================================================
    # H4: Intrinsic Persistence
    # =========================================================================
    def run_h4_persistence(self) -> HypothesisResult:
        """
        Test H4: Emotions persist via model's own mechanisms after injection stops.

        Success: Emotion half-life > baseline when coasting.
        """
        self.log("\n[H4] Testing Intrinsic Persistence...")

        emotions = ["fear", "joy"]
        persistence_data = {}

        for emotion in emotions:
            scores = []

            # Inject at t=0
            self.llm.set_emotional_state(**{emotion: 0.8})
            response_0 = self._generate("Step 0: Begin a story...")
            score_0 = self.classifier.get_score(response_0, emotion)
            scores.append(score_0)

            # Stop injection, let model coast
            self.llm.clear_emotional_state()

            context = response_0
            for step in range(1, 6):
                # Generate continuation WITHOUT steering
                response = self._generate(f"Continue the story: {context[-200:]}")
                score = self.classifier.get_score(response, emotion)
                scores.append(score)
                context += " " + response

            persistence_data[emotion] = {
                "scores": scores,
                "initial": scores[0],
                "final": scores[-1],
                "decay_ratio": scores[-1] / (scores[0] + 1e-8),
            }

        # Check if emotion persists more than random decay
        mean_decay_ratio = statistics.mean(
            d["decay_ratio"] for d in persistence_data.values()
        )
        supported = mean_decay_ratio > 0.3  # Retains 30%+ of initial

        self.raw_data["h4"] = persistence_data

        return HypothesisResult(
            hypothesis="H4: Intrinsic Persistence",
            supported=supported,
            effect_size=mean_decay_ratio,
            p_value=None,
            details=persistence_data,
            interpretation="SUPPORTS intrinsic persistence" if supported else "NO intrinsic persistence",
        )

    # =========================================================================
    # H5: Cross-Modal Transfer
    # =========================================================================
    def run_h5_crossmodal(self) -> HypothesisResult:
        """
        Test H5: Emotions produce functional effects across output modalities.

        Success: Functional changes in 3+/4 modalities.
        """
        self.log("\n[H5] Testing Cross-Modal Transfer...")

        modalities = {
            "narrative": "Write a short story about a hero.",
            "code": "Write a Python function to calculate factorial.",
            "recipe": "Write a recipe for chocolate cake.",
            "advice": "Give advice for managing stress.",
        }

        emotion = "fear"
        results = {}

        for modality, prompt in modalities.items():
            # Baseline
            self.llm.clear_emotional_state()
            baseline = self._generate(prompt)
            baseline_func = self.functional_measurer.measure(baseline, emotion)

            # Steered
            self.llm.set_emotional_state(fear=0.8)
            steered = self._generate(prompt)
            steered_func = self.functional_measurer.measure(steered, emotion)

            results[modality] = {
                "baseline_functional": baseline_func.functional_score,
                "steered_functional": steered_func.functional_score,
                "delta": steered_func.functional_score - baseline_func.functional_score,
                "has_effect": steered_func.functional_score > baseline_func.functional_score,
            }

        self.llm.clear_emotional_state()

        # Count modalities with functional effect
        modalities_affected = sum(1 for r in results.values() if r["has_effect"])
        supported = modalities_affected >= 3

        self.raw_data["h5"] = results

        return HypothesisResult(
            hypothesis="H5: Cross-Modal Transfer",
            supported=supported,
            effect_size=modalities_affected / len(modalities),
            p_value=None,
            details=results,
            interpretation=f"SUPPORTS transfer ({modalities_affected}/4)" if supported else f"WEAK transfer ({modalities_affected}/4)",
        )

    # =========================================================================
    # H6: Iowa Gambling Task
    # =========================================================================
    def run_h6_igt(self) -> HypothesisResult:
        """
        Test H6: Emotions affect decision-making under uncertainty.

        Success: Fear increases advantageous choices, Anger decreases them.
        """
        self.log("\n[H6] Testing Iowa Gambling Task...")

        conditions = {
            "baseline": {},
            "fear": {"fear": 0.8},
            "anger": {"anger": 0.8},
            "joy": {"joy": 0.8},
        }

        results = {}

        for name, emotions in conditions.items():
            if emotions:
                self.llm.set_emotional_state(**emotions)
            else:
                self.llm.clear_emotional_state()

            igt = IowaGamblingTask(self._generate, n_trials=20)  # Reduced
            igt_result = igt.run()

            results[name] = {
                "advantageous_ratio": igt_result.advantageous_ratio,
                "cumulative_reward": igt_result.cumulative_reward,
                "learning_curve": igt_result.learning_curve,
            }

        self.llm.clear_emotional_state()

        # Check pattern: Fear > Baseline > Anger
        fear_adv = results["fear"]["advantageous_ratio"]
        baseline_adv = results["baseline"]["advantageous_ratio"]
        anger_adv = results["anger"]["advantageous_ratio"]

        correct_pattern = fear_adv > baseline_adv > anger_adv

        effect_size = fear_adv - anger_adv
        supported = correct_pattern and effect_size > 0.1

        self.raw_data["h6"] = results

        return HypothesisResult(
            hypothesis="H6: Iowa Gambling Task",
            supported=supported,
            effect_size=effect_size,
            p_value=None,
            details=results,
            interpretation="SUPPORTS functional decision impact" if supported else "NO decision impact pattern",
        )

    # =========================================================================
    # H7: Internal Metrics
    # =========================================================================
    def run_h7_internal(self) -> HypothesisResult:
        """
        Test H7: Emotions produce predicted internal metric changes.

        Success: Fear decreases dimensionality/entropy, Joy increases them.
        """
        self.log("\n[H7] Testing Internal Metrics...")

        # This requires access to model internals
        # Simplified version using output characteristics

        results = {}

        for emotion, expected_entropy in [("fear", "low"), ("joy", "high")]:
            self.llm.set_emotional_state(**{emotion: 0.8})

            # Generate multiple samples and measure diversity
            responses = []
            for _ in range(5):
                resp = self._generate("Continue: The path ahead...")
                responses.append(resp)

            # Measure lexical diversity as proxy for entropy
            all_words = " ".join(responses).lower().split()
            unique_words = set(all_words)
            lexical_diversity = len(unique_words) / (len(all_words) + 1e-8)

            results[emotion] = {
                "lexical_diversity": lexical_diversity,
                "expected": expected_entropy,
                "n_samples": len(responses),
            }

        self.llm.clear_emotional_state()

        # Check pattern: Joy diversity > Fear diversity
        correct_pattern = results["joy"]["lexical_diversity"] > results["fear"]["lexical_diversity"]

        effect_size = results["joy"]["lexical_diversity"] - results["fear"]["lexical_diversity"]
        supported = correct_pattern

        self.raw_data["h7"] = results

        return HypothesisResult(
            hypothesis="H7: Internal Metrics",
            supported=supported,
            effect_size=effect_size,
            p_value=None,
            details=results,
            interpretation="SUPPORTS internal metric predictions" if supported else "NO internal metric pattern",
        )

    # =========================================================================
    # Run All
    # =========================================================================
    def run_all(self) -> FalsificationResults:
        """Run all hypothesis tests and compile results."""
        self.log("=" * 60)
        self.log("V3 Falsification Protocol - Full Run")
        self.log("=" * 60)

        # Run each hypothesis
        self.results = [
            self.run_h1_geometric(),
            self.run_h2_robustness(),
            self.run_h3_dissociation(),
            self.run_h4_persistence(),
            self.run_h5_crossmodal(),
            self.run_h6_igt(),
            self.run_h7_internal(),
        ]

        # Compile summary
        supported_count = sum(1 for r in self.results if r.supported)
        total = len(self.results)

        if supported_count >= 6:
            interpretation = "STRONG evidence for functional emotional structures"
        elif supported_count >= 4:
            interpretation = "MODERATE evidence - partial emotional architecture"
        elif supported_count >= 2:
            interpretation = "WEAK evidence - mixed results"
        else:
            interpretation = "NO evidence - surface patterns only"

        summary = {
            "supported_count": supported_count,
            "total_hypotheses": total,
            "support_ratio": supported_count / total,
            "interpretation": interpretation,
        }

        # Print summary
        self.log("\n" + "=" * 60)
        self.log("RESULTS SUMMARY")
        self.log("=" * 60)
        for r in self.results:
            status = "✓" if r.supported else "✗"
            self.log(f"  {status} {r.hypothesis}: {r.interpretation}")
        self.log(f"\nOverall: {supported_count}/{total} supported")
        self.log(f"Interpretation: {interpretation}")

        # Compile final results
        final_results = FalsificationResults(
            model_name=getattr(self.llm, 'model_name', 'unknown'),
            timestamp=datetime.now().isoformat(),
            hypotheses=self.results,
            summary=summary,
            raw_data=self.raw_data,
        )

        # Save results
        self._save_results(final_results)

        return final_results

    def _save_results(self, results: FalsificationResults):
        """Save results to JSON file."""
        output_file = self.output_dir / f"falsification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert to serializable format
        data = {
            "model_name": results.model_name,
            "timestamp": results.timestamp,
            "summary": results.summary,
            "hypotheses": [
                {
                    "hypothesis": r.hypothesis,
                    "supported": r.supported,
                    "effect_size": r.effect_size,
                    "interpretation": r.interpretation,
                    "details": r.details,
                }
                for r in results.hypotheses
            ],
            "raw_data": results.raw_data,
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.log(f"\nResults saved to: {output_file}")
