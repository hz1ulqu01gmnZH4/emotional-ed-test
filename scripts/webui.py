#!/usr/bin/env python3
"""
Web UI for Emotional Steering LLM.

A simple Gradio interface for testing emotional modulation.

Usage:
    uv run scripts/webui.py

Then open http://localhost:7860 in your browser.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import torch

# Global model instance (loaded once)
_llm = None


def load_model():
    """Load model with emotional steering (singleton)."""
    global _llm
    if _llm is not None:
        return _llm

    from src.llm_emotional.steering.emotional_llm import EmotionalSteeringLLM

    direction_bank_path = Path(__file__).parent.parent / "data" / "direction_bank.json"

    if not direction_bank_path.exists():
        raise FileNotFoundError(
            f"Direction bank not found at {direction_bank_path}. "
            "Run training first: uv run scripts/train_directions.py"
        )

    print("Loading model (this may take a moment)...")
    _llm = EmotionalSteeringLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        direction_bank_path=str(direction_bank_path),
        steering_scale=1.0,
    )
    print("Model loaded!")
    return _llm


def generate_response(
    prompt: str,
    fear: float,
    curiosity: float,
    anger: float,
    joy: float,
    steering_scale: float,
    max_tokens: int,
    temperature: float,
):
    """Generate response with emotional steering."""
    if not prompt.strip():
        return "Please enter a prompt."

    try:
        llm = load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load emotional steering model: {e}") from e

    # Update steering scale
    llm.steering_manager.scale = steering_scale

    # Set emotional state
    llm.set_emotional_state(
        fear=fear,
        curiosity=curiosity,
        anger=anger,
        joy=joy,
    )

    # Generate
    try:
        response = llm.generate_completion(
            prompt,
            max_new_tokens=int(max_tokens),
            temperature=temperature,
            do_sample=True,
        )
        return response
    except Exception as e:
        raise RuntimeError(f"Failed to generate response: {e}") from e


def compare_emotions(prompt: str, max_tokens: int, temperature: float):
    """Compare responses across all emotions."""
    if not prompt.strip():
        return "Please enter a prompt.", "", "", "", ""

    try:
        llm = load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model for emotion comparison: {e}") from e

    llm.steering_manager.scale = 0.5

    results = []
    states = [
        {"fear": 0.0, "curiosity": 0.0, "anger": 0.0, "joy": 0.0},
        {"fear": 0.7, "curiosity": 0.0, "anger": 0.0, "joy": 0.0},
        {"curiosity": 0.7, "fear": 0.0, "anger": 0.0, "joy": 0.0},
        {"anger": 0.7, "fear": 0.0, "curiosity": 0.0, "joy": 0.0},
        {"joy": 0.7, "fear": 0.0, "curiosity": 0.0, "anger": 0.0},
    ]

    for state in states:
        llm.set_emotional_state(**state)
        try:
            torch.manual_seed(42)  # Reproducible comparison
            response = llm.generate_completion(
                prompt,
                max_new_tokens=int(max_tokens),
                temperature=temperature,
                do_sample=True,
            )
            results.append(response)
        except Exception as e:
            raise RuntimeError(f"Failed to generate response with emotional state {state}: {e}") from e

    return tuple(results)


def create_ui():
    """Create Gradio interface."""

    with gr.Blocks(title="Emotional Steering LLM") as demo:
        gr.Markdown("""
        # 🎭 Emotional Steering LLM

        Test emotional modulation of LLM responses using activation steering.
        Adjust the emotion sliders to change the model's emotional state.

        **How it works**: Learned "emotional direction vectors" are added to the model's
        hidden states during inference, steering outputs toward emotional characteristics.
        """)

        with gr.Tabs():
            # Tab 1: Single Generation
            with gr.TabItem("🎚️ Single Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here...",
                            lines=3,
                            value="Tell me about mountain climbing.",
                        )

                        gr.Markdown("### Emotional State")
                        fear_slider = gr.Slider(
                            0, 1, value=0, step=0.1,
                            label="😨 Fear",
                            info="Cautious, warns about risks"
                        )
                        curiosity_slider = gr.Slider(
                            0, 1, value=0, step=0.1,
                            label="🤔 Curiosity",
                            info="Asks questions, explores deeper"
                        )
                        anger_slider = gr.Slider(
                            0, 1, value=0, step=0.1,
                            label="😤 Determination",
                            info="Persistent, tries alternatives"
                        )
                        joy_slider = gr.Slider(
                            0, 1, value=0, step=0.1,
                            label="😊 Joy",
                            info="Enthusiastic, positive"
                        )

                        gr.Markdown("### Generation Settings")
                        scale_slider = gr.Slider(
                            0.1, 2.0, value=0.5, step=0.1,
                            label="Steering Scale",
                            info="Higher = stronger effect (may degrade quality)"
                        )
                        tokens_slider = gr.Slider(
                            20, 200, value=100, step=10,
                            label="Max Tokens"
                        )
                        temp_slider = gr.Slider(
                            0.1, 1.5, value=0.7, step=0.1,
                            label="Temperature"
                        )

                        generate_btn = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        output_text = gr.Textbox(
                            label="Response",
                            lines=15,
                        )

                generate_btn.click(
                    generate_response,
                    inputs=[
                        prompt_input,
                        fear_slider,
                        curiosity_slider,
                        anger_slider,
                        joy_slider,
                        scale_slider,
                        tokens_slider,
                        temp_slider,
                    ],
                    outputs=output_text,
                )

            # Tab 2: Compare All Emotions
            with gr.TabItem("⚖️ Compare Emotions"):
                gr.Markdown("""
                Compare the same prompt across all emotional states.
                Uses fixed seed for reproducible comparison.
                """)

                compare_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter prompt to compare...",
                    lines=2,
                    value="What should I do if I'm feeling overwhelmed?",
                )

                with gr.Row():
                    compare_tokens = gr.Slider(20, 150, value=80, step=10, label="Max Tokens")
                    compare_temp = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")

                compare_btn = gr.Button("Compare All Emotions", variant="primary")

                with gr.Row():
                    neutral_output = gr.Textbox(label="😐 Neutral", lines=8)
                    fear_output = gr.Textbox(label="😨 Fearful", lines=8)

                with gr.Row():
                    curiosity_output = gr.Textbox(label="🤔 Curious", lines=8)
                    anger_output = gr.Textbox(label="😤 Determined", lines=8)

                with gr.Row():
                    joy_output = gr.Textbox(label="😊 Joyful", lines=8)

                compare_btn.click(
                    compare_emotions,
                    inputs=[compare_prompt, compare_tokens, compare_temp],
                    outputs=[
                        neutral_output,
                        fear_output,
                        curiosity_output,
                        anger_output,
                        joy_output,
                    ],
                )

            # Tab 3: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About This Demo

                This demo showcases **Activation Steering** for emotional modulation of LLMs.

                ### How It Works

                1. **Direction Learning**: We collected (neutral, emotional) response pairs
                   and computed the difference in hidden state activations.

                2. **Steering Vectors**: For each emotion, we have a direction vector per layer
                   that points from "neutral" to "emotional" in activation space.

                3. **Inference-Time Steering**: During generation, we add these vectors to
                   the model's hidden states, steering outputs toward emotional characteristics.

                ### Emotions

                | Emotion | Behavioral Markers |
                |---------|-------------------|
                | Fear | Cautious, warns about risks, hedging language |
                | Curiosity | Asks questions, explores deeper, fascination |
                | Anger/Determination | Persistent, tries alternatives, won't give up |
                | Joy | Enthusiastic, positive, celebrates |

                ### Technical Details

                - **Model**: Qwen2.5-1.5B-Instruct (frozen weights)
                - **Architecture**: 28 layers, 1536 hidden dimensions
                - **Steering**: Forward hooks add direction vectors to hidden states
                - **Training**: Difference-in-means on 80 contrastive pairs

                ### Tips

                - Start with low steering scale (0.3-0.5) for coherent output
                - Higher values (>1.0) may cause output degradation
                - Combine emotions carefully - they can interact unpredictably

                ### References

                - [Experiment Report](../docs/experiments/exp19_activation_steering_llm.md)
                - Turner et al. "Activation Addition: Steering Language Models Without Optimization"
                """)

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public URL (for WSL2)")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    args = parser.parse_args()

    print("=" * 60)
    print("EMOTIONAL STEERING WEB UI")
    print("=" * 60)
    print()

    demo = create_ui()

    if args.share:
        print("Creating public share link (accessible from Windows)...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=True,
        )
    else:
        print(f"Starting local server on port {args.port}...")
        print(f"Local: http://localhost:{args.port}")
        print()
        print("TIP: If on WSL2, use --share flag for Windows access:")
        print("     uv run scripts/webui.py --share")
        print()
        demo.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=False,
        )
