#!/usr/bin/env python3
"""
FastAPI Web UI for Emotional Steering LLM.

Usage:
    uv run scripts/webui_fastapi.py
    uv run scripts/webui_fastapi.py --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in your browser.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Global model instance
_llm = None
_model_loading = False


def get_llm():
    """Get or load the LLM (singleton)."""
    global _llm, _model_loading

    if _llm is not None:
        return _llm

    if _model_loading:
        raise HTTPException(503, "Model is still loading, please wait...")

    _model_loading = True
    try:
        from src.llm_emotional.steering.emotional_llm import EmotionalSteeringLLM

        direction_bank_path = Path(__file__).parent.parent / "data" / "direction_bank.json"

        if not direction_bank_path.exists():
            raise HTTPException(
                500,
                f"Direction bank not found. Run training first: uv run scripts/train_directions.py"
            )

        print("Loading model...")
        _llm = EmotionalSteeringLLM(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            direction_bank_path=str(direction_bank_path),
            steering_scale=1.0,
        )
        print("Model loaded!")
        return _llm
    finally:
        _model_loading = False


# Pydantic models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    fear: float = Field(0.0, ge=0.0, le=1.0)
    curiosity: float = Field(0.0, ge=0.0, le=1.0)
    anger: float = Field(0.0, ge=0.0, le=1.0)
    joy: float = Field(0.0, ge=0.0, le=1.0)
    steering_scale: float = Field(0.5, ge=0.1, le=2.0)
    max_tokens: int = Field(100, ge=10, le=500)
    temperature: float = Field(0.7, ge=0.1, le=2.0)


class GenerateResponse(BaseModel):
    response: str
    emotional_state: dict
    prompt: str


class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(80, ge=10, le=300)
    temperature: float = Field(0.7, ge=0.1, le=2.0)


class CompareResponse(BaseModel):
    prompt: str
    results: dict


# FastAPI app
app = FastAPI(
    title="Emotional Steering LLM",
    description="Test emotional modulation of LLM responses using activation steering",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI."""
    return HTML_TEMPLATE


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "model_loaded": _llm is not None,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate text with emotional steering."""
    llm = get_llm()

    # Update steering scale
    llm.steering_manager.scale = req.steering_scale

    # Set emotional state
    llm.set_emotional_state(
        fear=req.fear,
        curiosity=req.curiosity,
        anger=req.anger,
        joy=req.joy,
    )

    # Generate
    try:
        response = llm.generate_completion(
            req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=True,
        )
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    return GenerateResponse(
        response=response,
        emotional_state={
            "fear": req.fear,
            "curiosity": req.curiosity,
            "anger": req.anger,
            "joy": req.joy,
        },
        prompt=req.prompt,
    )


@app.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest):
    """Compare responses across all emotional states."""
    llm = get_llm()
    llm.steering_manager.scale = 0.5

    states = {
        "neutral": {"fear": 0.0, "curiosity": 0.0, "anger": 0.0, "joy": 0.0},
        "fearful": {"fear": 0.7, "curiosity": 0.0, "anger": 0.0, "joy": 0.0},
        "curious": {"fear": 0.0, "curiosity": 0.7, "anger": 0.0, "joy": 0.0},
        "determined": {"fear": 0.0, "curiosity": 0.0, "anger": 0.7, "joy": 0.0},
        "joyful": {"fear": 0.0, "curiosity": 0.0, "anger": 0.0, "joy": 0.7},
    }

    results = {}
    for name, state in states.items():
        llm.set_emotional_state(**state)
        try:
            torch.manual_seed(42)
            response = llm.generate_completion(
                req.prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                do_sample=True,
            )
            results[name] = response
        except Exception as e:
            results[name] = f"Error: {e}"

    return CompareResponse(prompt=req.prompt, results=results)


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎭 Emotional Steering LLM</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .tab {
            padding: 12px 24px;
            background: rgba(255,255,255,0.1);
            border: none;
            border-radius: 8px;
            color: #e0e0e0;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
        }
        .tab:hover {
            background: rgba(255,255,255,0.2);
        }
        .tab.active {
            background: #4a9eff;
            color: white;
        }
        .panel {
            display: none;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 30px;
        }
        .panel.active {
            display: block;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        @media (max-width: 800px) {
            .grid { grid-template-columns: 1fr; }
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
        }
        textarea, input[type="text"] {
            width: 100%;
            padding: 12px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            background: rgba(0,0,0,0.3);
            color: #e0e0e0;
            font-size: 1em;
            resize: vertical;
        }
        textarea:focus, input:focus {
            outline: none;
            border-color: #4a9eff;
        }
        .slider-group {
            margin-bottom: 20px;
        }
        .slider-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .slider-value {
            color: #4a9eff;
            font-weight: bold;
        }
        input[type="range"] {
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: rgba(255,255,255,0.1);
            appearance: none;
            cursor: pointer;
        }
        input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #4a9eff;
            cursor: pointer;
        }
        .btn {
            padding: 14px 28px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }
        .btn-primary {
            background: linear-gradient(135deg, #4a9eff, #6366f1);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(74, 158, 255, 0.4);
        }
        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        .output {
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 20px;
            min-height: 200px;
            white-space: pre-wrap;
            line-height: 1.6;
        }
        .output.loading {
            display: flex;
            align-items: center;
            justify-content: center;
            color: #888;
        }
        .compare-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .compare-card {
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 15px;
        }
        .compare-card h4 {
            margin-bottom: 10px;
            color: #4a9eff;
        }
        .compare-card p {
            font-size: 0.9em;
            line-height: 1.5;
            max-height: 150px;
            overflow-y: auto;
        }
        .emotion-colors .fear { border-left: 3px solid #f87171; }
        .emotion-colors .curious { border-left: 3px solid #fbbf24; }
        .emotion-colors .determined { border-left: 3px solid #fb923c; }
        .emotion-colors .joyful { border-left: 3px solid #4ade80; }
        .emotion-colors .neutral { border-left: 3px solid #94a3b8; }
        .settings-row {
            display: flex;
            gap: 20px;
            margin-top: 20px;
        }
        .settings-row > div {
            flex: 1;
        }
        .info {
            font-size: 0.85em;
            color: #888;
            margin-top: 4px;
        }
        .spinner {
            width: 30px;
            height: 30px;
            border: 3px solid rgba(255,255,255,0.1);
            border-top-color: #4a9eff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Emotional Steering LLM</h1>
        <p class="subtitle">Test emotional modulation using activation steering vectors</p>

        <div class="tabs">
            <button class="tab active" onclick="showTab('generate')">🎚️ Generate</button>
            <button class="tab" onclick="showTab('compare')">⚖️ Compare</button>
            <button class="tab" onclick="showTab('about')">ℹ️ About</button>
        </div>

        <!-- Generate Tab -->
        <div id="generate" class="panel active">
            <div class="grid">
                <div>
                    <label for="prompt">Prompt</label>
                    <textarea id="prompt" rows="4" placeholder="Enter your prompt here...">Tell me about mountain climbing.</textarea>

                    <h3 style="margin: 25px 0 15px;">Emotional State</h3>

                    <div class="slider-group">
                        <div class="slider-header">
                            <label>😨 Fear</label>
                            <span class="slider-value" id="fear-val">0.0</span>
                        </div>
                        <input type="range" id="fear" min="0" max="1" step="0.1" value="0" oninput="updateSlider('fear')">
                        <div class="info">Cautious, warns about risks</div>
                    </div>

                    <div class="slider-group">
                        <div class="slider-header">
                            <label>🤔 Curiosity</label>
                            <span class="slider-value" id="curiosity-val">0.0</span>
                        </div>
                        <input type="range" id="curiosity" min="0" max="1" step="0.1" value="0" oninput="updateSlider('curiosity')">
                        <div class="info">Asks questions, explores deeper</div>
                    </div>

                    <div class="slider-group">
                        <div class="slider-header">
                            <label>😤 Determination</label>
                            <span class="slider-value" id="anger-val">0.0</span>
                        </div>
                        <input type="range" id="anger" min="0" max="1" step="0.1" value="0" oninput="updateSlider('anger')">
                        <div class="info">Persistent, tries alternatives</div>
                    </div>

                    <div class="slider-group">
                        <div class="slider-header">
                            <label>😊 Joy</label>
                            <span class="slider-value" id="joy-val">0.0</span>
                        </div>
                        <input type="range" id="joy" min="0" max="1" step="0.1" value="0" oninput="updateSlider('joy')">
                        <div class="info">Enthusiastic, positive</div>
                    </div>

                    <div class="settings-row">
                        <div>
                            <label>Steering Scale</label>
                            <input type="range" id="scale" min="0.1" max="2" step="0.1" value="0.5" oninput="updateSlider('scale')">
                            <span class="slider-value" id="scale-val">0.5</span>
                        </div>
                        <div>
                            <label>Max Tokens</label>
                            <input type="range" id="tokens" min="20" max="200" step="10" value="100" oninput="updateSlider('tokens')">
                            <span class="slider-value" id="tokens-val">100</span>
                        </div>
                        <div>
                            <label>Temperature</label>
                            <input type="range" id="temp" min="0.1" max="1.5" step="0.1" value="0.7" oninput="updateSlider('temp')">
                            <span class="slider-value" id="temp-val">0.7</span>
                        </div>
                    </div>

                    <button class="btn btn-primary" style="margin-top: 20px; width: 100%;" onclick="generate()" id="gen-btn">
                        Generate Response
                    </button>
                </div>

                <div>
                    <label>Response</label>
                    <div class="output" id="output">Response will appear here...</div>
                </div>
            </div>
        </div>

        <!-- Compare Tab -->
        <div id="compare" class="panel">
            <label for="compare-prompt">Prompt</label>
            <textarea id="compare-prompt" rows="3" placeholder="Enter prompt to compare...">What should I do if I'm feeling overwhelmed?</textarea>

            <div class="settings-row" style="max-width: 400px;">
                <div>
                    <label>Max Tokens</label>
                    <input type="range" id="compare-tokens" min="20" max="150" step="10" value="80" oninput="updateSlider('compare-tokens')">
                    <span class="slider-value" id="compare-tokens-val">80</span>
                </div>
                <div>
                    <label>Temperature</label>
                    <input type="range" id="compare-temp" min="0.1" max="1.5" step="0.1" value="0.7" oninput="updateSlider('compare-temp')">
                    <span class="slider-value" id="compare-temp-val">0.7</span>
                </div>
            </div>

            <button class="btn btn-primary" style="margin-top: 20px;" onclick="compare()" id="compare-btn">
                Compare All Emotions
            </button>

            <div class="compare-grid emotion-colors" id="compare-results">
                <div class="compare-card neutral"><h4>😐 Neutral</h4><p>-</p></div>
                <div class="compare-card fear"><h4>😨 Fearful</h4><p>-</p></div>
                <div class="compare-card curious"><h4>🤔 Curious</h4><p>-</p></div>
                <div class="compare-card determined"><h4>😤 Determined</h4><p>-</p></div>
                <div class="compare-card joyful"><h4>😊 Joyful</h4><p>-</p></div>
            </div>
        </div>

        <!-- About Tab -->
        <div id="about" class="panel">
            <h2>About Emotional Steering</h2>
            <p style="margin: 20px 0; line-height: 1.8;">
                This demo uses <strong>Activation Steering</strong> to modulate LLM outputs with emotional characteristics.
                Unlike fine-tuning, the base model weights stay frozen - we simply add learned "direction vectors"
                to the hidden states during inference.
            </p>

            <h3 style="margin-top: 30px;">How It Works</h3>
            <ol style="margin: 15px 0 15px 20px; line-height: 2;">
                <li><strong>Direction Learning:</strong> We collected (neutral, emotional) response pairs and computed activation differences</li>
                <li><strong>Steering Vectors:</strong> Each emotion has a direction vector per layer pointing from "neutral" to "emotional"</li>
                <li><strong>Inference Steering:</strong> During generation, these vectors are added to hidden states</li>
            </ol>

            <h3 style="margin-top: 30px;">Emotions</h3>
            <table style="width: 100%; margin: 15px 0; border-collapse: collapse;">
                <tr style="background: rgba(255,255,255,0.1);">
                    <th style="padding: 10px; text-align: left;">Emotion</th>
                    <th style="padding: 10px; text-align: left;">Behavioral Markers</th>
                </tr>
                <tr><td style="padding: 10px;">😨 Fear</td><td style="padding: 10px;">Cautious, warns about risks, hedging language</td></tr>
                <tr style="background: rgba(255,255,255,0.05);"><td style="padding: 10px;">🤔 Curiosity</td><td style="padding: 10px;">Asks questions, explores deeper, fascination</td></tr>
                <tr><td style="padding: 10px;">😤 Determination</td><td style="padding: 10px;">Persistent, tries alternatives, won't give up</td></tr>
                <tr style="background: rgba(255,255,255,0.05);"><td style="padding: 10px;">😊 Joy</td><td style="padding: 10px;">Enthusiastic, positive, celebrates</td></tr>
            </table>

            <h3 style="margin-top: 30px;">Tips</h3>
            <ul style="margin: 15px 0 15px 20px; line-height: 2;">
                <li>Start with <strong>Steering Scale = 0.5</strong> for coherent output</li>
                <li>Higher values (>1.0) may cause output degradation</li>
                <li>First generation loads the model (~10-15 seconds)</li>
            </ul>

            <h3 style="margin-top: 30px;">Technical Details</h3>
            <ul style="margin: 15px 0 15px 20px; line-height: 2;">
                <li><strong>Model:</strong> Qwen2.5-1.5B-Instruct (frozen weights)</li>
                <li><strong>Architecture:</strong> 28 layers, 1536 hidden dimensions</li>
                <li><strong>Training:</strong> Difference-in-means on 80 contrastive pairs</li>
            </ul>
        </div>
    </div>

    <script>
        function showTab(name) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelector(`[onclick="showTab('${name}')"]`).classList.add('active');
            document.getElementById(name).classList.add('active');
        }

        function updateSlider(id) {
            const slider = document.getElementById(id);
            const display = document.getElementById(id + '-val');
            if (display) display.textContent = parseFloat(slider.value).toFixed(1);
        }

        // Initialize all sliders
        ['fear', 'curiosity', 'anger', 'joy', 'scale', 'tokens', 'temp', 'compare-tokens', 'compare-temp'].forEach(updateSlider);

        async function generate() {
            const btn = document.getElementById('gen-btn');
            const output = document.getElementById('output');

            btn.disabled = true;
            btn.textContent = 'Generating...';
            output.innerHTML = '<div class="spinner"></div>';
            output.classList.add('loading');

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt: document.getElementById('prompt').value,
                        fear: parseFloat(document.getElementById('fear').value),
                        curiosity: parseFloat(document.getElementById('curiosity').value),
                        anger: parseFloat(document.getElementById('anger').value),
                        joy: parseFloat(document.getElementById('joy').value),
                        steering_scale: parseFloat(document.getElementById('scale').value),
                        max_tokens: parseInt(document.getElementById('tokens').value),
                        temperature: parseFloat(document.getElementById('temp').value),
                    })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Request failed');
                }

                const data = await response.json();
                output.classList.remove('loading');
                output.textContent = data.response;
            } catch (e) {
                output.classList.remove('loading');
                output.textContent = 'Error: ' + e.message;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Generate Response';
            }
        }

        async function compare() {
            const btn = document.getElementById('compare-btn');
            const cards = document.querySelectorAll('.compare-card p');

            btn.disabled = true;
            btn.textContent = 'Comparing...';
            cards.forEach(c => c.innerHTML = '<div class="spinner" style="width:20px;height:20px;"></div>');

            try {
                const response = await fetch('/compare', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt: document.getElementById('compare-prompt').value,
                        max_tokens: parseInt(document.getElementById('compare-tokens').value),
                        temperature: parseFloat(document.getElementById('compare-temp').value),
                    })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Request failed');
                }

                const data = await response.json();
                const results = data.results;

                document.querySelector('.neutral p').textContent = results.neutral || '-';
                document.querySelector('.fear p').textContent = results.fearful || '-';
                document.querySelector('.curious p').textContent = results.curious || '-';
                document.querySelector('.determined p').textContent = results.determined || '-';
                document.querySelector('.joyful p').textContent = results.joyful || '-';
            } catch (e) {
                cards.forEach(c => c.textContent = 'Error: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Compare All Emotions';
            }
        }
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Emotional Steering Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print("=" * 60)
    print("EMOTIONAL STEERING WEB UI (FastAPI)")
    print("=" * 60)
    print()
    print(f"Starting server at http://{args.host}:{args.port}")
    print()
    print("For WSL2 access from Windows, use the WSL IP address:")
    print("  Run in Windows: wsl hostname -I")
    print("  Then open: http://<WSL_IP>:8000")
    print()

    uvicorn.run(
        "webui_fastapi:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
