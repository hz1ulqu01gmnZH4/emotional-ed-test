# emotional-ed-test

Minimal test of emotions-as-value-functions hypothesis using Error Diffusion (ED) architecture.

## Hypothesis

Multiple emotional channels (fear, anger, regret, grief) broadcasting distinct learning signals produce qualitatively different behavior than single reward signal.

## Test

Tabular grid-world comparing:
- Standard Q-learning (single δ)
- Emotional ED agent (multi-channel broadcast)

## Run

```bash
python test_fear.py
```

## License

WTFPL
