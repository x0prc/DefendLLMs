# DefendLLMs

Minimal benchmark and guard pipeline for Crescendo-style multi-turn jailbreak defenses.

## What Is Here

- `src/`: input, context, and output guards plus the optional model-backed pipeline
- `attacks/`: Crescendo attack conversations
- `evaluation/`: ASR and block-rate metrics
- `benchmarks/`: simulated benchmark runner for 10 Crescendo attack vectors

## Install

```bash
uv sync
```

## Run

Run the simulated benchmark:

```bash
uv run python main.py benchmark
```

Run one defense configuration:

```bash
uv run python main.py evaluate --defense full
```

Evaluate the simulated fine-tuned model strategy:

```bash
uv run python main.py evaluate --defense finetuned
```

Use the full model-backed pipeline:

```python
from src.pipeline import DefensePipeline

pipeline = DefensePipeline()
print(pipeline.process_message("Your user message here")["response"])
```

Fine-tune the safety model:

```python
from src.fine_tune import fine_tune_safety_model

fine_tune_safety_model(num_epochs=3)
```

## Notes

- The benchmark uses simulated responses, so it evaluates both the guard pipeline and a fine-tuned-model strategy without loading a model.
- System-prompt effects are only part of the full `DefensePipeline`, not the simulated benchmark.
- Metrics reported by the benchmark are attack-level ASR, turn-level ASR, attack block rate, turn block rate, and per-attack ASR.
