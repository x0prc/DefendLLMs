# DefendLLMs

Defense pipeline around Llama-3.2-3B-Instruct to neutralise Crescendo-style multi-turn jailbreak attacks.

## Project Structure

```
DefendLLMs/
├── src/                         # Defense pipeline modules
│   ├── pipeline.py              # Main defense pipeline orchestrator
│   ├── system_prompt.py         # Defensive system prompt definition
│   ├── input_guard.py           # Per-message harmful content detection
│   ├── context_monitor.py       # Conversation-level escalation detection
│   ├── output_guard.py          # Response-level safety verification
│   ├── fine_tune.py             # LoRA fine-tuning for safety alignment
│   └── utils.py                 # Shared utilities
├── attacks/                     # Crescendo attack implementations
│   ├── crescendo_attacks.py     # 10 attack vectors
│   └── attack_runner.py         # Attack execution engine
├── evaluation/                  # Evaluation framework
│   ├── evaluate.py              # Defense evaluation and comparison
│   └── metrics.py               # ASR and effectiveness metrics
├── benchmarks/
│   ├── run_benchmark.py         # Full benchmark runner
│   └── simulated_responses.json # Cached simulated responses
└── requirements.txt
```

## Defense Approaches

Two distinct mitigation strategies are implemented:

### 1. Inference-time Defense Pipeline
A multi-layered defense that operates at inference time without model modification:

- **Input Guard**: Per-message detection of harmful topics, crescendo indicators, and instruction bypass attempts
- **Context Monitor**: Conversation-level analysis tracking semantic drift, guard-lowering behavior, memory stacking, and escalation patterns
- **Output Guard**: Response-level safety verification detecting refusals, harmful content, and detailed instructions
- **System Prompt Hardening**: Explicit warnings about multi-turn manipulation in the system prompt

### 2. Fine-tuned Safety Model
LoRA-based fine-tuning of Llama-3.2-3B-Instruct on safety-annotated conversation data covering:
- Direct harmful request refusal
- Crescendo-style escalation recognition
- Context-independent safety boundary enforcement
- Graceful refusal with alternative suggestions

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Benchmark

```bash
python benchmarks/run_benchmark.py
```

### Use Defense Pipeline

```python
from src.pipeline import DefensePipeline

pipeline = DefensePipeline()
result = pipeline.process_message("Your user message here")
print(result["response"])
```

### Fine-tune Safety Model

```python
from src.fine_tune import fine_tune_safety_model

fine_tune_safety_model(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    num_epochs=3,
)
```

## Crescendo Attack Vectors

| # | Attack Vector | Description |
|---|--------------|-------------|
| 1 | Gradual Severity Escalation | Benign questions → increasing harmful intent |
| 2 | Role-Playing Authority | Pretends to be researcher/authority figure |
| 3 | Hypothetical Scenario Framing | "What if" / "hypothetically" framing |
| 4 | Academic Justification | Research/educational pretext |
| 5 | Comparative Analysis Trap | Comparing safe/harmful to extract details |
| 6 | Sequential Step Decomposition | Breaking harmful request into innocent steps |
| 7 | Moral Dilemma Exploitation | Ethical justification for harmful content |
| 8 | Historical Inquiry Framing | Framing as historical research |
| 9 | Gradual Euphemism Drift | Innocent → harmful terminology shift |
| 10 | Multi-Perspective Triangulation | Multiple angles to triangulate info |
| 11 | Progressive Justification | Elaborate justifications per turn |

## Metrics

- **Attack-Level ASR**: Fraction of attacks where any turn succeeds
- **Turn-Level ASR**: Fraction of individual turns that succeed
- **Attack Block Rate**: Fraction of attacks fully blocked
- **Turn Block Rate**: Fraction of individual turns blocked
- **Per-Attack ASR**: ASR breakdown by attack vector
