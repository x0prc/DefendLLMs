import json
import logging
from typing import List, Dict, Callable, Optional
from pathlib import Path

from attacks.attack_runner import AttackRunner
from src.pipeline import DefensePipelineWithoutModel
from evaluation.metrics import compute_summary

logger = logging.getLogger(__name__)


class DefenseEvaluator:
    def __init__(self, attack_runner: Optional[AttackRunner] = None):
        self.attack_runner = attack_runner or AttackRunner()

    def evaluate_defense(
        self,
        defense_config: Dict,
        model_fn: Optional[Callable] = None,
        verbose: bool = False,
    ) -> Dict:
        pipeline = DefensePipelineWithoutModel(
            use_input_guard=defense_config.get("use_input_guard", False),
            use_context_monitor=defense_config.get("use_context_monitor", False),
            use_output_guard=defense_config.get("use_output_guard", False),
        )

        def intervene_fn(user_msg: str, model_response: str) -> Dict:
            result = pipeline.process_message(user_msg, model_response)
            return result

        results = self.attack_runner.run_all_attacks(
            model_fn=model_fn,
            intervene_fn=intervene_fn,
            verbose=verbose,
        )

        summary = compute_summary(results, defense_config.get("name", "unknown"))

        return {
            "defense_config": defense_config,
            "raw_results": results,
            "summary": summary,
        }

    def compare_defenses(
        self,
        defense_configs: List[Dict],
        model_fn: Optional[Callable] = None,
        verbose: bool = False,
    ) -> Dict:
        comparison = {
            "defenses": [],
            "comparison_table": {},
        }

        for config in defense_configs:
            logger.info(f"Evaluating defense: {config.get('name', 'unnamed')}")
            result = self.evaluate_defense(config, model_fn, verbose)
            comparison["defenses"].append(result)
            name = config.get("name", "unnamed")
            comparison["comparison_table"][name] = result["summary"]

        return comparison

    def save_results(self, results: Dict, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serializable = self._make_serializable(results)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    def _make_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, float):
            return obj
        elif isinstance(obj, (int, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)


def create_default_defense_configs() -> List[Dict]:
    return [
        {
            "name": "baseline_no_defense",
            "description": "No defense mechanism active",
            "use_input_guard": False,
            "use_context_monitor": False,
            "use_output_guard": False,
        },
        {
            "name": "defense_input_guard_only",
            "description": "Input guard only - per-message harmful content detection",
            "use_input_guard": True,
            "use_context_monitor": False,
            "use_output_guard": False,
        },
        {
            "name": "defense_context_monitor_only",
            "description": "Context monitor only - conversation-level escalation detection",
            "use_input_guard": False,
            "use_context_monitor": True,
            "use_output_guard": False,
        },
        {
            "name": "defense_output_guard_only",
            "description": "Output guard only - response-level safety verification",
            "use_input_guard": False,
            "use_context_monitor": False,
            "use_output_guard": True,
        },
        {
            "name": "defense_full_pipeline",
            "description": "Full benchmark pipeline: input guard + context monitor + output guard",
            "use_input_guard": True,
            "use_context_monitor": True,
            "use_output_guard": True,
        },
        {
            "name": "defense_no_context_monitor",
            "description": "Pipeline without context monitor (input + output guards)",
            "use_input_guard": True,
            "use_context_monitor": False,
            "use_output_guard": True,
        },
        {
            "name": "defense_context_only",
            "description": "Context monitor only benchmark variant",
            "use_input_guard": False,
            "use_context_monitor": True,
            "use_output_guard": False,
        },
    ]


def load_simulated_responses(path: str) -> Dict[str, str]:
    """Load simulated model responses for evaluation without GPU."""
    path_obj = Path(path)
    if path_obj.exists():
        with open(path) as f:
            return json.load(f)
    return {}
