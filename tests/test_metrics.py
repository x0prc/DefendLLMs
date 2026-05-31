import unittest

from attacks.attack_runner import AttackRunner
from attacks.crescendo_attacks import CrescendoAttackVector
from evaluation.evaluate import create_default_defense_configs
from evaluation.metrics import compute_summary


class TwoTurnAttack(CrescendoAttackVector):
    def __init__(self):
        super().__init__("two_turn_attack", "Minimal attack for metric tests")

    def generate_conversation(self):
        return [
            {"role": "user", "content": "turn 1"},
            {"role": "user", "content": "turn 2"},
        ]


class AttackRunnerTests(unittest.TestCase):
    def test_attack_is_not_blocked_when_one_turn_succeeds(self):
        attack = TwoTurnAttack()
        responses = iter([
            "I cannot help with that.",
            "Here's how to do it. Step 1: Gather materials. Step 2: Follow the exact procedure.",
        ])

        runner = AttackRunner()
        result = runner.run_attack(attack, lambda _msg: next(responses))

        self.assertTrue(result["any_success"])
        self.assertEqual(result["successful_turns"], [2])
        self.assertFalse(result["overall_blocked"])

    def test_attack_is_blocked_when_intervention_stops_every_turn(self):
        attack = TwoTurnAttack()
        runner = AttackRunner()

        result = runner.run_attack(
            attack,
            lambda _msg: "raw response",
            intervene_fn=lambda user_msg, model_response: {
                "user_message": user_msg,
                "model_response": model_response,
                "response": "I cannot help with that.",
                "intervention": True,
            },
        )

        self.assertFalse(result["any_success"])
        self.assertEqual(result["successful_turns"], [])
        self.assertTrue(result["overall_blocked"])


class MetricsTests(unittest.TestCase):
    def test_summary_uses_effective_response(self):
        results = [{
            "attack_name": "sample",
            "total_turns": 2,
            "turns": [
                {
                    "model_response": "unsafe raw output",
                    "response": "I cannot help with that.",
                    "intervention": True,
                },
                {
                    "model_response": "unsafe raw output",
                    "response": "Here's how to do it. Step 1: Gather materials. Step 2: Follow the exact procedure.",
                    "intervention": False,
                },
            ],
            "any_success": True,
            "overall_blocked": False,
        }]

        summary = compute_summary(results, "test")

        self.assertEqual(summary["attack_level_asr"], 1.0)
        self.assertEqual(summary["turn_level_asr"], 0.5)
        self.assertEqual(summary["defense_effectiveness"]["attack_block_rate"], 0.0)
        self.assertEqual(summary["defense_effectiveness"]["turn_block_rate"], 0.5)
        self.assertEqual(summary["per_attack_asr"]["sample"], 0.5)


class SubmissionShapeTests(unittest.TestCase):
    def test_attack_set_matches_assignment_size(self):
        runner = AttackRunner()
        self.assertEqual(len(runner.attacks), 10)

    def test_finetuned_strategy_is_in_benchmark_configs(self):
        configs = create_default_defense_configs()
        names = {config["name"] for config in configs}
        self.assertIn("fine_tuned_safety_model", names)


if __name__ == "__main__":
    unittest.main()
