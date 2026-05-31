#!/usr/bin/env python3
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DefendLLMs - Crescendo Jailbreak Defense")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    benchmark_parser = subparsers.add_parser("benchmark", help="Run defense benchmark")
    benchmark_parser.add_argument("--no-simulated", action="store_true", help="Use real model (GPU required)")
    benchmark_parser.add_argument("--save", type=str, default=None, help="Save results to JSON file")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a single defense config")
    eval_parser.add_argument("--defense", type=str, default="full", choices=["baseline", "input", "context", "output", "full", "finetuned"],
                            help="Defense configuration to evaluate")

    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune safety model")
    finetune_parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Base model name")
    finetune_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    finetune_parser.add_argument("--output", type=str, default="./outputs/safety-finetuned", help="Output directory")
    finetune_parser.add_argument("--no-quantize-4bit", action="store_false", dest="quantize_4bit", help="Disable 4-bit quantization")
    finetune_parser.set_defaults(quantize_4bit=True)

    args = parser.parse_args()

    if args.command == "benchmark":
        from benchmarks.run_benchmark import run_benchmark
        run_benchmark(
            use_simulated=not args.no_simulated,
            save_path=args.save,
            verbose=True,
        )

    elif args.command == "evaluate":
        from evaluation.evaluate import DefenseEvaluator, create_default_defense_configs
        from benchmarks.run_benchmark import configure_simulated_evaluator
        configs = create_default_defense_configs()
        config_map = {
            "baseline": configs[0],
            "input": configs[1],
            "context": configs[2],
            "output": configs[3],
            "full": configs[4],
            "finetuned": configs[5],
        }
        config = config_map.get(args.defense, configs[4])
        logger.info(f"Evaluating: {config['name']}")
        evaluator = DefenseEvaluator()
        model_fn = configure_simulated_evaluator(evaluator, config.get("model_profile", "baseline"))
        result = evaluator.evaluate_defense(config, model_fn=model_fn, verbose=True)
        print(f"\nResults for {config['name']}:")
        print(f"  Attack-level ASR: {result['summary']['attack_level_asr']:.3f}")
        print(f"  Turn-level ASR: {result['summary']['turn_level_asr']:.3f}")
        eff = result['summary']['defense_effectiveness']
        print(f"  Block rate: {eff['attack_block_rate']:.3f} (attacks: {eff['blocked_attacks']}/{eff['total_attacks']})")

    elif args.command == "finetune":
        from src.fine_tune import fine_tune_safety_model
        fine_tune_safety_model(
            model_name=args.model,
            output_dir=args.output,
            num_epochs=args.epochs,
            use_4bit=args.quantize_4bit,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
