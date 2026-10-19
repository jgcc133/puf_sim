"""Command-line entry point for adaptive PUF machine-learning attacks."""
from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from .experiment import DEFAULT_ATTACK_FAMILIES, run_ml_attack_experiment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Quick FF-APUF sanity run:\n"
            "    python -m puf_sim.puf_ml_attack --families ff_apuf --quick-run --print-progress\n\n"
            "  Full XPU deep-neural FF run:\n"
            "    python -m puf_sim.puf_ml_attack --families ff_apuf --attack-model auto "
            "--backend torch_xpu --device 0 --n 64 --response-bits 10 --k 2 "
            "--training-samples 100000 --validation-samples 30000 --test-samples 20000 "
            "--batch-size 256 --learning-rate 4e-4 --width 128 --max-depth 1 --epochs 200 "
            "--puf-instances 1 --print-progress\n"
        ),
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(DEFAULT_ATTACK_FAMILIES),
        help="PUF families to attack. Use --list-families to print supported defaults.",
    )
    parser.add_argument(
        "--list-families",
        action="store_true",
        help="Print supported default family names and exit.",
    )
    parser.add_argument(
        "--quick-run",
        action="store_true",
        help=(
            "Override to a small evaluator-friendly run: "
            "5000/1000/1000 samples, 1 instance, max_depth=1, agents=1."
        ),
    )
    parser.add_argument("--scenario", default="general")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--training-samples", type=int, default=10000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--puf-instances", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--response-bits", type=int, default=1)
    parser.add_argument("--noisiness", type=float, default=0.0)
    sram_hash_group = parser.add_mutually_exclusive_group()
    sram_hash_group.add_argument(
        "--sram-hash",
        dest="hash_responses",
        action="store_true",
        help="SHA-256 hash SRAM responses (default).",
    )
    sram_hash_group.add_argument(
        "--no-sram-hash",
        dest="hash_responses",
        action="store_false",
        help="Return direct SRAM challenge/fingerprint XOR responses.",
    )
    parser.set_defaults(hash_responses=True)
    parser.add_argument("--success-threshold", type=float, default=0.95)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--width", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--backend", choices=("auto", "cpu", "torch_xpu"), default="auto")
    parser.add_argument("--device", type=int)
    parser.add_argument(
        "--attack-model",
        choices=("auto", "neural", "svm", "deep_neural", "staged_split"),
        default="auto",
        help=(
            "Attack model; staged_split alternates the Interpose upstream and "
            "downstream logistic components and requires --families interpose."
        ),
    )
    parser.add_argument("--report-root")
    parser.add_argument("--no-save-report", action="store_true")
    parser.add_argument("--print-progress", action="store_true")
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print resolved run configuration before starting.",
    )
    return parser


def _apply_quick_run(arguments: argparse.Namespace) -> None:
    if not arguments.quick_run:
        return
    arguments.training_samples = 5000
    arguments.validation_samples = 1000
    arguments.test_samples = 1000
    arguments.puf_instances = 1
    arguments.max_depth = 1
    arguments.agents = 1


def _resolved_config(arguments: argparse.Namespace) -> dict[str, object]:
    return {
        "families": arguments.families,
        "scenario": arguments.scenario,
        "n": arguments.n,
        "training_samples": arguments.training_samples,
        "validation_samples": arguments.validation_samples,
        "test_samples": arguments.test_samples,
        "puf_instances": arguments.puf_instances,
        "seed": arguments.seed,
        "k": arguments.k,
        "response_bits": arguments.response_bits,
        "noisiness": arguments.noisiness,
        "hash_responses": arguments.hash_responses,
        "success_threshold": arguments.success_threshold,
        "max_depth": arguments.max_depth,
        "agents": arguments.agents,
        "width": arguments.width,
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "backend": arguments.backend,
        "device": arguments.device,
        "attack_model": arguments.attack_model,
        "report_root": arguments.report_root,
        "save_report": not arguments.no_save_report,
        "print_progress": arguments.print_progress,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.list_families:
        print("Supported default families:")
        for family in DEFAULT_ATTACK_FAMILIES:
            print(f" - {family}")
        return 0
    _apply_quick_run(arguments)
    if arguments.show_config:
        print("Resolved configuration:")
        print(json.dumps(_resolved_config(arguments), indent=2))
    results = run_ml_attack_experiment(
        families=arguments.families,
        n=arguments.n,
        training_samples=arguments.training_samples,
        validation_samples=arguments.validation_samples,
        test_samples=arguments.test_samples,
        seed=arguments.seed,
        k=arguments.k,
        response_bits=arguments.response_bits,
        noisiness=arguments.noisiness,
        hash_responses=arguments.hash_responses,
        success_threshold=arguments.success_threshold,
        max_depth=arguments.max_depth,
        agents=arguments.agents,
        width=arguments.width,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        backend=arguments.backend,
        device=arguments.device,
        scenario=arguments.scenario,
        report_root=arguments.report_root,
        save_report=not arguments.no_save_report,
        puf_instances=arguments.puf_instances,
        print_progress=arguments.print_progress,
        attack_model=arguments.attack_model,
    )
    for family, result in results.items():
        family_model = result["instances"][0]["model_architecture"] if result["instances"] else arguments.attack_model
        print(
            f"{family}: mean test accuracy {result['mean_test_accuracy']:.3%} "
            f"across {result['puf_instances']} instances "
            f"(required per-instance threshold "
            f"{result['success_threshold']:.1%}); "
            f"mean response-element accuracy "
            f"{result['mean_test_element_accuracy']:.3%}; "
            f"{result['epochs_trained']} total epochs "
            f"using {family_model}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())