from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from . import __version__


def main() -> int:
    parser = argparse.ArgumentParser(description="AIAA3102 autonomous classification experiment runner")
    parser.add_argument("--version", action="version", version=f"SuperAgent {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("make-demo", help="Generate clearly labeled synthetic development data")
    demo.add_argument("--output", type=Path, default=Path("examples/demo"))
    code_demo = commands.add_parser("make-code-demo", help="Generate independent tabular/text/image code-generation fixtures")
    code_demo.add_argument("--output", type=Path, default=Path("examples/code_demo"))
    run_command = commands.add_parser("run", help="Run autonomous experiments and export the selected model")
    run_command.add_argument("--task", type=Path, required=True)
    run_command.add_argument("--output", type=Path, required=True)
    run_command.add_argument("--planner", choices=["offline", "llm", "llm-code"], default="offline")
    run_command.add_argument("--llm-config", type=Path)
    run_command.add_argument("--experiment-timeout", type=float, default=120)
    run_command.add_argument("--refit-timeout", type=float, default=240)
    batch = commands.add_parser("campaign", help="Compare LLM planners under a shared task budget; validation only")
    batch.add_argument("--task", type=Path, required=True)
    batch.add_argument("--llm-configs", type=Path, nargs="+", required=True)
    batch.add_argument("--planner", choices=["llm", "llm-code"], default="llm", help="llm-code generates and repairs complete Python candidate implementations")
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--budget-rmb", type=float, default=1.0)
    batch.add_argument("--previous", type=Path, nargs="*", default=[])
    batch.add_argument("--reconciliations", type=Path, nargs="*", default=[])
    batch.add_argument("--development-reserves", action="store_true", help="Development only: deduct unresolved transport reservations while retaining unknown usage")
    batch.add_argument("--experiment-timeout", type=float, default=180)
    batch.add_argument("--refit-timeout", type=float, default=300)
    reconcile = commands.add_parser("reconcile-usage", help="Import reviewed provider evidence without changing raw call logs")
    reconcile.add_argument("--run", type=Path, required=True)
    reconcile.add_argument("--evidence", type=Path, required=True)
    reconcile.add_argument("--output", type=Path, required=True)
    config_check = commands.add_parser("check-config", help="Validate model settings and key presence without a network request")
    config_check.add_argument("--llm-config", type=Path, required=True)
    api_check = commands.add_parser("check-api", help="Send one small metered request; no data or training")
    api_check.add_argument("--llm-config", type=Path, required=True)
    api_check.add_argument("--output", type=Path, required=True)
    api_check.add_argument("--max-cost-rmb", type=float, default=0.10)
    evaluate = commands.add_parser("evaluate", help="Evaluate a frozen model once; results never enter the planner")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--labels", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    predict = commands.add_parser("predict", help="Load a locally generated model and predict a new CSV or NPY")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "make-demo":
            from .demo import make_demo
            print("Synthetic development tasks:")
            for path in make_demo(args.output):
                print(path)
        elif args.command == "make-code-demo":
            from .code_demo import make_code_demo
            print("Synthetic code-generation development tasks (not course benchmarks):")
            for path in make_code_demo(args.output):
                print(path)
        elif args.command == "run":
            from .runner import run
            summary = run(args.task, args.output, args.planner, args.llm_config, args.experiment_timeout, args.refit_timeout)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            if summary["status"].startswith("partial_"):
                return 2
        elif args.command == "campaign":
            from .campaign import campaign
            result = campaign(args.task, args.llm_configs, args.output, args.budget_rmb,
                              args.previous, args.experiment_timeout, args.refit_timeout, args.reconciliations,
                              args.development_reserves, args.planner)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result["status"] != "completed":
                return 2
        elif args.command == "reconcile-usage":
            from .reconciliation import reconcile_run_usage
            result = reconcile_run_usage(args.run, args.evidence, args.output)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result["status"] != "completed":
                return 2
        elif args.command == "check-config":
            from .api_check import check_config
            print(json.dumps(check_config(args.llm_config), ensure_ascii=False, indent=2))
        elif args.command == "check-api":
            from .api_check import check_api
            result = check_api(args.llm_config, args.output, args.max_cost_rmb)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result["status"] != "passed":
                return 2
        elif args.command == "evaluate":
            from .evaluation import evaluate_model
            result = evaluate_model(args.model, args.input, args.output, labels_path=args.labels)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            import joblib
            from .runner import export_predictions
            if args.output.exists() or args.output.with_name(args.output.stem + "_probabilities.csv").exists():
                raise ValueError("Prediction output exists; choose a new filename")
            result = export_predictions(joblib.load(args.model), args.input, args.output)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
