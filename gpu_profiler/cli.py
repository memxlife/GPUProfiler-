import argparse
import json

from .models import RetryPolicy
from .orchestrator import build_default_orchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-agent GPU auto profiling infra")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a profiling cycle")
    run_parser.add_argument("--workload", required=True, help='Workload command, e.g. "python train.py"')
    run_parser.add_argument("--out", default="profiling_runs", help="Output directory for run artifacts")
    run_parser.add_argument("--samples", type=int, default=3, help="Metric samples per phase")
    run_parser.add_argument("--interval", type=float, default=0.5, help="Seconds between metric samples")
    run_parser.add_argument("--retries", type=int, default=1, help="Retries per task after first failure")
    run_parser.add_argument("--retry-delay", type=float, default=0.1, help="Delay between retries in seconds")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        orchestrator = build_default_orchestrator(
            retry_policy=RetryPolicy(max_retries=args.retries, retry_delay_sec=args.retry_delay)
        )
        result = orchestrator.run_profile(
            workload=args.workload,
            out_dir=args.out,
            samples=args.samples,
            interval_sec=args.interval,
        )
        print(json.dumps(result, indent=2))
