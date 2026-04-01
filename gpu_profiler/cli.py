import argparse
from pathlib import Path

from .core.models import RetryPolicy
from .core.store import write_data_artifact
from .runtime.orchestrator import build_default_orchestrator, build_orchestrator_with_planner


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

    autonomous_parser = subparsers.add_parser("autonomous", help="Run autonomous intent-driven profiling")
    autonomous_parser.add_argument("--intent", required=True, help='High-level goal, e.g. "model GB10 performance"')
    autonomous_parser.add_argument("--out", default="profiling_runs", help="Output directory for run artifacts")
    autonomous_parser.add_argument("--samples", type=int, default=3, help="Metric samples around each benchmark")
    autonomous_parser.add_argument("--interval", type=float, default=0.5, help="Seconds between metric samples")
    autonomous_parser.add_argument("--max-iterations", type=int, default=4, help="Maximum autonomous iterations")
    autonomous_parser.add_argument("--max-benchmarks", type=int, default=2, help="Benchmarks proposed per iteration")
    autonomous_parser.add_argument("--target-coverage", type=float, default=0.9, help="Target model coverage score")
    autonomous_parser.add_argument("--planner-backend", choices=["heuristic", "openai"], default="heuristic")
    autonomous_parser.add_argument("--planner-model", default="gpt-5.4", help="OpenAI model when planner-backend=openai")
    autonomous_parser.add_argument("--retries", type=int, default=1, help="Retries per task after first failure")
    autonomous_parser.add_argument("--retry-delay", type=float, default=0.1, help="Delay between retries in seconds")

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
        _write_final_result(result)
    elif args.command == "autonomous":
        orchestrator = build_orchestrator_with_planner(
            retry_policy=RetryPolicy(max_retries=args.retries, retry_delay_sec=args.retry_delay),
            planner_backend=args.planner_backend,
            planner_model=args.planner_model,
        )
        result = orchestrator.run_autonomous_profile(
            intent=args.intent,
            out_dir=args.out,
            samples=args.samples,
            interval_sec=args.interval,
            max_iterations=args.max_iterations,
            max_benchmarks=args.max_benchmarks,
            target_coverage=args.target_coverage,
        )
        _write_final_result(result)


def _write_final_result(result: dict) -> None:
    run_dir = Path(str(result.get("run_dir", "")).strip())
    if not run_dir:
        return
    output_path = run_dir / "final_result.md"
    write_data_artifact(output_path, result, title="Final Result")
