#!/usr/bin/env python3
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.openai_research_probe import (
    build_probe_knowledge_base,
    build_research_request,
    load_queries,
    masked_env,
    parse_timeout_values,
    slugify,
    summarize_latencies,
)
from gpu_profiler.workflow.llm import OpenAIWorkflowBackend

AGENT_CHOICES = (
    "plan_research_request",
    "research_context",
    "plan_benchmark",
    "generate_implementation",
    "analyze_results",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe OpenAI-backed planner, research, codegen, and analysis calls with comparable timing sweeps."
    )
    parser.add_argument("--model", default="gpt-5.4", help="Model name to call.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Base per-request timeout in seconds.")
    parser.add_argument(
        "--timeout-value",
        action="append",
        default=[],
        help="Timeout value to include in a sensitivity sweep. Repeat this flag for multiple values.",
    )
    parser.add_argument(
        "--agent",
        action="append",
        choices=AGENT_CHOICES,
        default=[],
        help="Agent/backend method to probe. Repeat this flag for multiple methods.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many times to run each condition.")
    parser.add_argument(
        "--intent",
        default="Develop a performance model for the local GPU",
        help="Base intent used to build probe fixtures.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Human query to probe. Repeat this flag for multiple queries.",
    )
    parser.add_argument(
        "--control-query",
        action="append",
        default=[],
        help="Additional control query to include in the sweep.",
    )
    parser.add_argument(
        "--queries-file",
        default="",
        help="Optional text file with one query per line. Blank lines and lines starting with # are ignored.",
    )
    parser.add_argument("--max-sources", type=int, default=8, help="Maximum findings to request for research probes.")
    parser.add_argument("--out", default="agent_probe_runs", help="Directory where probe artifacts will be written.")
    return parser


def agent_dimension_slug(agent_name: str, query: str) -> str:
    return f"{agent_name}_{slugify(query, limit=32)}"


def build_probe_plan(agent_name: str, query: str) -> dict[str, Any]:
    dimension = agent_dimension_slug(agent_name, query)
    return {
        "benchmark_plan": {
            "target_nodes": [dimension],
            "benchmarks": [
                {
                    "id": f"{agent_name}_benchmark_plan_0",
                    "title": query,
                    "objective": query,
                    "hypothesis": query,
                    "target_node_ids": [dimension],
                    "benchmark_role": "baseline",
                }
            ],
        },
        "benchmark_plan_md_artifact": "",
    }


def build_execution_results(agent_name: str, query: str) -> list[dict[str, Any]]:
    dimension = agent_dimension_slug(agent_name, query)
    return [
        {
            "iteration": 0,
            "benchmark_id": f"{agent_name}_benchmark_plan_0",
            "hypothesis": query,
            "dimensions": [dimension],
            "workload": {
                "command": "echo simulated benchmark",
                "returncode": 0,
                "stdout": f"simulated_result={query}",
                "stderr": "",
                "skipped": False,
                "artifact": f"simulated/{dimension}/workload_result.json",
            },
            "artifact": f"simulated/{dimension}/benchmark_result.json",
        }
    ]


def build_agent_fixture(agent_name: str, intent: str, query: str, max_sources: int) -> dict[str, Any]:
    kb = build_probe_knowledge_base(intent)
    dimension = agent_dimension_slug(agent_name, query)
    kb["target_dimensions"] = [dimension]
    kb["current_knowledge_model"]["focus_nodes"] = [dimension]
    plan = build_probe_plan(agent_name, query)
    research_request = build_research_request(intent, query)
    fixture = {
        "intent": intent,
        "kb": kb,
        "iteration": 0,
        "max_iterations": 1,
        "max_benchmarks": 1,
        "max_sources": max_sources,
        "research_request": research_request,
        "research_request_memo": query,
        "research_memo": query,
        "planning_memo": query,
        "plan": plan,
        "execution_results": build_execution_results(agent_name, query),
    }
    return fixture


def evaluate_agent_result(agent_name: str, result: Any) -> dict[str, Any]:
    if agent_name == "plan_research_request":
        request = result.research_request or {}
        target_questions = [str(x).strip() for x in request.get("target_questions", []) if str(x).strip()]
        request_summary = str(request.get("request_summary", "")).strip()
        meaningful = bool(result.reason.strip()) and bool(request_summary) and bool(target_questions)
        return {
            "meaningful": meaningful,
            "reason_present": bool(result.reason.strip()),
            "request_summary_present": bool(request_summary),
            "target_question_count": len(target_questions),
        }
    if agent_name == "research_context":
        findings = result.findings if isinstance(result.findings, list) else []
        unique_sources = sorted(
            {
                str(item.get("source_url", "")).strip()
                for item in findings
                if isinstance(item, dict) and str(item.get("source_url", "")).strip()
            }
        )
        meaningful = bool(result.reason.strip()) and len(findings) >= 1 and len(unique_sources) >= 1
        return {
            "meaningful": meaningful,
            "reason_present": bool(result.reason.strip()),
            "findings_count": len(findings),
            "unique_source_count": len(unique_sources),
        }
    if agent_name == "plan_benchmark":
        benchmark_plan = result.benchmark_plan if isinstance(result.benchmark_plan, dict) else {}
        benchmarks = benchmark_plan.get("benchmarks", []) if isinstance(benchmark_plan.get("benchmarks", []), list) else []
        meaningful = bool(result.reason.strip()) and len(benchmarks) >= 1
        return {
            "meaningful": meaningful,
            "reason_present": bool(result.reason.strip()),
            "benchmark_plan_count": len(benchmarks),
            "target_node_count": len(benchmark_plan.get("target_nodes", [])) if isinstance(benchmark_plan.get("target_nodes", []), list) else 0,
        }
    if agent_name == "generate_implementation":
        benchmarks = result.benchmarks if isinstance(result.benchmarks, list) else []
        runnable = sum(1 for item in benchmarks if str(item.get("command", "")).strip())
        meaningful = bool(result.reason.strip()) and runnable >= 1
        return {
            "meaningful": meaningful,
            "reason_present": bool(result.reason.strip()),
            "benchmark_count": len(benchmarks),
            "runnable_benchmark_count": runnable,
        }
    if agent_name == "analyze_results":
        covered = result.covered_dimensions if isinstance(result.covered_dimensions, list) else []
        claims = result.claims if isinstance(result.claims, list) else []
        meaningful = bool(result.summary.strip()) and (len(covered) >= 1 or len(claims) >= 1)
        return {
            "meaningful": meaningful,
            "summary_present": bool(result.summary.strip()),
            "claim_count": len(claims),
            "covered_dimension_count": len(covered),
            "stop": bool(result.stop),
        }
    raise ValueError(f"Unsupported agent name: {agent_name}")


def serialize_agent_result(agent_name: str, result: Any) -> dict[str, Any]:
    if agent_name == "plan_research_request":
        return {
            "reason": result.reason,
            "research_request": result.research_request,
            "planner": result.planner,
            "raw_response": result.raw_response,
        }
    if agent_name == "research_context":
        return {
            "reason": result.reason,
            "request_summary": result.request_summary,
            "unanswered_questions": result.unanswered_questions,
            "findings": result.findings,
            "proposed_dimensions": result.proposed_dimensions,
            "planner": result.planner,
            "raw_response": result.raw_response,
        }
    if agent_name == "plan_benchmark":
        return {
            "reason": result.reason,
            "benchmark_plan": result.benchmark_plan,
            "planner": result.planner,
            "raw_response": result.raw_response,
        }
    if agent_name == "generate_implementation":
        return {
            "reason": result.reason,
            "benchmarks": result.benchmarks,
            "negotiation": result.negotiation,
            "contract_amendments": result.contract_amendments,
            "planner": result.planner,
            "raw_response": result.raw_response,
        }
    if agent_name == "analyze_results":
        return {
            "summary": result.summary,
            "claims": result.claims,
            "covered_dimensions": result.covered_dimensions,
            "stop": result.stop,
            "reason": result.reason,
            "veto_next_plan": result.veto_next_plan,
            "veto_reason": result.veto_reason,
            "required_observability": result.required_observability,
            "contract_amendments": result.contract_amendments,
            "planner": result.planner,
        }
    raise ValueError(f"Unsupported agent name: {agent_name}")


def run_agent_probe_attempt(
    backend: OpenAIWorkflowBackend,
    *,
    agent_name: str,
    intent: str,
    query: str,
    max_sources: int,
) -> dict[str, Any]:
    fixture = build_agent_fixture(agent_name, intent, query, max_sources)
    started = time.time()
    try:
        if agent_name == "plan_research_request":
            result = backend.plan_research_request(
                intent=fixture["intent"],
                kb=fixture["kb"],
                iteration=fixture["iteration"],
                max_iterations=fixture["max_iterations"],
                max_benchmarks=fixture["max_benchmarks"],
            )
        elif agent_name == "research_context":
            result = backend.research_context(
                intent=fixture["intent"],
                kb=fixture["kb"],
                iteration=fixture["iteration"],
                research_request=fixture["research_request"],
                research_request_memo=fixture["research_request_memo"],
                max_sources=fixture["max_sources"],
            )
        elif agent_name == "plan_benchmark":
            result = backend.plan_benchmark(
                intent=fixture["intent"],
                kb=fixture["kb"],
                iteration=fixture["iteration"],
                max_iterations=fixture["max_iterations"],
                max_benchmarks=fixture["max_benchmarks"],
                research_memo=fixture["research_memo"],
            )
        elif agent_name == "generate_implementation":
            result = backend.generate_implementation(
                intent=fixture["intent"],
                kb=fixture["kb"],
                plan=fixture["plan"],
                iteration=fixture["iteration"],
                max_benchmarks=fixture["max_benchmarks"],
                planning_memo=fixture["planning_memo"],
            )
        elif agent_name == "analyze_results":
            result = backend.analyze_results(
                intent=fixture["intent"],
                kb=fixture["kb"],
                plan=fixture["plan"],
                execution_results=fixture["execution_results"],
                iteration=fixture["iteration"],
                max_iterations=fixture["max_iterations"],
            )
        else:
            raise ValueError(f"Unsupported agent name: {agent_name}")
        elapsed = time.time() - started
        evaluation = evaluate_agent_result(agent_name, result)
        return {
            "status": "ok",
            "agent": agent_name,
            "elapsed_sec": elapsed,
            "query": query,
            "intent": intent,
            "request_timeout_sec": backend.request_timeout_sec,
            "fixture": fixture,
            "result": serialize_agent_result(agent_name, result),
            "evaluation": evaluation,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "agent": agent_name,
            "elapsed_sec": time.time() - started,
            "query": query,
            "intent": intent,
            "request_timeout_sec": backend.request_timeout_sec,
            "fixture": fixture,
            "error_type": type(exc).__name__,
            "error": repr(exc),
        }


def summarize_agent_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "attempt_count": len(attempts),
        "ok_count": sum(1 for item in attempts if item.get("status") == "ok"),
        "meaningful_count": sum(1 for item in attempts if bool(item.get("evaluation", {}).get("meaningful"))),
        "error_count": sum(1 for item in attempts if item.get("status") != "ok"),
        "latency_summary_sec": summarize_latencies([float(item.get("elapsed_sec", 0.0)) for item in attempts]),
        "per_agent": [],
        "per_agent_timeout": [],
        "per_agent_query": [],
    }
    per_agent: dict[str, dict[str, Any]] = {}
    per_agent_timeout: dict[str, dict[str, Any]] = {}
    per_agent_query: dict[str, dict[str, Any]] = {}
    for item in attempts:
        agent_name = str(item.get("agent", "")).strip()
        query = str(item.get("query", "")).strip()
        timeout_sec = float(item.get("request_timeout_sec", 0.0))
        meaningful = bool(item.get("evaluation", {}).get("meaningful"))
        for key, bucket_map, seed in (
            (
                agent_name,
                per_agent,
                {"agent": agent_name, "attempts": 0, "ok": 0, "meaningful": 0, "latencies_sec": []},
            ),
            (
                f"{agent_name}||{timeout_sec:.3f}",
                per_agent_timeout,
                {
                    "agent": agent_name,
                    "timeout_sec": timeout_sec,
                    "attempts": 0,
                    "ok": 0,
                    "meaningful": 0,
                    "latencies_sec": [],
                },
            ),
            (
                f"{agent_name}||{query}",
                per_agent_query,
                {
                    "agent": agent_name,
                    "query": query,
                    "attempts": 0,
                    "ok": 0,
                    "meaningful": 0,
                    "latencies_sec": [],
                },
            ),
        ):
            bucket = bucket_map.setdefault(key, seed)
            bucket["attempts"] += 1
            bucket["latencies_sec"].append(float(item.get("elapsed_sec", 0.0)))
            if item.get("status") == "ok":
                bucket["ok"] += 1
            if meaningful:
                bucket["meaningful"] += 1
    for bucket_map, key in (
        (per_agent, "per_agent"),
        (per_agent_timeout, "per_agent_timeout"),
        (per_agent_query, "per_agent_query"),
    ):
        values: list[dict[str, Any]] = []
        for bucket in bucket_map.values():
            bucket["latency_summary_sec"] = summarize_latencies(bucket.pop("latencies_sec"))
            values.append(bucket)
        if key == "per_agent":
            values.sort(key=lambda item: item["agent"])
        elif key == "per_agent_timeout":
            values.sort(key=lambda item: (item["agent"], float(item["timeout_sec"])))
        else:
            values.sort(key=lambda item: (item["agent"], item["query"]))
        summary[key] = values
    return summary


def write_summary_markdown(report: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# OpenAI Agent Probe",
        "",
        f"- model: `{report['model']}`",
        f"- agents: `{', '.join(report['agents'])}`",
        f"- repeats: `{report['repeats']}`",
        f"- query_count: `{len(report['queries'])}`",
        f"- ok_count: `{report['summary']['ok_count']}`",
        f"- meaningful_count: `{report['summary']['meaningful_count']}`",
        f"- error_count: `{report['summary']['error_count']}`",
        "",
        "## Overall Latency",
        "",
        f"- min: `{report['summary']['latency_summary_sec']['min']:.3f}` s",
        f"- median: `{report['summary']['latency_summary_sec']['median']:.3f}` s",
        f"- p95: `{report['summary']['latency_summary_sec']['p95']:.3f}` s",
        f"- max: `{report['summary']['latency_summary_sec']['max']:.3f}` s",
        "",
        "## Per Agent",
        "",
    ]
    for item in report["summary"]["per_agent"]:
        lines.extend(
            [
                f"### {item['agent']}",
                "",
                f"- attempts: `{item['attempts']}`",
                f"- ok: `{item['ok']}`",
                f"- meaningful: `{item['meaningful']}`",
                f"- min_latency: `{item['latency_summary_sec']['min']:.3f}` s",
                f"- median_latency: `{item['latency_summary_sec']['median']:.3f}` s",
                f"- p95_latency: `{item['latency_summary_sec']['p95']:.3f}` s",
                f"- max_latency: `{item['latency_summary_sec']['max']:.3f}` s",
                "",
            ]
        )
    lines.extend(["## Per Agent x Timeout", ""])
    for item in report["summary"]["per_agent_timeout"]:
        lines.extend(
            [
                f"### {item['agent']} @ timeout={item['timeout_sec']:.3f}s",
                "",
                f"- attempts: `{item['attempts']}`",
                f"- ok: `{item['ok']}`",
                f"- meaningful: `{item['meaningful']}`",
                f"- min_latency: `{item['latency_summary_sec']['min']:.3f}` s",
                f"- median_latency: `{item['latency_summary_sec']['median']:.3f}` s",
                f"- p95_latency: `{item['latency_summary_sec']['p95']:.3f}` s",
                f"- max_latency: `{item['latency_summary_sec']['max']:.3f}` s",
                "",
            ]
        )
    lines.extend(["## Per Agent x Query", ""])
    for item in report["summary"]["per_agent_query"]:
        lines.extend(
            [
                f"### {item['agent']} :: {item['query']}",
                "",
                f"- attempts: `{item['attempts']}`",
                f"- ok: `{item['ok']}`",
                f"- meaningful: `{item['meaningful']}`",
                f"- min_latency: `{item['latency_summary_sec']['min']:.3f}` s",
                f"- median_latency: `{item['latency_summary_sec']['median']:.3f}` s",
                f"- p95_latency: `{item['latency_summary_sec']['p95']:.3f}` s",
                f"- max_latency: `{item['latency_summary_sec']['max']:.3f}` s",
                "",
            ]
        )
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    agents = list(dict.fromkeys(args.agent or list(AGENT_CHOICES)))
    queries = load_queries(args.query + list(args.control_query), args.queries_file)
    if not queries:
        raise SystemExit("Provide at least one --query or a non-empty --queries-file.")
    timeout_values = parse_timeout_values(args.timeout, list(args.timeout_value))

    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir = out_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    attempts: list[dict[str, Any]] = []
    for timeout_index, timeout_value in enumerate(timeout_values):
        backend = OpenAIWorkflowBackend(model=args.model, request_timeout_sec=timeout_value)
        for agent_index, agent_name in enumerate(agents):
            for query_index, query in enumerate(queries):
                for attempt_index in range(max(1, args.repeats)):
                    result = run_agent_probe_attempt(
                        backend,
                        agent_name=agent_name,
                        intent=args.intent,
                        query=query,
                        max_sources=args.max_sources,
                    )
                    result["timeout_index"] = timeout_index
                    result["agent_index"] = agent_index
                    result["query_index"] = query_index
                    result["attempt"] = attempt_index + 1
                    attempts.append(result)
                    stem = (
                        f"t{timeout_index:02d}_g{agent_index:02d}_q{query_index:02d}_a{attempt_index + 1:02d}_"
                        f"{slugify(agent_name, limit=20)}_{slugify(query)}.json"
                    )
                    (attempts_dir / stem).write_text(json.dumps(result, indent=2), encoding="utf-8")

    report = {
        "tool": "openai_agent_probe",
        "python": sys.version.split()[0],
        "model": args.model,
        "timeout_sec": args.timeout,
        "timeout_values": timeout_values,
        "agents": agents,
        "repeats": args.repeats,
        "intent": args.intent,
        "queries": queries,
        "max_sources": args.max_sources,
        "env": masked_env(),
        "started_at": started_at.isoformat(),
        "run_dir": str(out_dir),
        "attempts": attempts,
        "summary": summarize_agent_attempts(attempts),
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary_markdown(report, out_dir / "summary.md")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["ok_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
