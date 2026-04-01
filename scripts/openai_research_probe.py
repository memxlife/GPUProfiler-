#!/usr/bin/env python3
import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from gpu_profiler.workflow.llm import OpenAIWorkflowBackend, ResearchDecision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe the OpenAI-backed research agent with user-supplied queries and measure latency."
    )
    parser.add_argument("--model", default="gpt-5.4", help="Model name to call.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--timeout-value",
        action="append",
        default=[],
        help="Timeout value to include in a sensitivity sweep. Repeat this flag for multiple values.",
    )
    parser.add_argument("--max-sources", type=int, default=8, help="Maximum findings to request.")
    parser.add_argument("--repeats", type=int, default=1, help="How many times to run each query.")
    parser.add_argument(
        "--intent",
        default="Measure research-agent latency and result quality for GPU performance-modeling queries.",
        help="High-level intent supplied to the research backend.",
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
        help="Additional control query to include in the sweep. Repeat this flag for multiple controls.",
    )
    parser.add_argument(
        "--queries-file",
        default="",
        help="Optional text file with one query per line. Blank lines and lines starting with # are ignored.",
    )
    parser.add_argument(
        "--out",
        default="research_probe_runs",
        help="Directory where probe artifacts will be written.",
    )
    parser.add_argument(
        "--min-findings",
        type=int,
        default=1,
        help="Minimum number of findings required for a result to count as meaningful.",
    )
    parser.add_argument(
        "--min-unique-sources",
        type=int,
        default=1,
        help="Minimum number of unique source URLs required for a result to count as meaningful.",
    )
    return parser


def masked_env() -> dict[str, Any]:
    return {
        "OPENAI_API_KEY_present": bool(os.environ.get("OPENAI_API_KEY")),
        "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", ""),
        "HTTPS_PROXY": os.environ.get("HTTPS_PROXY", ""),
        "HTTP_PROXY": os.environ.get("HTTP_PROXY", ""),
        "ALL_PROXY": os.environ.get("ALL_PROXY", ""),
        "NO_PROXY": os.environ.get("NO_PROXY", ""),
    }


def load_queries(inline_queries: list[str], queries_file: str = "") -> list[str]:
    queries: list[str] = []
    for item in inline_queries:
        text = str(item).strip()
        if text:
            queries.append(text)
    if queries_file:
        for raw_line in Path(queries_file).read_text(encoding="utf-8").splitlines():
            text = raw_line.strip()
            if text and not text.startswith("#"):
                queries.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in queries:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def parse_timeout_values(default_timeout: float, explicit_values: list[str]) -> list[float]:
    values: list[float] = []
    if explicit_values:
        for item in explicit_values:
            try:
                parsed = float(item)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid timeout value: {item}") from exc
            if parsed <= 0:
                raise ValueError(f"Timeout values must be positive: {item}")
            values.append(parsed)
    else:
        values.append(float(default_timeout))
    deduped: list[float] = []
    seen: set[float] = set()
    for item in values:
        rounded = round(float(item), 6)
        if rounded not in seen:
            seen.add(rounded)
            deduped.append(float(item))
    return deduped


def build_probe_knowledge_base(intent: str) -> dict[str, Any]:
    return {
        "intent": intent,
        "target_dimensions": [],
        "covered_dimensions": [],
        "coverage_score": 0.0,
        "target_coverage": 1.0,
        "available_tools": {
            "nvidia-smi": shutil.which("nvidia-smi") is not None,
            "ncu": shutil.which("ncu") is not None,
            "nsys": shutil.which("nsys") is not None,
            "python": shutil.which("python") is not None,
        },
        "history": [],
        "claims": [],
        "research_history": [],
        "current_knowledge_model": {
            "intent": {"summary": intent},
            "domain_hierarchy": [],
            "focus_nodes": [],
            "generated_at": "",
            "planner_notes": "Research probe created a minimal empty knowledge base.",
        },
    }


def build_research_request(intent: str, query: str) -> dict[str, Any]:
    return {
        "intent_summary": intent,
        "request_summary": query,
        "target_nodes": [],
        "target_questions": [query],
        "search_topics": [query],
        "source_preferences": [
            "vendor documentation",
            "profiler documentation",
            "papers",
            "benchmark methodology references",
        ],
        "source_constraints": [],
        "expected_outputs": [
            "Concise findings with source URLs",
            "Remaining unanswered questions",
            "Candidate performance-model dimensions if applicable",
        ],
        "notes": "Direct probe query supplied by a human operator.",
    }


def evaluate_research_result(
    decision: ResearchDecision,
    min_findings: int = 1,
    min_unique_sources: int = 1,
) -> dict[str, Any]:
    source_urls = [
        str(item.get("source_url", "")).strip()
        for item in decision.findings
        if isinstance(item, dict) and str(item.get("source_url", "")).strip()
    ]
    unique_sources = sorted(set(source_urls))
    findings_count = len(decision.findings)
    meaningful = (
        bool(decision.reason.strip())
        and bool(decision.request_summary.strip())
        and findings_count >= max(0, int(min_findings))
        and len(unique_sources) >= max(0, int(min_unique_sources))
    )
    return {
        "meaningful": meaningful,
        "reason_present": bool(decision.reason.strip()),
        "request_summary_present": bool(decision.request_summary.strip()),
        "findings_count": findings_count,
        "unique_source_count": len(unique_sources),
        "unanswered_question_count": len(decision.unanswered_questions),
        "proposed_dimension_count": len(decision.proposed_dimensions),
        "unique_sources": unique_sources,
    }


def slugify(text: str, limit: int = 48) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text))
    while "--" in clean:
        clean = clean.replace("--", "-")
    clean = clean.strip("-")
    return (clean or "query")[:limit].rstrip("-") or "query"


def run_probe_attempt(
    backend: OpenAIWorkflowBackend,
    *,
    intent: str,
    query: str,
    max_sources: int,
    min_findings: int,
    min_unique_sources: int,
) -> dict[str, Any]:
    kb = build_probe_knowledge_base(intent)
    research_request = build_research_request(intent, query)
    started = time.time()
    try:
        decision = backend.research_context(
            intent=intent,
            kb=kb,
            iteration=0,
            research_request=research_request,
            research_request_memo=query,
            max_sources=max_sources,
        )
        elapsed = time.time() - started
        evaluation = evaluate_research_result(
            decision,
            min_findings=min_findings,
            min_unique_sources=min_unique_sources,
        )
        return {
            "status": "ok",
            "elapsed_sec": elapsed,
            "query": query,
            "intent": intent,
            "request_timeout_sec": backend.request_timeout_sec,
            "research_request": research_request,
            "decision": {
                "reason": decision.reason,
                "request_summary": decision.request_summary,
                "unanswered_questions": decision.unanswered_questions,
                "findings": decision.findings,
                "proposed_dimensions": decision.proposed_dimensions,
                "planner": decision.planner,
                "raw_response": decision.raw_response,
            },
            "evaluation": evaluation,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "elapsed_sec": time.time() - started,
            "query": query,
            "intent": intent,
            "request_timeout_sec": backend.request_timeout_sec,
            "research_request": research_request,
            "error_type": type(exc).__name__,
            "error": repr(exc),
        }


def summarize_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(item["elapsed_sec"]) for item in attempts]
    ok_attempts = [item for item in attempts if item.get("status") == "ok"]
    meaningful_attempts = [
        item for item in ok_attempts if bool(item.get("evaluation", {}).get("meaningful"))
    ]
    per_query: dict[str, dict[str, Any]] = {}
    per_timeout: dict[str, dict[str, Any]] = {}
    per_query_timeout: dict[str, dict[str, Any]] = {}
    for item in attempts:
        query = str(item.get("query", "")).strip()
        timeout_key = f"{float(item.get('request_timeout_sec', 0.0)):.3f}"
        bucket = per_query.setdefault(
            query,
            {"attempts": 0, "ok": 0, "meaningful": 0, "latencies_sec": []},
        )
        bucket["attempts"] += 1
        bucket["latencies_sec"].append(float(item.get("elapsed_sec", 0.0)))
        if item.get("status") == "ok":
            bucket["ok"] += 1
        if bool(item.get("evaluation", {}).get("meaningful")):
            bucket["meaningful"] += 1
        timeout_bucket = per_timeout.setdefault(
            timeout_key,
            {"attempts": 0, "ok": 0, "meaningful": 0, "latencies_sec": [], "timeout_sec": float(timeout_key)},
        )
        timeout_bucket["attempts"] += 1
        timeout_bucket["latencies_sec"].append(float(item.get("elapsed_sec", 0.0)))
        if item.get("status") == "ok":
            timeout_bucket["ok"] += 1
        if bool(item.get("evaluation", {}).get("meaningful")):
            timeout_bucket["meaningful"] += 1
        qt_key = f"{query}||{timeout_key}"
        qt_bucket = per_query_timeout.setdefault(
            qt_key,
            {
                "query": query,
                "timeout_sec": float(timeout_key),
                "attempts": 0,
                "ok": 0,
                "meaningful": 0,
                "latencies_sec": [],
            },
        )
        qt_bucket["attempts"] += 1
        qt_bucket["latencies_sec"].append(float(item.get("elapsed_sec", 0.0)))
        if item.get("status") == "ok":
            qt_bucket["ok"] += 1
        if bool(item.get("evaluation", {}).get("meaningful")):
            qt_bucket["meaningful"] += 1
    for query, bucket in per_query.items():
        bucket["latency_summary_sec"] = summarize_latencies(bucket.pop("latencies_sec"))
        bucket["query"] = query
    for bucket in per_timeout.values():
        bucket["latency_summary_sec"] = summarize_latencies(bucket.pop("latencies_sec"))
    for bucket in per_query_timeout.values():
        bucket["latency_summary_sec"] = summarize_latencies(bucket.pop("latencies_sec"))
    return {
        "attempt_count": len(attempts),
        "ok_count": len(ok_attempts),
        "meaningful_count": len(meaningful_attempts),
        "error_count": len(attempts) - len(ok_attempts),
        "latency_summary_sec": summarize_latencies(latencies),
        "per_query": list(per_query.values()),
        "per_timeout": sorted(per_timeout.values(), key=lambda item: float(item["timeout_sec"])),
        "per_query_timeout": sorted(
            per_query_timeout.values(),
            key=lambda item: (str(item["query"]), float(item["timeout_sec"])),
        ),
    }


def summarize_latencies(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(float(item) for item in latencies)
    p95_index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * 0.95))))
    return {
        "min": min(ordered),
        "median": median(ordered),
        "p95": ordered[p95_index],
        "max": max(ordered),
    }


def write_summary_markdown(report: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# OpenAI Research Probe",
        "",
        f"- model: `{report['model']}`",
        f"- timeout_sec: `{report['timeout_sec']}`",
        f"- repeats: `{report['repeats']}`",
        f"- max_sources: `{report['max_sources']}`",
        f"- query_count: `{len(report['queries'])}`",
        f"- ok_count: `{report['summary']['ok_count']}`",
        f"- meaningful_count: `{report['summary']['meaningful_count']}`",
        f"- error_count: `{report['summary']['error_count']}`",
        "",
        "## Latency Summary",
        "",
        f"- min: `{report['summary']['latency_summary_sec']['min']:.3f}` s",
        f"- median: `{report['summary']['latency_summary_sec']['median']:.3f}` s",
        f"- p95: `{report['summary']['latency_summary_sec']['p95']:.3f}` s",
        f"- max: `{report['summary']['latency_summary_sec']['max']:.3f}` s",
        "",
        "## Per Timeout",
        "",
    ]
    for item in report["summary"]["per_timeout"]:
        lines.extend(
            [
                f"### timeout={item['timeout_sec']:.3f}s",
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
    lines.extend(
        [
            "## Per Query",
            "",
        ]
    )
    for item in report["summary"]["per_query"]:
        lines.extend(
            [
                f"### {item['query']}",
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
    lines.extend(
        [
            "## Query x Timeout",
            "",
        ]
    )
    for item in report["summary"]["per_query_timeout"]:
        lines.extend(
            [
                f"### {item['query']} @ timeout={item['timeout_sec']:.3f}s",
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
        for query_index, query in enumerate(queries):
            for attempt_index in range(max(1, args.repeats)):
                result = run_probe_attempt(
                    backend,
                    intent=args.intent,
                    query=query,
                    max_sources=args.max_sources,
                    min_findings=args.min_findings,
                    min_unique_sources=args.min_unique_sources,
                )
                result["query_index"] = query_index
                result["attempt"] = attempt_index + 1
                result["timeout_index"] = timeout_index
                attempts.append(result)
                stem = (
                    f"t{timeout_index:02d}_q{query_index:02d}_a{attempt_index + 1:02d}_"
                    f"{slugify(query)}.json"
                )
                (attempts_dir / stem).write_text(json.dumps(result, indent=2), encoding="utf-8")

    report = {
        "tool": "openai_research_probe",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "model": args.model,
        "timeout_sec": args.timeout,
        "timeout_values": timeout_values,
        "max_sources": args.max_sources,
        "repeats": args.repeats,
        "intent": args.intent,
        "queries": queries,
        "env": masked_env(),
        "started_at": started_at.isoformat(),
        "run_dir": str(out_dir),
        "attempts": attempts,
        "summary": summarize_attempts(attempts),
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary_markdown(report, out_dir / "summary.md")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["ok_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
