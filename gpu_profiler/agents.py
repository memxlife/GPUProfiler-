import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from .llm import HeuristicWorkflowBackend, LLMWorkflowBackend
from .models import AgentContext, Task
from .store import read_json, write_json, write_text


class Agent:
    name = "agent"

    def can_handle(self, task: Task) -> bool:
        raise NotImplementedError

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        raise NotImplementedError


class PlannerAgent(Agent):
    name = "planner"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "plan"

    def run(self, task: Task, _ctx: AgentContext) -> dict[str, Any]:
        workload = task.payload.get("workload")
        samples = int(task.payload.get("samples", 3))
        interval_sec = float(task.payload.get("interval_sec", 0.5))

        return {
            "stages": [
                [
                    {
                        "kind": "collect_metrics",
                        "payload": {"phase": "baseline", "samples": samples, "interval_sec": interval_sec},
                    },
                    {"kind": "collect_system_info", "payload": {}},
                ],
                [{"kind": "run_workload", "payload": {"command": workload}}],
                [
                    {
                        "kind": "collect_metrics",
                        "payload": {"phase": "post_workload", "samples": samples, "interval_sec": interval_sec},
                    }
                ],
                [{"kind": "analyze", "payload": {}}],
                [{"kind": "report", "payload": {}}],
            ]
        }


class LLMPlanningAgent(Agent):
    name = "llm-planner"

    def __init__(self, workflow_backend: LLMWorkflowBackend | None = None):
        self.workflow_backend = workflow_backend or HeuristicWorkflowBackend()

    def can_handle(self, task: Task) -> bool:
        return task.kind == "llm_plan"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        intent = str(task.payload.get("intent", "")).strip()
        if not intent:
            raise ValueError("llm_plan requires intent")

        iteration = int(task.payload.get("iteration", 0))
        max_iterations = int(task.payload.get("max_iterations", 4))
        max_benchmarks = int(task.payload.get("max_benchmarks", 2))
        kb = task.payload.get("knowledge_base", {})

        decision = self.workflow_backend.propose_plan(
            intent=intent,
            kb=kb,
            iteration=iteration,
            max_iterations=max_iterations,
            max_benchmarks=max_benchmarks,
        )

        plan = {
            "iteration": iteration,
            "intent": intent,
            "planner": decision.planner,
            "stop": decision.stop,
            "reason": decision.reason,
            "focus_dimensions": decision.focus_dimensions,
            "plan_items": decision.plan_items,
        }
        iter_dir = _iteration_dir(ctx.run_dir, iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)
        json_path = iter_dir / "plan.json"
        md_path = iter_dir / "plan.md"
        write_json(json_path, plan)
        write_text(md_path, _render_plan_md(plan))
        plan["artifact"] = str(json_path)
        plan["artifact_md"] = str(md_path)
        return plan


class LLMSchemaContractAgent(Agent):
    name = "llm-schema-contract"

    def __init__(self, workflow_backend: LLMWorkflowBackend | None = None):
        self.workflow_backend = workflow_backend or HeuristicWorkflowBackend()

    def can_handle(self, task: Task) -> bool:
        return task.kind == "llm_schema_contract"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        intent = str(task.payload.get("intent", "")).strip()
        iteration = int(task.payload.get("iteration", 0))
        max_iterations = int(task.payload.get("max_iterations", 4))
        kb = task.payload.get("knowledge_base", {})

        decision = self.workflow_backend.negotiate_schema(
            intent=intent,
            kb=kb,
            iteration=iteration,
            max_iterations=max_iterations,
        )
        result = {
            "iteration": iteration,
            "planner": decision.planner,
            "reason": decision.reason,
            "schema_contract": decision.schema_contract,
        }
        iter_dir = _iteration_dir(ctx.run_dir, iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)
        json_path = iter_dir / "schema_contract.json"
        md_path = iter_dir / "schema_contract.md"
        write_json(json_path, result)
        write_text(md_path, _render_schema_contract_md(result))
        result["artifact"] = str(json_path)
        result["artifact_md"] = str(md_path)
        return result


class LLMResearchAgent(Agent):
    name = "llm-research"

    def __init__(self, workflow_backend: LLMWorkflowBackend | None = None):
        self.workflow_backend = workflow_backend or HeuristicWorkflowBackend()

    def can_handle(self, task: Task) -> bool:
        return task.kind == "llm_research"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        intent = str(task.payload.get("intent", "")).strip()
        iteration = int(task.payload.get("iteration", 0))
        kb = task.payload.get("knowledge_base", {})
        max_sources = int(task.payload.get("max_sources", 8))
        decision = self.workflow_backend.research_context(
            intent=intent,
            kb=kb,
            iteration=iteration,
            max_sources=max_sources,
        )
        result = {
            "iteration": iteration,
            "intent": intent,
            "planner": decision.planner,
            "reason": decision.reason,
            "findings": decision.findings,
            "proposed_dimensions": decision.proposed_dimensions,
        }
        iter_dir = _iteration_dir(ctx.run_dir, iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)
        json_path = iter_dir / "research.json"
        md_path = iter_dir / "research.md"
        write_json(json_path, result)
        write_text(md_path, _render_research_md(result))
        result["artifact"] = str(json_path)
        result["artifact_md"] = str(md_path)
        return result


class LLMTestSuiteAgent(Agent):
    name = "llm-suite"

    def __init__(self, workflow_backend: LLMWorkflowBackend | None = None):
        self.workflow_backend = workflow_backend or HeuristicWorkflowBackend()

    def can_handle(self, task: Task) -> bool:
        return task.kind == "llm_generate_suite"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        intent = str(task.payload.get("intent", "")).strip()
        iteration = int(task.payload.get("iteration", 0))
        kb = task.payload.get("knowledge_base", {})
        plan = task.payload.get("plan", {})
        max_benchmarks = int(task.payload.get("max_benchmarks", 2))
        amendment_round = int(task.payload.get("amendment_round", 0))
        amendment_feedback = task.payload.get("amendment_feedback", [])

        decision = self.workflow_backend.generate_suite(
            intent=intent,
            kb=kb,
            plan=plan,
            iteration=iteration,
            max_benchmarks=max_benchmarks,
        )
        accepted, rejected, policy = _apply_negotiation_policy(
            benchmarks=decision.benchmarks,
            schema_contract=kb.get("schema_contract", {}),
        )

        suite = {
            "iteration": iteration,
            "intent": intent,
            "planner": decision.planner,
            "reason": decision.reason,
            "benchmarks": accepted,
            "rejected_benchmarks": rejected,
            "contract_amendments": decision.contract_amendments,
            "negotiation": {
                "policy": policy,
                "amendment_round": amendment_round,
                "amendment_feedback": amendment_feedback,
                "accepted_count": len(accepted),
                "rejected_count": len(rejected),
            },
        }
        iter_dir = _iteration_dir(ctx.run_dir, iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)
        generated_files = _materialize_generated_files(iter_dir, accepted)
        suite["generated_files"] = generated_files
        json_path = iter_dir / "suite.json"
        md_path = iter_dir / "suite.md"
        write_json(json_path, suite)
        write_text(md_path, _render_suite_md(suite))
        suite["artifact"] = str(json_path)
        suite["artifact_md"] = str(md_path)
        return suite


class MetricsCollectorAgent(Agent):
    name = "collector"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "collect_metrics"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        samples = int(task.payload.get("samples", 3))
        interval_sec = float(task.payload.get("interval_sec", 0.5))
        phase = task.payload.get("phase", "unknown")
        out_name = task.payload.get("out_name", f"metrics_{phase}.json")
        records = self.collect(samples=samples, interval_sec=interval_sec)
        out = ctx.run_dir / out_name
        write_json(out, records)
        return {"phase": phase, "samples": samples, "artifact": str(out), "records": records}

    def collect(self, samples: int, interval_sec: float) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for idx in range(samples):
            records.append(self._sample_gpu())
            if idx != samples - 1:
                time.sleep(interval_sec)
        return records

    def _sample_gpu(self) -> dict[str, Any]:
        now = time.time()
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            line = result.stdout.strip().splitlines()[0]
            gpu_util, mem_util, mem_used, mem_total, temp_c, power_w = [v.strip() for v in line.split(",")]
            parse_errors: list[str] = []
            return {
                "ts": now,
                "source": "nvidia-smi",
                "available": True,
                "gpu_util_pct": self._parse_metric("utilization.gpu", gpu_util, parse_errors),
                "mem_util_pct": self._parse_metric("utilization.memory", mem_util, parse_errors),
                "mem_used_mib": self._parse_metric("memory.used", mem_used, parse_errors),
                "mem_total_mib": self._parse_metric("memory.total", mem_total, parse_errors),
                "temp_c": self._parse_metric("temperature.gpu", temp_c, parse_errors),
                "power_w": self._parse_metric("power.draw", power_w, parse_errors),
                "parse_errors": parse_errors,
            }
        except Exception as exc:  # noqa: BLE001
            return {"ts": now, "source": "nvidia-smi", "available": False, "error": str(exc)}

    def _parse_metric(self, name: str, raw: str, errors: list[str]) -> float | None:
        if raw in {"[N/A]", "N/A", ""}:
            errors.append(f"{name}:unavailable")
            return None
        try:
            return float(raw)
        except Exception:  # noqa: BLE001
            errors.append(f"{name}:invalid({raw})")
            return None


class SystemInfoAgent(Agent):
    name = "system-info"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "collect_system_info"

    def run(self, _task: Task, ctx: AgentContext) -> dict[str, Any]:
        info = {
            "ts": time.time(),
            "python_version": subprocess.run(["python", "--version"], capture_output=True, text=True).stdout.strip()
            or subprocess.run(["python", "--version"], capture_output=True, text=True).stderr.strip(),
        }
        out = ctx.run_dir / "system_info.json"
        write_json(out, info)
        return {"artifact": str(out), "info": info}


class WorkloadRunnerAgent(Agent):
    name = "runner"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "run_workload"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        command = task.payload.get("command")
        if not command:
            raise ValueError("run_workload task missing command")

        start = time.time()
        proc = subprocess.run(shlex.split(command), capture_output=True, text=True)
        elapsed = time.time() - start

        out = {
            "command": command,
            "returncode": proc.returncode,
            "elapsed_sec": elapsed,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "skipped": _is_skipped_workload(proc.stdout, proc.stderr),
        }
        artifact = ctx.run_dir / "workload_result.json"
        artifact_override = task.payload.get("artifact_path")
        if artifact_override:
            artifact = Path(artifact_override)
        write_json(artifact, out)
        out["artifact"] = str(artifact)
        return out


class AnalyzerAgent(Agent):
    name = "analyzer"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "analyze"

    def run(self, _task: Task, ctx: AgentContext) -> dict[str, Any]:
        baseline = read_json(ctx.run_dir / "metrics_baseline.json", [])
        post = read_json(ctx.run_dir / "metrics_post_workload.json", [])
        summary = {"baseline": self._summarize(baseline), "post_workload": self._summarize(post)}
        analysis_file = ctx.run_dir / "analysis.json"
        write_json(analysis_file, summary)
        return {"artifact": str(analysis_file), "summary": summary}

    def _summarize(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [r for r in records if r.get("available")]
        if not valid:
            return {"available": False, "samples": len(records)}
        gpu = self._avg(valid, "gpu_util_pct")
        mem = self._avg(valid, "mem_util_pct")
        power = self._avg(valid, "power_w")
        temp = self._avg(valid, "temp_c")
        has_signal = any(v is not None for v in [gpu, mem, power, temp])
        return {
            "available": has_signal,
            "samples": len(records),
            "gpu_util_avg": gpu,
            "mem_util_avg": mem,
            "power_avg_w": power,
            "temp_avg_c": temp,
        }

    def _avg(self, records: list[dict[str, Any]], key: str) -> float | None:
        values = [float(r[key]) for r in records if r.get(key) is not None]
        if not values:
            return None
        return sum(values) / len(values)


class BenchmarkCycleAgent(Agent):
    name = "benchmark-cycle"

    def __init__(self):
        self.collector = MetricsCollectorAgent()
        self.runner = WorkloadRunnerAgent()
        self.analyzer = AnalyzerAgent()

    def can_handle(self, task: Task) -> bool:
        return task.kind == "run_benchmark_cycle"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        benchmark = task.payload.get("benchmark", {})
        command = benchmark.get("command")
        benchmark_id = benchmark.get("id", "unknown-benchmark")
        iteration = int(task.payload.get("iteration", 0))
        index = int(task.payload.get("index", 0))
        samples = int(task.payload.get("samples", 3))
        interval_sec = float(task.payload.get("interval_sec", 0.5))
        if not command:
            raise ValueError("run_benchmark_cycle missing benchmark.command")

        bench_dir = _benchmark_dir(ctx.run_dir, iteration, index, benchmark_id)
        bench_dir.mkdir(parents=True, exist_ok=True)

        baseline_records = self.collector.collect(samples=samples, interval_sec=interval_sec)
        baseline_path = bench_dir / "metrics_baseline.json"
        write_json(baseline_path, baseline_records)

        workload_task = Task(
            id=f"bench-workload-{iteration}-{index}",
            kind="run_workload",
            payload={"command": command, "artifact_path": str(bench_dir / "workload_result.json")},
        )
        workload_result = self.runner.run(workload_task, ctx)

        post_records = self.collector.collect(samples=samples, interval_sec=interval_sec)
        post_path = bench_dir / "metrics_post_workload.json"
        write_json(post_path, post_records)

        baseline_summary = self.analyzer._summarize(baseline_records)
        post_summary = self.analyzer._summarize(post_records)
        analysis = {
            "baseline": baseline_summary,
            "post_workload": post_summary,
            "delta": {
                "gpu_util_avg": _delta(baseline_summary.get("gpu_util_avg"), post_summary.get("gpu_util_avg")),
                "power_avg_w": _delta(baseline_summary.get("power_avg_w"), post_summary.get("power_avg_w")),
                "temp_avg_c": _delta(baseline_summary.get("temp_avg_c"), post_summary.get("temp_avg_c")),
            },
        }
        analysis_path = bench_dir / "analysis.json"
        write_json(analysis_path, analysis)

        result = {
            "iteration": iteration,
            "benchmark_id": benchmark_id,
            "hypothesis": benchmark.get("hypothesis"),
            "dimensions": benchmark.get("dimensions", []),
            "benchmark_dir": str(bench_dir),
            "baseline_artifact": str(baseline_path),
            "post_artifact": str(post_path),
            "workload_artifact": workload_result.get("artifact"),
            "analysis_artifact": str(analysis_path),
            "analysis": analysis,
            "workload": workload_result,
            "provenance": benchmark.get("provenance", {}),
        }
        out = bench_dir / "benchmark_result.json"
        write_json(out, result)
        result["artifact"] = str(out)
        return result


class BenchmarkExecutorAgent(Agent):
    name = "benchmark-executor"

    def __init__(self):
        self.cycle_agent = BenchmarkCycleAgent()

    def can_handle(self, task: Task) -> bool:
        return task.kind == "execute_suite"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        iteration = int(task.payload.get("iteration", 0))
        samples = int(task.payload.get("samples", 3))
        interval_sec = float(task.payload.get("interval_sec", 0.5))
        suite = task.payload.get("suite", {})
        benchmarks = suite.get("benchmarks", [])

        results: list[dict[str, Any]] = []
        for idx, benchmark in enumerate(benchmarks):
            cycle_task = Task(
                id=f"iter{iteration}-bench-{idx}",
                kind="run_benchmark_cycle",
                payload={
                    "iteration": iteration,
                    "index": idx,
                    "benchmark": benchmark,
                    "samples": samples,
                    "interval_sec": interval_sec,
                },
            )
            results.append(self.cycle_agent.run(cycle_task, ctx))

        iter_dir = _iteration_dir(ctx.run_dir, iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)
        out = iter_dir / "suite_results.json"
        payload = {"iteration": iteration, "benchmarks_run": len(results), "results": results}
        write_json(out, payload)
        payload["artifact"] = str(out)
        return payload


class LLMAnalysisAgent(Agent):
    name = "llm-analysis"

    def __init__(self, workflow_backend: LLMWorkflowBackend | None = None):
        self.workflow_backend = workflow_backend or HeuristicWorkflowBackend()

    def can_handle(self, task: Task) -> bool:
        return task.kind == "llm_analyze_update"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        intent = str(task.payload.get("intent", "")).strip()
        iteration = int(task.payload.get("iteration", 0))
        max_iterations = int(task.payload.get("max_iterations", 4))
        kb_path = Path(task.payload.get("kb_path") or (ctx.run_dir / "performance_model.json"))
        kb = read_json(kb_path, {})
        plan = task.payload.get("plan", {})
        suite_results = task.payload.get("suite_results", [])

        decision = self.workflow_backend.analyze_results(
            intent=intent,
            kb=kb,
            plan=plan,
            suite_results=suite_results,
            iteration=iteration,
            max_iterations=max_iterations,
        )

        covered = sorted(set(kb.get("covered_dimensions", [])).union(decision.covered_dimensions))
        target = kb.get("target_dimensions", [])
        if not target:
            target = _infer_target_dimensions(plan=plan, suite_results=suite_results, claims=decision.claims)
            kb["target_dimensions"] = target
        coverage_score = 0.0
        if target:
            coverage_score = len(set(target).intersection(covered)) / len(target)

        kb.setdefault("history", [])
        kb.setdefault("claims", [])
        kb["covered_dimensions"] = covered
        kb["coverage_score"] = coverage_score
        kb["last_summary"] = decision.summary
        kb["last_planner"] = decision.planner
        kb["claims"].extend(decision.claims)
        kb["history"].append(
            {
                "iteration": iteration,
                "summary": decision.summary,
                "claims_added": len(decision.claims),
                "covered_dimensions": covered,
                "coverage_score": coverage_score,
                "stop": decision.stop,
                "reason": decision.reason,
                "veto_next_plan": decision.veto_next_plan,
                "veto_reason": decision.veto_reason,
                "required_observability": decision.required_observability,
                "contract_amendments": decision.contract_amendments,
                "timestamp": time.time(),
            }
        )
        write_json(kb_path, kb)

        iter_dir = _iteration_dir(ctx.run_dir, iteration)
        analysis_path = iter_dir / "analysis_update.json"
        out = {
            "iteration": iteration,
            "summary": decision.summary,
            "claims_added": len(decision.claims),
            "covered_dimensions": covered,
            "coverage_score": coverage_score,
            "stop": decision.stop,
            "reason": decision.reason,
            "veto_next_plan": decision.veto_next_plan,
            "veto_reason": decision.veto_reason,
            "required_observability": decision.required_observability,
            "contract_amendments": decision.contract_amendments,
            "planner": decision.planner,
            "kb_artifact": str(kb_path),
        }
        write_json(analysis_path, out)
        out["artifact"] = str(analysis_path)
        return out


class AutonomousReporterAgent(Agent):
    name = "autonomous-reporter"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "autonomous_report"

    def run(self, _task: Task, ctx: AgentContext) -> dict[str, Any]:
        model = read_json(ctx.run_dir / "performance_model.json", {})
        lines = [
            f"# Autonomous GPU Performance Model ({ctx.run_id})",
            "",
            "## Intent",
            f"- `{model.get('intent', '')}`",
            "",
            "## Coverage",
            f"- target_dimensions: `{model.get('target_dimensions', [])}`",
            f"- covered_dimensions: `{model.get('covered_dimensions', [])}`",
            f"- coverage_score: `{model.get('coverage_score', 0.0)}`",
            "",
            "## History",
        ]
        for item in model.get("history", []):
            lines.append(
                f"- iter {item.get('iteration')}: coverage={item.get('coverage_score')}, claims_added={item.get('claims_added')}, stop={item.get('stop')}"
            )
        lines.extend(["", "## Claims", f"- total: `{len(model.get('claims', []))}`"])
        report_path = ctx.run_dir / "autonomous_report.md"
        write_text(report_path, "\n".join(lines))
        return {"artifact": str(report_path)}


class ReporterAgent(Agent):
    name = "reporter"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "report"

    def run(self, _task: Task, ctx: AgentContext) -> dict[str, Any]:
        analysis = read_json(ctx.run_dir / "analysis.json", {})
        workload = read_json(ctx.run_dir / "workload_result.json", {})
        lines = [
            f"# GPU Auto Profiling Report ({ctx.run_id})",
            "",
            "## Workload",
            f"- command: `{workload.get('command', 'N/A')}`",
            f"- return code: `{workload.get('returncode', 'N/A')}`",
            f"- elapsed sec: `{round(workload.get('elapsed_sec', 0.0), 3)}`",
            "",
            "## Metrics Summary",
            f"- baseline: `{analysis.get('baseline', {})}`",
            f"- post_workload: `{analysis.get('post_workload', {})}`",
            "",
            "## Artifacts",
            "- metrics_baseline.json",
            "- system_info.json",
            "- workload_result.json",
            "- metrics_post_workload.json",
            "- analysis.json",
            "- run_log.json",
        ]
        report_path = ctx.run_dir / "report.md"
        write_text(report_path, "\n".join(lines))
        return {"artifact": str(report_path)}


def default_agents(workflow_backend: LLMWorkflowBackend | None = None) -> list[Agent]:
    backend = workflow_backend or HeuristicWorkflowBackend()
    return [
        PlannerAgent(),
        LLMSchemaContractAgent(workflow_backend=backend),
        LLMResearchAgent(workflow_backend=backend),
        LLMPlanningAgent(workflow_backend=backend),
        LLMTestSuiteAgent(workflow_backend=backend),
        MetricsCollectorAgent(),
        SystemInfoAgent(),
        WorkloadRunnerAgent(),
        BenchmarkCycleAgent(),
        BenchmarkExecutorAgent(),
        AnalyzerAgent(),
        LLMAnalysisAgent(workflow_backend=backend),
        AutonomousReporterAgent(),
        ReporterAgent(),
    ]


def _iteration_dir(run_dir: Path, iteration: int) -> Path:
    return run_dir / "iterations" / f"iter_{iteration:02d}"


def _benchmark_dir(run_dir: Path, iteration: int, index: int, benchmark_id: str) -> Path:
    safe_id = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in benchmark_id)
    return _iteration_dir(run_dir, iteration) / f"bench_{index:02d}_{safe_id}"


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def _is_skipped_workload(stdout: str, stderr: str) -> bool:
    for text in [stdout or "", stderr or ""]:
        for line in text.splitlines():
            if line.strip().startswith("SKIP:"):
                return True
    return False


def _default_negotiation_policy() -> dict[str, Any]:
    return {
        "thresholds": {
            "coverage_gain_min": 0.45,
            "implementability_min": 0.55,
            "observability_min": 0.5,
            "utility_min": 0.55,
        },
        "weights": {
            "coverage_gain_score": 0.4,
            "implementability_score": 0.3,
            "observability_score": 0.3,
        },
        "max_amendment_rounds": 2,
    }


def _coerce_unit(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        out = default
    return max(0.0, min(1.0, out))


def _policy_from_contract(schema_contract: dict[str, Any]) -> dict[str, Any]:
    base = _default_negotiation_policy()
    negotiation = schema_contract.get("negotiation_policy", {}) if isinstance(schema_contract, dict) else {}
    if not isinstance(negotiation, dict):
        return base
    thresholds = negotiation.get("thresholds", {})
    weights = negotiation.get("weights", {})
    merged = {
        **base,
        **{k: v for k, v in negotiation.items() if k not in {"thresholds", "weights"}},
        "thresholds": {**base["thresholds"], **(thresholds if isinstance(thresholds, dict) else {})},
        "weights": {**base["weights"], **(weights if isinstance(weights, dict) else {})},
    }
    merged["thresholds"] = {k: _coerce_unit(v, base["thresholds"].get(k, 0.5)) for k, v in merged["thresholds"].items()}
    merged["weights"] = {k: _coerce_unit(v, base["weights"].get(k, 0.0)) for k, v in merged["weights"].items()}
    try:
        merged["max_amendment_rounds"] = max(0, int(merged.get("max_amendment_rounds", 2)))
    except Exception:
        merged["max_amendment_rounds"] = 2
    return merged


def _utility(scores: dict[str, Any], weights: dict[str, Any]) -> float:
    cov = _coerce_unit(scores.get("coverage_gain_score"), 0.0)
    imp = _coerce_unit(scores.get("implementability_score"), 0.0)
    obs = _coerce_unit(scores.get("observability_score"), 0.0)
    w_cov = _coerce_unit(weights.get("coverage_gain_score"), 0.4)
    w_imp = _coerce_unit(weights.get("implementability_score"), 0.3)
    w_obs = _coerce_unit(weights.get("observability_score"), 0.3)
    total = w_cov + w_imp + w_obs
    if total <= 0:
        return round((cov + imp + obs) / 3.0, 4)
    return round((cov * w_cov + imp * w_imp + obs * w_obs) / total, 4)


def _apply_negotiation_policy(
    benchmarks: list[dict[str, Any]], schema_contract: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    policy = _policy_from_contract(schema_contract)
    thresholds = policy.get("thresholds", {})
    weights = policy.get("weights", {})

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        scores = benchmark.get("scores", {}) if isinstance(benchmark.get("scores", {}), dict) else {}
        cov = _coerce_unit(scores.get("coverage_gain_score"), 0.5)
        imp = _coerce_unit(scores.get("implementability_score"), 0.5)
        obs = _coerce_unit(scores.get("observability_score"), 0.5)
        util = _utility(scores, weights)
        reasons: list[str] = []
        if cov < _coerce_unit(thresholds.get("coverage_gain_min"), 0.45):
            reasons.append("coverage_gain_below_min")
        if imp < _coerce_unit(thresholds.get("implementability_min"), 0.55):
            reasons.append("implementability_below_min")
        if obs < _coerce_unit(thresholds.get("observability_min"), 0.5):
            reasons.append("observability_below_min")
        if util < _coerce_unit(thresholds.get("utility_min"), 0.55):
            reasons.append("utility_below_min")

        benchmark["utility_score"] = util
        if reasons:
            rejected.append(
                {
                    "id": benchmark.get("id"),
                    "dimensions": benchmark.get("dimensions", []),
                    "scores": {
                        "coverage_gain_score": cov,
                        "implementability_score": imp,
                        "observability_score": obs,
                        "utility_score": util,
                    },
                    "reasons": reasons,
                }
            )
            continue
        accepted.append(benchmark)

    return accepted, rejected, policy


def _render_plan_md(plan: dict[str, Any]) -> str:
    lines = [
        f"# Iteration {plan.get('iteration')} Plan",
        "",
        f"- planner: `{plan.get('planner')}`",
        f"- stop: `{plan.get('stop')}`",
        f"- reason: {plan.get('reason')}",
        "",
        "## Focus Dimensions",
        f"- {plan.get('focus_dimensions', [])}",
        "",
        "## Plan Items",
    ]
    for item in plan.get("plan_items", []):
        lines.append(
            f"- objective: {item.get('objective')} | dimension: {item.get('dimension')} | success: {item.get('success_criteria')}"
        )
    return "\n".join(lines)


def _render_research_md(research: dict[str, Any]) -> str:
    lines = [
        f"# Iteration {research.get('iteration')} Research",
        "",
        f"- planner: `{research.get('planner')}`",
        f"- reason: {research.get('reason')}",
        "",
        "## Proposed Dimensions",
        f"- {research.get('proposed_dimensions', [])}",
        "",
        "## Findings",
    ]
    for item in research.get("findings", []):
        lines.extend(
            [
                f"- title: {item.get('title')}",
                f"- relevance: {item.get('relevance')}",
                f"- source: {item.get('source_url')}",
                f"- summary: {item.get('summary')}",
                "",
            ]
        )
    return "\n".join(lines)


def _render_suite_md(suite: dict[str, Any]) -> str:
    lines = [
        f"# Iteration {suite.get('iteration')} Suite",
        "",
        f"- planner: `{suite.get('planner')}`",
        f"- reason: {suite.get('reason')}",
        "",
        "## Benchmarks",
    ]
    for bench in suite.get("benchmarks", []):
        lines.extend(
            [
                f"- id: `{bench.get('id')}`",
                f"- dimensions: `{bench.get('dimensions', [])}`",
                f"- hypothesis: {bench.get('hypothesis')}",
                f"- scores: `{bench.get('scores', {})}`",
                f"- utility_score: `{bench.get('utility_score')}`",
                f"- command: `{bench.get('command')}`",
                "",
            ]
        )
    if suite.get("rejected_benchmarks"):
        lines.extend(["## Rejected Benchmarks"])
        for item in suite.get("rejected_benchmarks", []):
            lines.append(
                f"- id: `{item.get('id')}` | reasons: `{item.get('reasons', [])}` | scores: `{item.get('scores', {})}`"
            )
    if suite.get("negotiation"):
        lines.extend(["## Negotiation", f"- {suite.get('negotiation')}"])
    if suite.get("contract_amendments"):
        lines.extend(["## Contract Amendments"])
        for item in suite.get("contract_amendments", []):
            lines.append(
                f"- path: `{item.get('path')}` | change: {item.get('change')} | priority: `{item.get('priority')}`"
            )
    if suite.get("generated_files"):
        lines.extend(["## Generated Files"])
        for path in suite.get("generated_files", []):
            lines.append(f"- `{path}`")
    return "\n".join(lines)


def _render_schema_contract_md(contract: dict[str, Any]) -> str:
    lines = [
        f"# Iteration {contract.get('iteration')} Schema Contract",
        "",
        f"- planner: `{contract.get('planner')}`",
        f"- reason: {contract.get('reason')}",
        "",
        "## Contract",
        "```json",
        json.dumps(contract.get("schema_contract", {}), indent=2),
        "```",
    ]
    return "\n".join(lines)


def _materialize_generated_files(iter_dir: Path, benchmarks: list[dict[str, Any]]) -> list[str]:
    written: list[str] = []
    for bench in benchmarks:
        bench_id = str(bench.get("id", "benchmark"))
        for file_spec in bench.get("files", []):
            path = str(file_spec.get("path", "")).strip()
            content = str(file_spec.get("content", ""))
            if not path or not content:
                continue
            if ".." in path or path.startswith("/"):
                continue
            if not path.endswith((".cu", ".md", ".json")):
                continue
            out_path = iter_dir / bench_id / path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_text(out_path, content)
            written.append(str(out_path))
    return written


def _infer_target_dimensions(
    plan: dict[str, Any], suite_results: list[dict[str, Any]], claims: list[dict[str, Any]]
) -> list[str]:
    dims: list[str] = []
    for d in plan.get("focus_dimensions", []):
        dim = str(d).strip()
        if dim and dim not in dims:
            dims.append(dim)
    for res in suite_results:
        for d in res.get("dimensions", []):
            dim = str(d).strip()
            if dim and dim not in dims:
                dims.append(dim)
    for claim in claims:
        for d in claim.get("dimensions", []):
            dim = str(d).strip()
            if dim and dim not in dims:
                dims.append(dim)
    return dims
