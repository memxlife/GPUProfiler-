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
            "reason": decision.reason,
            "knowledge_model": decision.knowledge_model,
            "proposal": decision.proposal,
            "research_request": decision.research_request,
        }
        iter_dir = _iteration_dir(ctx.run_dir, iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)
        model_path = iter_dir / "knowledge_model.json"
        proposal_path = iter_dir / "proposal.json"
        proposal_md_path = iter_dir / "proposal.md"
        research_request_path = iter_dir / "research_request.json"
        write_json(model_path, decision.knowledge_model)
        write_json(proposal_path, decision.proposal)
        write_text(proposal_md_path, _render_proposal_md(plan))
        if isinstance(decision.research_request, dict):
            write_json(research_request_path, decision.research_request)
            plan["research_request_artifact"] = str(research_request_path)
        else:
            plan["research_request_artifact"] = None
        plan["knowledge_model_artifact"] = str(model_path)
        plan["proposal_artifact"] = str(proposal_path)
        plan["proposal_md_artifact"] = str(proposal_md_path)
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
        iteration = int(task.payload.get("iteration", 0))
        kb = task.payload.get("knowledge_base", {})
        request_path = task.payload.get("research_request_artifact")
        if not request_path:
            raise ValueError("llm_research requires research_request_artifact")
        research_request = read_json(Path(request_path), {})
        if not isinstance(research_request, dict) or not research_request:
            raise ValueError("research_request_artifact did not contain a valid request object")
        intent = str(research_request.get("intent_summary", task.payload.get("intent", ""))).strip()
        max_sources = int(task.payload.get("max_sources", 8))
        decision = self.workflow_backend.research_context(
            intent=intent,
            kb=kb,
            iteration=iteration,
            research_request=research_request,
            max_sources=max_sources,
        )
        result = {
            "iteration": iteration,
            "intent": intent,
            "planner": decision.planner,
            "reason": decision.reason,
            "request_summary": decision.request_summary,
            "unanswered_questions": decision.unanswered_questions,
            "findings": decision.findings,
            "proposed_dimensions": decision.proposed_dimensions,
            "research_request_artifact": str(request_path),
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


class LLMCodegenAgent(Agent):
    name = "llm-codegen"

    def __init__(self, workflow_backend: LLMWorkflowBackend | None = None):
        self.workflow_backend = workflow_backend or HeuristicWorkflowBackend()

    def can_handle(self, task: Task) -> bool:
        return task.kind == "llm_generate_implementation"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        intent = str(task.payload.get("intent", "")).strip()
        iteration = int(task.payload.get("iteration", 0))
        kb = task.payload.get("knowledge_base", {})
        plan = task.payload.get("plan", {})
        max_benchmarks = int(task.payload.get("max_benchmarks", 2))
        amendment_round = int(task.payload.get("amendment_round", 0))
        amendment_feedback = task.payload.get("amendment_feedback", [])

        decision = self.workflow_backend.generate_implementation(
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
        accepted = [_annotate_feasibility(item, policy) for item in accepted]
        feasibility_report = _build_feasibility_report(
            accepted=accepted,
            rejected=rejected,
            proposal=plan.get("proposal", {}),
        )

        implementation = {
            "iteration": iteration,
            "intent": intent,
            "planner": decision.planner,
            "reason": decision.reason,
            "benchmarks": accepted,
            "rejected_benchmarks": rejected,
            "contract_amendments": decision.contract_amendments,
            "feasibility_summary": feasibility_report.get("summary", {}),
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
        implementation["generated_files"] = generated_files
        json_path = iter_dir / "implementation.json"
        md_path = iter_dir / "implementation.md"
        feasibility_path = iter_dir / "feasibility_report.json"
        write_json(json_path, implementation)
        write_json(feasibility_path, feasibility_report)
        write_text(md_path, _render_implementation_md(implementation))
        implementation["artifact"] = str(json_path)
        implementation["artifact_md"] = str(md_path)
        implementation["feasibility_artifact"] = str(feasibility_path)
        return implementation


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

        result = {
            "iteration": iteration,
            "benchmark_id": benchmark_id,
            "hypothesis": benchmark.get("hypothesis"),
            "dimensions": benchmark.get("dimensions", []),
            "benchmark_dir": str(bench_dir),
            "baseline_artifact": str(baseline_path),
            "post_artifact": str(post_path),
            "workload_artifact": workload_result.get("artifact"),
            "raw_artifacts": [
                {"path": str(baseline_path), "kind": "metric"},
                {"path": str(post_path), "kind": "metric"},
                {"path": str(workload_result.get("artifact", "")), "kind": "log"},
            ],
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
        return task.kind == "execute_implementation"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        iteration = int(task.payload.get("iteration", 0))
        samples = int(task.payload.get("samples", 3))
        interval_sec = float(task.payload.get("interval_sec", 0.5))
        implementation = task.payload.get("implementation", task.payload.get("suite", {}))
        benchmarks = implementation.get("benchmarks", [])

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
        out = iter_dir / "execution_results.json"
        md_out = iter_dir / "execution.md"
        payload = {"iteration": iteration, "benchmarks_run": len(results), "results": results}
        write_json(out, payload)
        write_text(md_out, _render_execution_md(payload))
        payload["artifact"] = str(out)
        payload["artifact_md"] = str(md_out)
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
        execution_results = task.payload.get("execution_results", [])

        decision = self.workflow_backend.analyze_results(
            intent=intent,
            kb=kb,
            plan=plan,
            execution_results=execution_results,
            iteration=iteration,
            max_iterations=max_iterations,
        )

        covered = sorted(set(kb.get("covered_dimensions", [])).union(decision.covered_dimensions))
        target = kb.get("target_dimensions", [])
        if not target:
            target = _infer_target_dimensions(plan=plan, execution_results=execution_results, claims=decision.claims)
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
        analysis_md_path = iter_dir / "analysis.md"
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
        write_text(analysis_md_path, _render_analysis_md(out))
        out["artifact"] = str(analysis_path)
        out["artifact_md"] = str(analysis_md_path)
        return out


class AutonomousReporterAgent(Agent):
    name = "autonomous-reporter"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "autonomous_report"

    def run(self, _task: Task, ctx: AgentContext) -> dict[str, Any]:
        model = read_json(ctx.run_dir / "performance_model.json", {})
        run_state = model.get("run_state", {}) if isinstance(model.get("run_state", {}), dict) else {}
        lines = [
            f"# Autonomous GPU Performance Model ({ctx.run_id})",
            "",
            "## Intent",
            f"- `{model.get('intent', '')}`",
            "",
            "## Run State",
            f"- status: `{run_state.get('status', '')}`",
            f"- reason: `{run_state.get('reason', '')}`",
            f"- iterations_completed: `{run_state.get('iterations_completed', 0)}`",
            f"- planner_calls: `{run_state.get('planner_calls', 0)}`",
            f"- search_calls: `{run_state.get('search_calls', 0)}`",
            f"- codegen_calls: `{run_state.get('codegen_calls', 0)}`",
            f"- runner_calls: `{run_state.get('runner_calls', 0)}`",
            f"- analyzer_calls: `{run_state.get('analyzer_calls', 0)}`",
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
        LLMCodegenAgent(workflow_backend=backend),
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


def _implementation_complexity(benchmark: dict[str, Any]) -> str:
    files = benchmark.get("files", []) if isinstance(benchmark.get("files", []), list) else []
    command = str(benchmark.get("command", "")).strip().lower()
    command_length = len(command)
    file_count = len(files)
    profiler_count = sum(1 for tool in ["ncu", "nsys"] if tool in command)
    if file_count >= 5 or command_length > 500 or profiler_count >= 2:
        return "excessive"
    if file_count >= 4 or command_length > 260 or profiler_count == 1:
        return "high"
    if file_count >= 3 or command_length > 120:
        return "medium"
    return "low"


def _annotate_feasibility(benchmark: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    out = dict(benchmark)
    thresholds = policy.get("thresholds", {}) if isinstance(policy, dict) else {}
    impl_min = _coerce_unit(thresholds.get("implementability_min"), 0.55)
    scores = out.get("scores", {}) if isinstance(out.get("scores", {}), dict) else {}
    implementability = _coerce_unit(scores.get("implementability_score"), 0.5)
    complexity = _implementation_complexity(out)
    status = "feasible"
    if complexity in {"high", "excessive"} or implementability < max(impl_min + 0.1, 0.75):
        status = "feasible_with_revision"
    out["implementation_complexity"] = complexity
    out["feasibility_status"] = status
    return out


def _build_feasibility_report(
    accepted: list[dict[str, Any]], rejected: list[dict[str, Any]], proposal: dict[str, Any]
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    proposal_artifact = ""
    if isinstance(proposal, dict):
        proposal_artifact = str(proposal.get("artifact", "")).strip()
    for item in accepted:
        benchmark_id = str(item.get("id", "")).strip()
        complexity = str(item.get("implementation_complexity", "medium")).strip() or "medium"
        feasibility_status = str(item.get("feasibility_status", "feasible")).strip() or "feasible"
        recommended_changes: list[str] = []
        if feasibility_status == "feasible_with_revision":
            recommended_changes.append("Reduce implementation complexity or split this proposal into smaller steps.")
        items.append(
            {
                "proposal_id": benchmark_id,
                "feasibility_status": feasibility_status,
                "implementation_complexity": complexity,
                "summary": "Implementation can be generated under the current codegen constraints.",
                "blocking_issues": [],
                "evidence_refs": [str(entry.get("path", "")).strip() for entry in item.get("files", []) if str(entry.get("path", "")).strip()],
                "planner_feedback": {
                    "revise": feasibility_status != "feasible",
                    "recommended_changes": recommended_changes,
                },
            }
        )
    for item in rejected:
        reasons = [str(x).strip() for x in item.get("reasons", []) if str(x).strip()]
        recommend = ["Revise the proposal to improve implementability and reduce implementation risk."]
        if "implementability_below_min" in reasons:
            recommend.append("Simplify the implementation scope or reduce dependency/tooling assumptions.")
        if "utility_below_min" in reasons or "coverage_gain_below_min" in reasons:
            recommend.append("Clarify why this benchmark is needed at the current curriculum stage.")
        items.append(
            {
                "proposal_id": str(item.get("id", "")).strip(),
                "feasibility_status": "not_feasible",
                "implementation_complexity": "excessive" if "implementability_below_min" in reasons else "high",
                "summary": "Implementation was rejected during codegen acceptance checks.",
                "blocking_issues": reasons,
                "evidence_refs": [],
                "planner_feedback": {
                    "revise": True,
                    "recommended_changes": recommend,
                },
            }
        )
    summary = {
        "item_count": len(items),
        "feasible_count": sum(1 for item in items if item.get("feasibility_status") == "feasible"),
        "revision_needed_count": sum(1 for item in items if item.get("feasibility_status") == "feasible_with_revision"),
        "not_feasible_count": sum(1 for item in items if item.get("feasibility_status") == "not_feasible"),
    }
    return {
        "proposal_artifact": proposal_artifact,
        "items": items,
        "summary": summary,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _render_proposal_md(plan: dict[str, Any]) -> str:
    proposal = plan.get("proposal", {}) if isinstance(plan.get("proposal", {}), dict) else {}
    knowledge_model = plan.get("knowledge_model", {}) if isinstance(plan.get("knowledge_model", {}), dict) else {}
    lines = [
        f"# Iteration {plan.get('iteration')} Proposal",
        "",
        f"- planner: `{plan.get('planner')}`",
        f"- reason: {plan.get('reason')}",
        "",
        "## Knowledge Model",
        f"- focus_nodes: `{knowledge_model.get('focus_nodes', [])}`",
        f"- domain_nodes: `{len(knowledge_model.get('domain_hierarchy', []))}`",
        "",
        "## Proposal Summary",
        f"- {proposal.get('proposal_summary', '')}",
        "",
        "## Proposed Benchmarks",
    ]
    for item in proposal.get("proposals", []):
        lines.append(
            f"- title: {item.get('title')} | targets: {item.get('target_node_ids', [])} | role: {item.get('benchmark_role')}"
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


def _render_implementation_md(implementation: dict[str, Any]) -> str:
    lines = [
        f"# Iteration {implementation.get('iteration')} Implementation",
        "",
        f"- planner: `{implementation.get('planner')}`",
        f"- reason: {implementation.get('reason')}",
        "",
        "## Benchmarks",
    ]
    for bench in implementation.get("benchmarks", []):
        lines.extend(
            [
                f"- id: `{bench.get('id')}`",
                f"- dimensions: `{bench.get('dimensions', [])}`",
                f"- hypothesis: {bench.get('hypothesis')}",
                f"- feasibility_status: `{bench.get('feasibility_status')}`",
                f"- implementation_complexity: `{bench.get('implementation_complexity')}`",
                f"- scores: `{bench.get('scores', {})}`",
                f"- utility_score: `{bench.get('utility_score')}`",
                f"- command: `{bench.get('command')}`",
                "",
            ]
        )
    if implementation.get("feasibility_summary"):
        lines.extend(["## Feasibility Summary", f"- {implementation.get('feasibility_summary')}"])
    if implementation.get("rejected_benchmarks"):
        lines.extend(["## Rejected Benchmarks"])
        for item in implementation.get("rejected_benchmarks", []):
            lines.append(
                f"- id: `{item.get('id')}` | reasons: `{item.get('reasons', [])}` | scores: `{item.get('scores', {})}`"
            )
    if implementation.get("negotiation"):
        lines.extend(["## Negotiation", f"- {implementation.get('negotiation')}"])
    if implementation.get("contract_amendments"):
        lines.extend(["## Contract Amendments"])
        for item in implementation.get("contract_amendments", []):
            lines.append(
                f"- path: `{item.get('path')}` | change: {item.get('change')} | priority: `{item.get('priority')}`"
            )
    if implementation.get("generated_files"):
        lines.extend(["## Generated Files"])
        for path in implementation.get("generated_files", []):
            lines.append(f"- `{path}`")
    return "\n".join(lines)


def _render_execution_md(execution: dict[str, Any]) -> str:
    lines = [
        f"# Iteration {execution.get('iteration')} Execution",
        "",
        f"- benchmarks_run: `{execution.get('benchmarks_run', 0)}`",
        "",
        "## Results",
    ]
    for result in execution.get("results", []):
        workload = result.get("workload", {}) if isinstance(result.get("workload", {}), dict) else {}
        lines.extend(
            [
                f"- benchmark_id: `{result.get('benchmark_id')}`",
                f"- dimensions: `{result.get('dimensions', [])}`",
                f"- returncode: `{workload.get('returncode')}`",
                f"- skipped: `{workload.get('skipped', False)}`",
                f"- elapsed_sec: `{workload.get('elapsed_sec')}`",
                f"- raw_artifacts: `{result.get('raw_artifacts', [])}`",
                "",
            ]
        )
    return "\n".join(lines)


def _render_analysis_md(analysis: dict[str, Any]) -> str:
    lines = [
        f"# Iteration {analysis.get('iteration')} Analysis",
        "",
        f"- planner: `{analysis.get('planner')}`",
        f"- summary: {analysis.get('summary')}",
        f"- claims_added: `{analysis.get('claims_added', 0)}`",
        f"- covered_dimensions: `{analysis.get('covered_dimensions', [])}`",
        f"- coverage_score: `{analysis.get('coverage_score', 0.0)}`",
        f"- stop: `{analysis.get('stop', False)}`",
        f"- reason: {analysis.get('reason')}",
    ]
    veto_reason = str(analysis.get("veto_reason", "")).strip()
    if veto_reason:
        lines.append(f"- veto_reason: {veto_reason}")
    required_observability = analysis.get("required_observability", [])
    if required_observability:
        lines.extend(["", "## Required Observability"])
        for item in required_observability:
            lines.append(f"- {item}")
    amendments = analysis.get("contract_amendments", [])
    if amendments:
        lines.extend(["", "## Contract Amendments"])
        for item in amendments:
            lines.append(
                f"- path: `{item.get('path')}` | change: {item.get('change')} | priority: `{item.get('priority')}`"
            )
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
    plan: dict[str, Any], execution_results: list[dict[str, Any]], claims: list[dict[str, Any]]
) -> list[str]:
    dims: list[str] = []
    proposal = plan.get("proposal", {}) if isinstance(plan.get("proposal", {}), dict) else {}
    for d in proposal.get("target_nodes", []):
        dim = str(d).strip()
        if dim and dim not in dims:
            dims.append(dim)
    for res in execution_results:
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
