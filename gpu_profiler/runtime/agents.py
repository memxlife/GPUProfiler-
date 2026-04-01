import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from ..workflow.llm import (
    CODEGEN_SYSTEM_PROMPT,
    HeuristicWorkflowBackend,
    LLMWorkflowBackend,
    _compact_codegen_kb,
    _compact_codegen_plan,
    _enforce_payload_budget,
    _benchmark_plan_focus_nodes,
    _render_codegen_prompt,
    _slugify_dimension,
    _trim_codegen_payload,
    _trim_text,
    CODEGEN_INPUT_HARD_CAP_CHARS,
    CODEGEN_INPUT_TARGET_CHARS,
)
from ..knowledge.knowledge_base import load_markdown_knowledge_base_memos, update_markdown_knowledge_base
from ..core.models import AgentContext, Task
from ..core.store import read_json, write_json, write_text

PASSWORDLESS_SUDO_NCU_CANDIDATES = (
    "/usr/local/cuda/bin/ncu",
    "/usr/local/cuda-13.0/bin/ncu",
)


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
        return task.kind in {"llm_plan_research", "llm_plan_benchmark", "llm_plan"}

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        intent = str(task.payload.get("intent", "")).strip()
        if not intent:
            raise ValueError(f"{task.kind} requires intent")

        iteration = int(task.payload.get("iteration", 0))
        max_iterations = int(task.payload.get("max_iterations", 4))
        max_benchmarks = int(task.payload.get("max_benchmarks", 2))
        kb = task.payload.get("knowledge_base", {})
        kb = {**kb, **load_markdown_knowledge_base_memos(kb)} if isinstance(kb, dict) else {}
        iter_dir = _iteration_dir(ctx.run_dir, iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)
        if task.kind == "llm_plan_research":
            decision = self.workflow_backend.plan_research_request(
                intent=intent,
                kb=kb,
                iteration=iteration,
                max_iterations=max_iterations,
                max_benchmarks=max_benchmarks,
            )
            result = {
                "iteration": iteration,
                "intent": intent,
                "planner": decision.planner,
                "reason": decision.reason,
                "current_question": decision.current_question,
                "research_request": decision.research_request,
            }
            research_request_raw_path = iter_dir / "research_request_raw.md"
            if decision.raw_response:
                write_text(research_request_raw_path, decision.raw_response)
                result["research_request_raw_artifact"] = str(research_request_raw_path)
            else:
                result["research_request_raw_artifact"] = None
            return result

        if task.kind == "llm_plan_benchmark":
            research_memo = _read_text_artifact(task.payload.get("research_artifact_md"))
            question_memo = str(task.payload.get("current_question", "")).strip()
            decision = self.workflow_backend.plan_benchmark(
                intent=intent,
                kb=kb,
                iteration=iteration,
                max_iterations=max_iterations,
                max_benchmarks=max_benchmarks,
                question_memo=question_memo,
                research_memo=research_memo,
            )
            plan = {
                "iteration": iteration,
                "intent": intent,
                "planner": decision.planner,
                "reason": decision.reason,
                "current_question": decision.current_question,
                "benchmark_plan": decision.benchmark_plan,
                "knowledge_model_artifact": task.payload.get("knowledge_model_artifact"),
            }
            benchmark_plan_raw_path = iter_dir / "benchmark_plan_raw.md"
            if decision.raw_response:
                write_text(benchmark_plan_raw_path, decision.raw_response)
            plan["benchmark_plan_raw_artifact"] = str(benchmark_plan_raw_path) if decision.raw_response else None
            return plan

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
            "current_question": decision.current_question,
            "knowledge_model": decision.knowledge_model,
            "benchmark_plan": decision.benchmark_plan,
            "research_request": decision.research_request,
        }
        model_path = iter_dir / "knowledge_model.json"
        write_json(model_path, decision.knowledge_model)
        plan["knowledge_model_artifact"] = str(model_path)
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
        research_request = task.payload.get("research_request", {})
        if not isinstance(research_request, dict):
            research_request = {}
        current_question = str(task.payload.get("current_question", "")).strip()
        if not research_request:
            research_request = _research_request_from_question(
                intent=str(task.payload.get("intent", "")).strip(),
                current_question=current_question,
            )
        intent = str(research_request.get("intent_summary", task.payload.get("intent", ""))).strip()
        max_sources = int(task.payload.get("max_sources", 8))
        request_memo = _render_research_request_md(
            {
                "iteration": iteration,
                "planner": task.payload.get("planner"),
                "reason": task.payload.get("reason"),
                "current_question": current_question,
                "research_request": research_request,
            }
        )
        decision = self.workflow_backend.research_context(
            intent=intent,
            kb=kb,
            iteration=iteration,
            research_request=research_request,
            research_request_memo=request_memo,
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
            "current_question": current_question,
        }
        iter_dir = _iteration_dir(ctx.run_dir, iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)
        json_path = iter_dir / "research.json"
        md_path = iter_dir / "research.md"
        raw_path = iter_dir / "research_raw.md"
        write_json(json_path, result)
        write_text(md_path, _render_research_md(result))
        if decision.raw_response:
            write_text(raw_path, decision.raw_response)
        result["artifact"] = str(json_path)
        result["artifact_md"] = str(md_path)
        result["raw_artifact"] = str(raw_path) if decision.raw_response else None
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
        working_planning_memo = _build_planning_context_memo(plan=plan, kb=kb)

        prompt_sections: list[str] = []
        focus = _benchmark_plan_focus_nodes(plan.get("benchmark_plan", {}))[: max(1, max_benchmarks)]
        if not focus:
            focus = _planning_focus_nodes(plan=plan, kb=kb)[: max(1, max_benchmarks)]
        if not focus:
            focus = [f"dimension_{iteration + 1}"]
        iter_dir = _iteration_dir(ctx.run_dir, iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)
        decision = None
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        policy: dict[str, Any] = {}
        preflight_results: list[dict[str, Any]] = []
        preflight_feedback: list[str] = []
        max_codegen_attempts = 2
        for codegen_attempt in range(max_codegen_attempts):
            prompt_sections.extend(
                _build_codegen_prompt_sections(
                    intent=intent,
                    kb=kb,
                    plan=plan,
                    iteration=iteration,
                    focus=focus,
                    planning_memo=working_planning_memo,
                    start_index=len(prompt_sections),
                )
            )
            decision = self.workflow_backend.generate_implementation(
                intent=intent,
                kb=kb,
                plan=plan,
                iteration=iteration,
                max_benchmarks=max_benchmarks,
                planning_memo=working_planning_memo,
            )
            accepted, rejected, policy = _apply_negotiation_policy(
                benchmarks=decision.benchmarks,
                schema_contract=kb.get("schema_contract", {}),
            )
            generated_files = _materialize_generated_files(iter_dir, accepted)
            accepted, preflight_rejected, preflight_results = _preflight_codegen_benchmarks(iter_dir, accepted)
            rejected.extend(preflight_rejected)
            if accepted or not preflight_rejected or codegen_attempt + 1 >= max_codegen_attempts:
                break
            failure_notes = _format_preflight_feedback(preflight_results)
            if not failure_notes:
                break
            preflight_feedback.append(failure_notes)
            working_planning_memo = _append_codegen_feedback(working_planning_memo, preflight_feedback)
        else:
            generated_files = []

        accepted = [_annotate_feasibility(item, policy) for item in accepted]
        feasibility_report = _build_feasibility_report(
            accepted=accepted,
            rejected=rejected,
            benchmark_plan=plan.get("benchmark_plan", {}),
        )

        implementation = {
            "iteration": iteration,
            "intent": intent,
            "planner": decision.planner if decision else "unknown-planner",
            "reason": decision.reason if decision else "Code generation failed before producing an implementation.",
            "benchmarks": accepted,
            "rejected_benchmarks": rejected,
            "contract_amendments": decision.contract_amendments if decision else [],
            "feasibility_summary": feasibility_report.get("summary", {}),
            "negotiation": {
                "policy": policy,
                "amendment_round": amendment_round,
                "amendment_feedback": amendment_feedback,
                "accepted_count": len(accepted),
                "rejected_count": len(rejected),
            },
        }
        implementation["generated_files"] = generated_files
        json_path = iter_dir / "implementation.json"
        md_path = iter_dir / "implementation.md"
        feasibility_path = iter_dir / "feasibility_report.json"
        preflight_path = iter_dir / "implementation_preflight.json"
        raw_path = iter_dir / "implementation_raw.md"
        prompt_path = iter_dir / "implementation_prompt.md"
        write_json(json_path, implementation)
        write_json(feasibility_path, feasibility_report)
        write_json(preflight_path, preflight_results)
        write_text(md_path, _render_implementation_md(implementation))
        write_text(prompt_path, "\n\n---\n\n".join(prompt_sections))
        if decision and decision.raw_response:
            write_text(raw_path, decision.raw_response)
        implementation["artifact"] = str(json_path)
        implementation["artifact_md"] = str(md_path)
        implementation["feasibility_artifact"] = str(feasibility_path)
        implementation["preflight_artifact"] = str(preflight_path)
        implementation["prompt_artifact"] = str(prompt_path)
        implementation["raw_artifact"] = str(raw_path) if decision and decision.raw_response else None
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
        cwd_path = str(task.payload.get("cwd_path", "")).strip()
        cwd = Path(cwd_path) if cwd_path else None

        start = time.time()
        executed_command = str(command)
        proc = _run_shell_command(executed_command, cwd=cwd)
        retried_with_sudo_ncu = False
        sudo_ncu_path = ""
        if _should_retry_with_sudo_ncu(executed_command, proc):
            sudo_ncu_path = _find_passwordless_sudo_ncu_path()
            retry_command = _rewrite_command_with_sudo_ncu(executed_command, sudo_ncu_path)
            if retry_command != executed_command:
                proc = _run_shell_command(retry_command, cwd=cwd)
                executed_command = retry_command
                retried_with_sudo_ncu = True
        elapsed = time.time() - start

        out = {
            "command": executed_command,
            "original_command": str(command),
            "cwd": str(cwd) if cwd else "",
            "returncode": proc.returncode,
            "elapsed_sec": elapsed,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "skipped": _is_skipped_workload(proc.stdout, proc.stderr),
            "retried_with_sudo_ncu": retried_with_sudo_ncu,
            "sudo_ncu_path": sudo_ncu_path,
        }
        artifact = ctx.run_dir / "workload_result.json"
        artifact_override = task.payload.get("artifact_path")
        if artifact_override:
            artifact = Path(artifact_override)
        write_json(artifact, out)
        out["artifact"] = str(artifact)
        return out


def _run_shell_command(command: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", "-lc", str(command)], capture_output=True, text=True, cwd=cwd)


def _should_retry_with_sudo_ncu(command: str, proc: subprocess.CompletedProcess[str]) -> bool:
    if proc.returncode == 0:
        return False
    if not _contains_ncu_invocation(command):
        return False
    combined_output = f"{proc.stdout}\n{proc.stderr}"
    if "ERR_NVGPUCTRPERM" not in combined_output:
        return False
    return "sudo -n " not in str(command)


def _contains_ncu_invocation(command: str) -> bool:
    return bool(re.search(r"(?<!\S)(?:ncu|/usr/local/cuda(?:-[^/\s]+)?/bin/ncu)(?=\s|$)", str(command)))


def _find_passwordless_sudo_ncu_path() -> str:
    for candidate in PASSWORDLESS_SUDO_NCU_CANDIDATES:
        probe = subprocess.run(["sudo", "-n", candidate, "--version"], capture_output=True, text=True)
        if probe.returncode == 0:
            return candidate
    return ""


def _rewrite_command_with_sudo_ncu(command: str, sudo_ncu_path: str) -> str:
    if not sudo_ncu_path:
        return str(command)
    pattern = r"(?<!\S)(?:ncu|/usr/local/cuda(?:-[^/\s]+)?/bin/ncu)(?=\s|$)"
    return re.sub(pattern, f"sudo -n {shlex.quote(sudo_ncu_path)}", str(command))


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
        command_cwd = _iteration_dir(ctx.run_dir, iteration) / benchmark_id

        workload_task = Task(
            id=f"bench-workload-{iteration}-{index}",
            kind="run_workload",
            payload={
                "command": command,
                "artifact_path": str(bench_dir / "workload_result.json"),
                "cwd_path": str(command_cwd),
            },
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
        kb_path = Path(task.payload.get("kb_path") or (ctx.run_dir / "run_state.json"))
        kb = read_json(kb_path, {})
        knowledge_model_path = Path(task.payload.get("knowledge_model_path") or (ctx.run_dir / "knowledge_model.json"))
        current_model = read_json(knowledge_model_path, {})
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
        kb.setdefault("knowledge_model_history", [])
        kb["covered_dimensions"] = covered
        kb["coverage_score"] = coverage_score
        kb["last_summary"] = decision.summary
        kb["last_planner"] = decision.planner
        kb["claims"].extend(decision.claims)
        updated_model = _update_knowledge_model(
            current_model=current_model,
            intent=intent,
            plan=plan,
            covered_dimensions=covered,
            claims=decision.claims,
            required_observability=decision.required_observability,
            iteration=iteration,
        )
        kb["current_knowledge_model"] = updated_model
        kb["knowledge_model_artifact"] = str(knowledge_model_path)
        kb["knowledge_model_history"].append(
            {
                "iteration": iteration,
                "artifact": str(knowledge_model_path),
                "focus_nodes": updated_model.get("focus_nodes", []),
                "node_count": len(updated_model.get("domain_hierarchy", [])),
                "timestamp": time.time(),
            }
        )
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
        write_json(knowledge_model_path, updated_model)
        kb_files = update_markdown_knowledge_base(
            ctx.run_dir,
            intent=intent,
            kb=kb,
            knowledge_model=updated_model,
            iteration=iteration,
            analysis={
                "summary": decision.summary,
                "claims": decision.claims,
                "covered_dimensions": covered,
                "required_observability": decision.required_observability,
            },
        )
        kb.update(kb_files)
        write_json(kb_path, kb)

        iter_dir = _iteration_dir(ctx.run_dir, iteration)
        analysis_path = iter_dir / "analysis_update.json"
        analysis_md_path = iter_dir / "analysis.md"
        iter_model_path = iter_dir / "knowledge_model.json"
        write_json(iter_model_path, updated_model)
        out = {
            "iteration": iteration,
            "summary": decision.summary,
            "claims": decision.claims,
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
            "knowledge_model_artifact": str(knowledge_model_path),
            "knowledge_model_iteration_artifact": str(iter_model_path),
            "knowledge_base_index_artifact": kb.get("knowledge_base_index_artifact"),
            "knowledge_base_frontier_artifact": kb.get("knowledge_base_frontier_artifact"),
        }
        write_json(analysis_path, out)
        write_text(analysis_md_path, _render_analysis_md(out))
        out["artifact"] = str(analysis_path)
        out["artifact_md"] = str(analysis_md_path)
        return out


class CommunicationMonitorAgent(Agent):
    name = "communication-monitor"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "monitor_communications"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        event = task.payload.get("event", {})
        if not isinstance(event, dict):
            raise ValueError("monitor_communications requires event payload")

        global_json_path = ctx.run_dir / "agent_conversation.json"
        global_md_path = ctx.run_dir / "agent_conversation.md"
        events = read_json(global_json_path, [])
        if not isinstance(events, list):
            events = []
        events.append(event)
        write_json(global_json_path, events)
        write_text(global_md_path, _render_agent_conversation(events, ctx.run_id))
        latest_turn = _render_agent_conversation_turn(event, len(events))
        latest_screen_line = _render_agent_conversation_screen_line(event, len(events))

        result: dict[str, Any] = {
            "artifact": str(global_json_path),
            "artifact_md": str(global_md_path),
            "events_recorded": len(events),
            "screen_output": latest_screen_line,
        }

        iteration = event.get("iteration")
        if iteration is not None:
            try:
                iter_dir = _iteration_dir(ctx.run_dir, int(iteration))
                iter_dir.mkdir(parents=True, exist_ok=True)
                iter_json_path = iter_dir / "conversation.json"
                iter_md_path = iter_dir / "conversation.md"
                iter_events = read_json(iter_json_path, [])
                if not isinstance(iter_events, list):
                    iter_events = []
                iter_events.append(event)
                write_json(iter_json_path, iter_events)
                write_text(iter_md_path, _render_agent_conversation(iter_events, f"{ctx.run_id} iter {int(iteration):02d}"))
                result["iteration_artifact"] = str(iter_json_path)
                result["iteration_artifact_md"] = str(iter_md_path)
            except Exception:
                pass
        return result


class AutonomousReporterAgent(Agent):
    name = "autonomous-reporter"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "autonomous_report"

    def run(self, _task: Task, ctx: AgentContext) -> dict[str, Any]:
        model = read_json(ctx.run_dir / "run_state.json", {})
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
        lines.extend(
            [
                "",
                "## Claims",
                f"- total: `{len(model.get('claims', []))}`",
                "",
                "## Knowledge Base",
                f"- index: `{model.get('knowledge_base_index_artifact', '')}`",
                f"- frontier: `{model.get('knowledge_base_frontier_artifact', '')}`",
            ]
        )
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
        CommunicationMonitorAgent(),
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


def _render_agent_conversation(events: list[dict[str, Any]], title: str) -> str:
    lines = [f"# Agent Conversation ({title})", ""]
    for index, event in enumerate(events, start=1):
        lines.extend(_render_agent_conversation_turn(event, index).splitlines())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_agent_conversation_turn(event: dict[str, Any], turn_index: int) -> str:
    sender = str(event.get("sender", "")).strip() or "unknown"
    recipient = str(event.get("recipient", "")).strip() or "unknown"
    lines = [f"## Turn {turn_index}"]
    summary = str(event.get("summary", "")).strip()
    if summary:
        lines.append(f"{sender} to {recipient}:")
        lines.append(summary)
        lines.append("")
    body = str(event.get("message", "")).strip()
    if body:
        lines.append(body)
    return "\n".join(lines).rstrip()


def _render_agent_conversation_screen_line(event: dict[str, Any], turn_index: int) -> str:
    sender = str(event.get("sender", "")).strip() or "unknown"
    recipient = str(event.get("recipient", "")).strip() or "unknown"
    summary = " ".join(str(event.get("summary", "")).strip().split())
    return f"Round #{turn_index}, {sender} -> {recipient}: message: {summary}"


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
    accepted: list[dict[str, Any]], rejected: list[dict[str, Any]], benchmark_plan: dict[str, Any]
) -> dict[str, Any]:
    _ = benchmark_plan
    items: list[dict[str, Any]] = []
    for item in accepted:
        benchmark_id = str(item.get("id", "")).strip()
        complexity = str(item.get("implementation_complexity", "medium")).strip() or "medium"
        feasibility_status = str(item.get("feasibility_status", "feasible")).strip() or "feasible"
        recommended_changes: list[str] = []
        if feasibility_status == "feasible_with_revision":
            recommended_changes.append("Reduce implementation complexity or split this benchmark plan into smaller steps.")
        items.append(
            {
                "benchmark_plan_id": benchmark_id,
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
        recommend = ["Revise the benchmark plan to improve implementability and reduce implementation risk."]
        if "implementability_below_min" in reasons:
            recommend.append("Simplify the implementation scope or reduce dependency/tooling assumptions.")
        if "utility_below_min" in reasons or "coverage_gain_below_min" in reasons:
            recommend.append("Clarify why this benchmark is needed at the current curriculum stage.")
        items.append(
            {
                "benchmark_plan_id": str(item.get("id", "")).strip(),
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
        "items": items,
        "summary": summary,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _render_benchmark_plan_md(plan: dict[str, Any]) -> str:
    benchmark_plan = plan.get("benchmark_plan", {}) if isinstance(plan.get("benchmark_plan", {}), dict) else {}
    lines = [
        f"# Iteration {plan.get('iteration')} Benchmark Plan",
        "",
        "## Metadata",
        f"- planner: `{plan.get('planner')}`",
        f"- reason: {plan.get('reason')}",
        "",
        "## Benchmark Plan Summary",
        benchmark_plan.get("plan_summary", ""),
        "",
        "## Current Question",
        str(plan.get("current_question", "")).strip() or "No current question recorded.",
        "",
        "## Target Nodes",
    ]
    target_nodes = [str(x).strip() for x in benchmark_plan.get("target_nodes", []) if str(x).strip()]
    if target_nodes:
        for item in target_nodes:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.extend(["", "## Planned Benchmarks"])
    for index, item in enumerate(benchmark_plan.get("benchmarks", []), start=1):
        lines.extend(
            [
                f"### Benchmark {index}",
                f"- id: `{item.get('id', '')}`",
                f"- title: {item.get('title', '')}",
                f"- benchmark role: {item.get('benchmark_role', '')}",
                f"- objective: {item.get('objective', '')}",
                f"- hypothesis: {item.get('hypothesis', '')}",
                f"- rationale: {item.get('rationale', '')}",
                "#### Target Nodes",
            ]
        )
        target_node_ids = [str(x).strip() for x in item.get("target_node_ids", []) if str(x).strip()]
        if target_node_ids:
            for node in target_node_ids:
                lines.append(f"- {node}")
        else:
            lines.append("- none")
        lines.append("#### Required Evidence")
        required_evidence = [str(x).strip() for x in item.get("required_evidence", []) if str(x).strip()]
        if required_evidence:
            for evidence in required_evidence:
                lines.append(f"- {evidence}")
        else:
            lines.append("- none specified")
        lines.extend(
            [
                "#### What Success Unlocks",
                str(item.get("success_unlocks", "")).strip() or str(item.get("rationale", "")).strip() or "Not specified.",
                "",
            ]
        )
    return "\n".join(lines)


def _render_research_request_md(result: dict[str, Any]) -> str:
    request = result.get("research_request", {}) if isinstance(result.get("research_request", {}), dict) else {}
    lines = [
        f"# Iteration {result.get('iteration')} Research Request",
        "",
        "## Metadata",
        f"- planner: `{result.get('planner')}`",
        f"- reason: {result.get('reason')}",
        "",
        "## Current Question",
        str(result.get("current_question", "")).strip() or "No current question recorded.",
        "",
        "## Objective",
        request.get("request_summary", ""),
        "",
        "## Target Nodes",
    ]
    for item in request.get("target_nodes", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Questions"])
    for item in request.get("target_questions", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Search Topics"])
    for item in request.get("search_topics", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Expected Outputs"])
    for item in request.get("expected_outputs", []):
        lines.append(f"- {item}")
    return "\n".join(lines)


def _empty_knowledge_model(intent: str) -> dict[str, Any]:
    return {
        "intent": {"summary": intent},
        "domain_hierarchy": [],
        "focus_nodes": [],
        "generated_at": "",
        "planner_notes": "Initialized local knowledge model.",
    }


def _update_knowledge_model(
    current_model: dict[str, Any],
    intent: str,
    plan: dict[str, Any],
    covered_dimensions: list[str],
    claims: list[dict[str, Any]],
    required_observability: list[str],
    iteration: int,
) -> dict[str, Any]:
    model = current_model if isinstance(current_model, dict) and isinstance(current_model.get("domain_hierarchy", []), list) else _empty_knowledge_model(intent)
    hierarchy = [dict(item) for item in model.get("domain_hierarchy", []) if isinstance(item, dict)]
    nodes_by_id = {str(item.get("id", "")).strip(): item for item in hierarchy if str(item.get("id", "")).strip()}
    benchmark_plan = (
        plan.get("benchmark_plan", {}) if isinstance(plan.get("benchmark_plan", {}), dict) else {}
    )
    benchmark_items = (
        benchmark_plan.get("benchmarks", []) if isinstance(benchmark_plan.get("benchmarks", []), list) else []
    )
    covered = {str(x).strip() for x in covered_dimensions if str(x).strip()}

    for item in benchmark_items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        objective = str(item.get("objective", "")).strip()
        hypothesis = str(item.get("hypothesis", "")).strip()
        target_ids = [str(x).strip() for x in item.get("target_node_ids", []) if str(x).strip()]
        if not target_ids:
            target_ids = [str(x).strip() for x in benchmark_plan.get("target_nodes", []) if str(x).strip()]
        for node_id in target_ids:
            if not node_id:
                continue
            node = nodes_by_id.get(node_id)
            if node is None:
                node = {
                    "id": node_id,
                    "name": node_id,
                    "description": objective or title or node_id,
                    "parent_id": None,
                    "node_type": "feature",
                    "status": "unknown",
                    "rationale": hypothesis or "Added from planner benchmark plan.",
                    "evidence_refs": [],
                    "open_gaps": [],
                }
                hierarchy.append(node)
                nodes_by_id[node_id] = node
            if node_id in covered:
                node["status"] = "partially_supported"
            if required_observability:
                gaps = [str(x).strip() for x in node.get("open_gaps", []) if str(x).strip()]
                for gap in required_observability:
                    item_gap = str(gap).strip()
                    if item_gap and item_gap not in gaps:
                        gaps.append(item_gap)
                node["open_gaps"] = gaps[:6]

    for idx, claim in enumerate(claims):
        if not isinstance(claim, dict):
            continue
        for dim in [str(x).strip() for x in claim.get("dimensions", []) if str(x).strip()]:
            node = nodes_by_id.get(dim)
            if node is None:
                node = {
                    "id": dim,
                    "name": dim,
                    "description": f"Observed dimension derived from analysis claim for {dim}.",
                    "parent_id": None,
                    "node_type": "feature",
                    "status": "partially_supported",
                    "rationale": "Added from analyzer claim.",
                    "evidence_refs": [],
                    "open_gaps": [],
                }
                hierarchy.append(node)
                nodes_by_id[dim] = node
            node["status"] = "partially_supported"
            refs = [str(x).strip() for x in node.get("evidence_refs", []) if str(x).strip()]
            ref = f"iter_{iteration:02d}_claim_{idx}"
            if ref not in refs:
                refs.append(ref)
            node["evidence_refs"] = refs[:10]

    focus_nodes = [str(x).strip() for x in benchmark_plan.get("target_nodes", []) if str(x).strip()]
    if not focus_nodes:
        focus_nodes = [item["id"] for item in hierarchy[: min(3, len(hierarchy))]]
    return {
        "intent": {"summary": intent},
        "domain_hierarchy": hierarchy,
        "focus_nodes": focus_nodes,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "planner_notes": str(model.get("planner_notes", "Updated incrementally from analyzer outputs.")).strip()
        or "Updated incrementally from analyzer outputs.",
    }


def _render_research_md(research: dict[str, Any]) -> str:
    lines = [
        f"# Iteration {research.get('iteration')} Research",
        "",
        "## Metadata",
        f"- planner: `{research.get('planner')}`",
        f"- reason: {research.get('reason')}",
        "",
        "## Current Question",
        str(research.get("current_question", "")).strip() or "No current question recorded.",
        "",
        "## Request Summary",
        str(research.get("request_summary", "")).strip() or "No request summary recorded.",
        "",
        "## Proposed Dimensions",
    ]
    proposed_dimensions = [str(x).strip() for x in research.get("proposed_dimensions", []) if str(x).strip()]
    if proposed_dimensions:
        for item in proposed_dimensions:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.extend(["", "## Unanswered Questions"])
    unanswered_questions = [str(x).strip() for x in research.get("unanswered_questions", []) if str(x).strip()]
    if unanswered_questions:
        for item in unanswered_questions:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.extend(["", "## Findings"])
    for index, item in enumerate(research.get("findings", []), start=1):
        lines.extend(
            [
                f"### Finding {index}",
                f"- title: {item.get('title')}",
                f"- relevance: {item.get('relevance')}",
                f"- source: {item.get('source_url')}",
                f"- summary: {item.get('summary')}",
                "",
            ]
        )
    if not research.get("findings", []):
        lines.append("No findings recorded.")
    return "\n".join(lines)


def _research_request_from_question(intent: str, current_question: str) -> dict[str, Any]:
    question = str(current_question).strip()
    if not question:
        question = "What evidence is needed to advance the current GPU knowledge frontier?"
    target = _slugify_dimension(question) or "gpu_frontier"
    return {
        "intent_summary": intent,
        "request_summary": f"Gather external context that helps answer: {question}",
        "target_nodes": [target],
        "target_questions": [question],
        "search_topics": [question],
        "source_preferences": ["vendor_doc", "official_tool_doc", "paper", "article"],
        "expected_outputs": ["mechanism summary", "measurement guidance", "open questions"],
    }


def _planning_focus_nodes(plan: dict[str, Any], kb: dict[str, Any]) -> list[str]:
    out: list[str] = []
    question = str(plan.get("current_question", "")).strip()
    slug = _slugify_dimension(question)
    if slug:
        out.append(slug)
    latest_research = kb.get("latest_research", {}) if isinstance(kb.get("latest_research", {}), dict) else {}
    for item in latest_research.get("proposed_dimensions", []):
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    for item in kb.get("covered_dimensions", []) if isinstance(kb.get("covered_dimensions", []), list) else []:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out[:6]


def _build_planning_context_memo(plan: dict[str, Any], kb: dict[str, Any]) -> str:
    question = str(plan.get("current_question", "")).strip() or "No current question recorded."
    reason = str(plan.get("reason", "")).strip() or "No explicit planner rationale recorded."
    latest_research = kb.get("latest_research", {}) if isinstance(kb.get("latest_research", {}), dict) else {}
    research_memo = _read_text_artifact(latest_research.get("artifact_md"))
    lines = [
        "# Planning Context",
        "",
        "## Current Question",
        question,
        "",
        "## Planner Rationale",
        reason,
    ]
    if research_memo.strip():
        lines.extend(["", "## Latest Research", research_memo.strip()])
    return "\n".join(lines)


def _read_text_artifact(path_value: Any) -> str:
    path = str(path_value or "").strip()
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


def _render_implementation_md(implementation: dict[str, Any]) -> str:
    lines = [
        f"# Iteration {implementation.get('iteration')} Implementation",
        "",
        "## Metadata",
        f"- planner: `{implementation.get('planner')}`",
        f"- reason: {implementation.get('reason')}",
        "",
        "## Benchmarks",
    ]
    for index, bench in enumerate(implementation.get("benchmarks", []), start=1):
        lines.extend(
            [
                f"### Benchmark {index}",
                f"- id: `{bench.get('id')}`",
                f"- dimensions: `{bench.get('dimensions', [])}`",
                f"- hypothesis: {bench.get('hypothesis')}",
                f"- feasibility status: `{bench.get('feasibility_status')}`",
                f"- implementation complexity: `{bench.get('implementation_complexity')}`",
                f"- scores: `{bench.get('scores', {})}`",
                f"- utility score: `{bench.get('utility_score')}`",
                f"- command: `{bench.get('command')}`",
                "",
            ]
        )
    if not implementation.get("benchmarks"):
        lines.append("No accepted benchmarks were generated.")
    if implementation.get("feasibility_summary"):
        lines.extend(["", "## Feasibility Summary", f"- {implementation.get('feasibility_summary')}"])
    if implementation.get("rejected_benchmarks"):
        lines.extend(["", "## Rejected Benchmarks"])
        for index, item in enumerate(implementation.get("rejected_benchmarks", []), start=1):
            lines.extend(
                [
                    f"### Rejected Benchmark {index}",
                    f"- id: `{item.get('id')}`",
                    f"- reasons: `{item.get('reasons', [])}`",
                    f"- scores: `{item.get('scores', {})}`",
                    "",
                ]
            )
    if implementation.get("negotiation"):
        lines.extend(["", "## Negotiation", f"- {implementation.get('negotiation')}"])
    if implementation.get("contract_amendments"):
        lines.extend(["", "## Contract Amendments"])
        for item in implementation.get("contract_amendments", []):
            lines.append(
                f"- path: `{item.get('path')}` | change: {item.get('change')} | priority: `{item.get('priority')}`"
            )
    if implementation.get("generated_files"):
        lines.extend(["", "## Generated Files"])
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
        "## Metadata",
        f"- planner: `{analysis.get('planner')}`",
        f"- summary: {analysis.get('summary')}",
        f"- claims added: `{analysis.get('claims_added', 0)}`",
        f"- coverage score: `{analysis.get('coverage_score', 0.0)}`",
        f"- stop: `{analysis.get('stop', False)}`",
        f"- reason: {analysis.get('reason')}",
    ]
    veto_reason = str(analysis.get("veto_reason", "")).strip()
    if veto_reason:
        lines.append(f"- veto_reason: {veto_reason}")
    lines.extend(["", "## Covered Dimensions"])
    covered_dimensions = [str(x).strip() for x in analysis.get("covered_dimensions", []) if str(x).strip()]
    if covered_dimensions:
        for item in covered_dimensions:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.extend(["", "## Claims"])
    claims = analysis.get("claims", [])
    if claims:
        for index, item in enumerate(claims, start=1):
            evidence = item.get("evidence", {}) if isinstance(item.get("evidence", {}), dict) else {}
            evidence_refs = [str(value).strip() for value in evidence.values() if str(value).strip()]
            lines.extend(
                [
                    f"### Claim {index}",
                    f"- claim: {item.get('claim', '')}",
                    f"- claim type: {item.get('claim_type', item.get('type', 'inference'))}",
                    f"- confidence: `{item.get('confidence', '')}`",
                    f"- status: `{item.get('status', 'active')}`",
                    f"- method summary: {item.get('method_summary', 'Derived from the recorded execution evidence and analysis memo.')}",
                    "#### Dimensions",
                ]
            )
            dimensions = [str(x).strip() for x in item.get("dimensions", []) if str(x).strip()]
            if dimensions:
                for dim in dimensions:
                    lines.append(f"- {dim}")
            else:
                lines.append("- none")
            lines.append("#### Evidence")
            if evidence_refs:
                for ref in evidence_refs:
                    lines.append(f"- {ref}")
            else:
                lines.append("- none recorded")
            lines.append("")
    else:
        lines.append("No claims recorded.")
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


def _build_codegen_prompt_sections(
    intent: str,
    kb: dict[str, Any],
    plan: dict[str, Any],
    iteration: int,
    focus: list[str],
    planning_memo: str,
    start_index: int = 0,
) -> list[str]:
    sections: list[str] = []
    for dim in focus:
        prompt_payload = {
            "intent": intent,
            "knowledge_base": _compact_codegen_kb(kb=kb, dimension=dim),
            "plan": _compact_codegen_plan(plan=plan, dimension=dim),
            "iteration": iteration,
            "dimension": dim,
            "planning_memo": _trim_text(planning_memo, 3000),
            "constraints": {
                "bounded_runtime": "Each command should complete in <= 45 seconds when possible.",
                "safety": "No destructive commands, no system configuration mutation.",
                "no_inventory_only": "Do not return inventory/topology-only probes as benchmarks.",
            },
        }
        prompt_payload = _enforce_payload_budget(
            prompt_payload,
            target_chars=CODEGEN_INPUT_TARGET_CHARS,
            hard_cap_chars=CODEGEN_INPUT_HARD_CAP_CHARS,
            trimmers=[_trim_codegen_payload],
        )
        sections.append(
            "\n".join(
                [
                    f"# Codegen Prompt {start_index + len(sections) + 1}",
                    "",
                    f"## Target Dimension\n{dim}",
                    "",
                    "## System Prompt",
                    CODEGEN_SYSTEM_PROMPT,
                    "",
                    "## User Prompt",
                    _render_codegen_prompt(prompt_payload),
                ]
            )
        )
    return sections


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


def _preflight_codegen_benchmarks(
    iter_dir: Path, benchmarks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for bench in benchmarks:
        report = _preflight_single_benchmark(iter_dir, bench)
        reports.append(report)
        bench["preflight"] = report
        if report.get("ok"):
            accepted.append(bench)
            continue
        rejected.append(
            {
                "id": bench.get("id"),
                "command": bench.get("command"),
                "dimensions": bench.get("dimensions", []),
                "hypothesis": bench.get("hypothesis"),
                "rejection_reason": report.get("reason", "preflight failed"),
                "preflight": report,
            }
        )
    return accepted, rejected, reports


def _preflight_single_benchmark(iter_dir: Path, benchmark: dict[str, Any]) -> dict[str, Any]:
    bench_id = str(benchmark.get("id", "benchmark")).strip() or "benchmark"
    command = str(benchmark.get("command", "")).strip()
    cwd = iter_dir / bench_id
    report: dict[str, Any] = {
        "benchmark_id": bench_id,
        "cwd": str(cwd),
        "command": command,
        "checked": False,
        "ok": True,
        "reason": "",
        "missing_files": [],
        "preflight_command": "",
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }
    missing_files = []
    for file_spec in benchmark.get("files", []):
        rel_path = str((file_spec or {}).get("path", "")).strip()
        if not rel_path:
            continue
        target = cwd / rel_path
        if not target.exists():
            missing_files.append(rel_path)
    if missing_files:
        report["ok"] = False
        report["reason"] = "generated source files were not materialized at the expected paths"
        report["missing_files"] = missing_files
        return report

    preflight_command, expected_output = _extract_compile_preflight(command)
    if not preflight_command:
        report["reason"] = "no compile preflight required"
        return report

    report["checked"] = True
    report["preflight_command"] = preflight_command
    try:
        proc = subprocess.run(
            ["bash", "-lc", preflight_command],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=60,
        )
        report["returncode"] = proc.returncode
        report["stdout"] = proc.stdout
        report["stderr"] = proc.stderr
        if proc.returncode != 0:
            report["ok"] = False
            report["reason"] = "compile preflight failed"
            return report
    except Exception as exc:  # noqa: BLE001
        report["ok"] = False
        report["reason"] = f"compile preflight errored: {exc}"
        report["stderr"] = str(exc)
        return report

    if expected_output:
        output_path = cwd / expected_output
        report["expected_output"] = str(output_path)
        if not output_path.exists():
            report["ok"] = False
            report["reason"] = "compile preflight succeeded but expected binary was not created"
            return report
    report["reason"] = "compile preflight passed"
    return report


def _extract_compile_preflight(command: str) -> tuple[str, str]:
    segments = [segment.strip() for segment in str(command or "").split("&&") if segment.strip()]
    if not segments:
        return "", ""
    for index, segment in enumerate(segments):
        if _looks_like_compile_segment(segment):
            return " && ".join(segments[: index + 1]), _extract_compile_output_path(segment)
    return "", ""


def _looks_like_compile_segment(segment: str) -> bool:
    lowered = str(segment or "").strip().lower()
    return lowered.startswith("nvcc ") or lowered == "nvcc" or " nvcc " in lowered


def _extract_compile_output_path(segment: str) -> str:
    try:
        tokens = shlex.split(segment)
    except Exception:
        tokens = str(segment or "").split()
    for index, token in enumerate(tokens):
        if token == "-o" and index + 1 < len(tokens):
            return tokens[index + 1]
    return ""


def _format_preflight_feedback(reports: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for report in reports:
        if report.get("ok"):
            continue
        bench_id = str(report.get("benchmark_id", "")).strip() or "benchmark"
        reason = str(report.get("reason", "")).strip() or "preflight failed"
        stderr = _trim_text(report.get("stderr", ""), 1200)
        missing = report.get("missing_files", [])
        lines.append(f"- {bench_id}: {reason}")
        if missing:
            lines.append(f"  missing_files: {missing}")
        if stderr:
            lines.append(f"  compiler_stderr: {stderr}")
    return "\n".join(lines).strip()


def _append_codegen_feedback(planning_memo: str, feedback: list[str]) -> str:
    clean_feedback = [str(item).strip() for item in feedback if str(item).strip()]
    if not clean_feedback:
        return planning_memo
    return (
        f"{planning_memo.rstrip()}\n\n"
        "Preflight repair feedback:\n"
        "Your previous implementation did not compile locally. Return a corrected full implementation.\n"
        + "\n\n".join(clean_feedback)
    ).strip()


def _infer_target_dimensions(
    plan: dict[str, Any], execution_results: list[dict[str, Any]], claims: list[dict[str, Any]]
) -> list[str]:
    dims: list[str] = []
    benchmark_plan = (
        plan.get("benchmark_plan", {}) if isinstance(plan.get("benchmark_plan", {}), dict) else {}
    )
    for d in benchmark_plan.get("target_nodes", []):
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
