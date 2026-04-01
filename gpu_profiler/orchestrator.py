import json
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .agents import Agent, default_agents
from .knowledge_base import initialize_markdown_knowledge_base
from .llm import HeuristicWorkflowBackend, OpenAIWorkflowBackend, ResilientWorkflowBackend
from .markdown_artifacts import (
    parse_analysis_markdown,
    parse_proposal_markdown,
    parse_research_markdown,
    parse_research_request_markdown,
)
from .models import AgentContext, RetryPolicy, Task
from .store import write_json

DEFAULT_TARGET_DIMENSIONS: list[str] = []


class Orchestrator:
    def __init__(
        self,
        agents: list[Agent],
        retry_policy: RetryPolicy | None = None,
        emit_live_trace: bool = True,
        trace_stream: Any | None = None,
        emit_live_conversation: bool = True,
        conversation_stream: Any | None = None,
    ):
        self.agents = agents
        self.retry_policy = retry_policy or RetryPolicy()
        self.emit_live_trace = emit_live_trace
        self.trace_stream = trace_stream or sys.stderr
        self._trace_stream_is_default = trace_stream is None
        self._trace_log_handle: Any | None = None
        self.emit_live_conversation = emit_live_conversation
        self.conversation_stream = conversation_stream or self._default_conversation_stream()
        self.monitor_agent = self._find_monitor_agent()

    def run_profile(
        self,
        workload: str,
        out_dir: str = "profiling_runs",
        samples: int = 3,
        interval_sec: float = 0.5,
    ) -> dict[str, Any]:
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        run_dir = Path(out_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ctx = AgentContext(run_id=run_id, run_dir=run_dir)
        self._attach_run_debug_log(run_dir)
        try:
            plan = Task(
                id="plan-0",
                kind="plan",
                payload={"workload": workload, "samples": samples, "interval_sec": interval_sec},
            )
            self._run_with_retry(plan, ctx)

            completed: list[Task] = [plan]
            stages = plan.result.get("stages", []) if plan.result else []

            for stage_idx, stage in enumerate(stages):
                stage_tasks = [
                    Task(id=f"stage{stage_idx}-{item['kind']}-{i}", kind=item["kind"], payload=item.get("payload", {}))
                    for i, item in enumerate(stage)
                ]
                stage_results = self._run_stage_parallel(stage_tasks, ctx)
                completed.extend(stage_results)
                self._persist_run_log(run_dir, completed)

            self._persist_run_log(run_dir, completed)
            return {
                "mode": "fixed",
                "run_id": run_id,
                "run_dir": str(run_dir),
                "tasks": [self._task_to_dict(t) for t in completed],
            }
        finally:
            self._close_run_debug_log()

    def run_autonomous_profile(
        self,
        intent: str,
        out_dir: str = "profiling_runs",
        samples: int = 3,
        interval_sec: float = 0.5,
        max_iterations: int = 4,
        max_benchmarks: int = 2,
        target_coverage: float = 0.9,
        target_dimensions: list[str] | None = None,
    ) -> dict[str, Any]:
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        run_dir = Path(out_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ctx = AgentContext(run_id=run_id, run_dir=run_dir)
        self._attach_run_debug_log(run_dir)
        try:
            dimensions = target_dimensions or DEFAULT_TARGET_DIMENSIONS
            knowledge_model_path = run_dir / "knowledge_model.json"
            initial_knowledge_model = {
                "intent": {"summary": intent},
                "domain_hierarchy": [],
                "focus_nodes": [],
                "generated_at": "",
                "planner_notes": "Initialized local knowledge model.",
            }
            write_json(knowledge_model_path, initial_knowledge_model)
            kb_files = initialize_markdown_knowledge_base(run_dir, intent)
            knowledge_base: dict[str, Any] = {
                "intent": intent,
                "target_dimensions": dimensions,
                "covered_dimensions": [],
                "coverage_score": 0.0,
                "target_coverage": target_coverage,
                "available_tools": self._detect_available_tools(),
                "history": [],
                "run_state": {
                    "status": "running",
                    "reason": "autonomous_run_started",
                    "iterations_completed": 0,
                    "planner_calls": 0,
                    "search_calls": 0,
                    "codegen_calls": 0,
                    "runner_calls": 0,
                    "analyzer_calls": 0,
                },
                "claims": [],
                "research_history": [],
                "current_knowledge_model": initial_knowledge_model,
                "knowledge_model_artifact": str(knowledge_model_path),
                **kb_files,
            }
            kb_path = run_dir / "run_state.json"
            write_json(kb_path, knowledge_base)

            completed: list[Task] = []
            sys_task = Task(id="boot-collect_system_info-0", kind="collect_system_info", payload={})
            completed.append(self._run_with_retry(sys_task, ctx))
            self._persist_run_log(run_dir, completed)

            for iteration in range(max_iterations):
                knowledge_base = self._load_kb(kb_path)
                research_plan_task = Task(
                    id=f"iter{iteration}-llm-plan-research-0",
                    kind="llm_plan_research",
                    payload={
                        "intent": intent,
                        "iteration": iteration,
                        "max_iterations": max_iterations,
                        "max_benchmarks": max_benchmarks,
                        "knowledge_base": knowledge_base,
                        "knowledge_model_artifact": knowledge_base.get("knowledge_model_artifact"),
                    },
                )
                completed.append(self._run_with_retry(research_plan_task, ctx))
                self._persist_run_log(run_dir, completed)
                if research_plan_task.status != "done":
                    self._update_run_state(kb_path, status="stopped", reason="planner_failed", iteration=iteration)
                    break
                self._increment_run_counter(kb_path, "planner_calls")

                research_plan = self._canonicalize_markdown_result("llm_plan_research", research_plan_task.result or {})
                research_plan_task.result = research_plan
                self._append_planner_research_outputs(kb_path=kb_path, iteration=iteration, result=research_plan)
                knowledge_base = self._load_kb(kb_path)
                research_request_artifact = research_plan.get("research_request_artifact")
                if research_request_artifact:
                    research_task = Task(
                        id=f"iter{iteration}-llm-research-0",
                        kind="llm_research",
                        payload={
                            "intent": intent,
                            "iteration": iteration,
                            "knowledge_base": knowledge_base,
                            "research_request_artifact": research_request_artifact,
                            "research_request_meta_artifact": research_plan.get("research_request_meta_artifact"),
                            "max_sources": 10,
                        },
                    )
                    completed.append(self._run_with_retry(research_task, ctx))
                    self._persist_run_log(run_dir, completed)
                    if research_task.status != "done":
                        self._update_run_state(kb_path, status="stopped", reason="search_failed", iteration=iteration)
                        break
                    self._increment_run_counter(kb_path, "search_calls")
                    research = self._canonicalize_markdown_result("llm_research", research_task.result or {})
                    research_task.result = research
                    self._append_research_history(kb_path=kb_path, iteration=iteration, research=research)
                    knowledge_base = self._load_kb(kb_path)

                proposal_task = Task(
                    id=f"iter{iteration}-llm-plan-proposal-0",
                    kind="llm_plan_proposal",
                    payload={
                        "intent": intent,
                        "iteration": iteration,
                        "max_iterations": max_iterations,
                        "max_benchmarks": max_benchmarks,
                        "knowledge_base": knowledge_base,
                        "knowledge_model_artifact": knowledge_base.get("knowledge_model_artifact"),
                        "research_artifact_md": (
                            knowledge_base.get("latest_research", {})
                            if isinstance(knowledge_base.get("latest_research", {}), dict)
                            else {}
                        ).get("artifact_md"),
                    },
                )
                completed.append(self._run_with_retry(proposal_task, ctx))
                self._persist_run_log(run_dir, completed)
                if proposal_task.status != "done":
                    self._update_run_state(kb_path, status="stopped", reason="planner_proposal_failed", iteration=iteration)
                    break
                self._increment_run_counter(kb_path, "planner_calls")
                plan = self._canonicalize_markdown_result("llm_plan_proposal", proposal_task.result or {})
                proposal_task.result = plan
                self._append_planner_outputs(kb_path=kb_path, iteration=iteration, plan=plan)
                knowledge_base = self._load_kb(kb_path)

                implementation: dict[str, Any] = {}
                implementation_task: Task | None = None
                amendment_feedback: list[dict[str, Any]] = []
                for amend_round in range(1):
                    implementation_task = Task(
                        id=f"iter{iteration}-llm-implementation-{amend_round}",
                        kind="llm_generate_implementation",
                        payload={
                            "intent": intent,
                            "iteration": iteration,
                            "max_benchmarks": max_benchmarks,
                            "knowledge_base": knowledge_base,
                            "plan": plan,
                            "amendment_round": amend_round,
                            "amendment_feedback": amendment_feedback,
                        },
                    )
                    completed.append(self._run_with_retry(implementation_task, ctx))
                    self._persist_run_log(run_dir, completed)
                    if implementation_task.status != "done":
                        self._update_run_state(kb_path, status="stopped", reason="codegen_failed", iteration=iteration)
                        break
                    self._increment_run_counter(kb_path, "codegen_calls")
                    implementation = implementation_task.result or {}
                    benchmarks = implementation.get("benchmarks", [])
                    if benchmarks:
                        break
                    amendment_feedback = implementation.get("rejected_benchmarks", [])
                    self._append_codegen_history(
                        kb_path=kb_path,
                        iteration=iteration,
                        amendment_round=amend_round,
                        implementation=implementation,
                    )
                if not implementation_task or implementation_task.status != "done":
                    break
                benchmarks = implementation.get("benchmarks", [])
                if not benchmarks:
                    self._update_run_state(kb_path, status="stopped", reason="no_feasible_implementation", iteration=iteration)
                    break
                self._append_codegen_history(
                    kb_path=kb_path,
                    iteration=iteration,
                    amendment_round=int(implementation.get("negotiation", {}).get("amendment_round", 0)),
                    implementation=implementation,
                )

                execute_task = Task(
                    id=f"iter{iteration}-execute-implementation-0",
                    kind="execute_implementation",
                    payload={
                        "iteration": iteration,
                        "implementation": implementation,
                        "samples": samples,
                        "interval_sec": interval_sec,
                    },
                )
                completed.append(self._run_with_retry(execute_task, ctx))
                self._persist_run_log(run_dir, completed)
                if execute_task.status != "done":
                    self._update_run_state(kb_path, status="stopped", reason="runner_failed", iteration=iteration)
                    break
                self._increment_run_counter(kb_path, "runner_calls")

                analysis_task = Task(
                    id=f"iter{iteration}-llm-analysis-0",
                    kind="llm_analyze_update",
                    payload={
                        "intent": intent,
                        "iteration": iteration,
                        "max_iterations": max_iterations,
                        "kb_path": str(kb_path),
                        "knowledge_model_path": str(knowledge_model_path),
                        "plan": plan,
                        "execution_results": (execute_task.result or {}).get("results", []),
                    },
                )
                completed.append(self._run_with_retry(analysis_task, ctx))
                self._persist_run_log(run_dir, completed)
                if analysis_task.status != "done":
                    self._update_run_state(kb_path, status="stopped", reason="analyzer_failed", iteration=iteration)
                    break
                self._increment_run_counter(kb_path, "analyzer_calls")
                self._update_run_state(kb_path, status="running", reason="iteration_completed", iteration=iteration + 1)

                analysis_result = self._canonicalize_markdown_result("llm_analyze_update", analysis_task.result or {})
                analysis_task.result = analysis_result
                veto_next_plan = bool(analysis_result.get("veto_next_plan", False))
                if float(analysis_result.get("coverage_score", 0.0)) >= target_coverage and not veto_next_plan:
                    self._update_run_state(kb_path, status="stopped", reason="target_coverage_reached", iteration=iteration + 1)
                    break
                if bool(analysis_result.get("stop", False)) and not veto_next_plan:
                    self._update_run_state(kb_path, status="stopped", reason="analyzer_requested_stop", iteration=iteration + 1)
                    break
            else:
                self._update_run_state(kb_path, status="stopped", reason="max_iterations_reached", iteration=max_iterations)

            final_task = Task(id="final-autonomous-report-0", kind="autonomous_report", payload={})
            completed.append(self._run_with_retry(final_task, ctx))
            self._persist_run_log(run_dir, completed)

            return {
                "mode": "autonomous",
                "run_id": run_id,
                "run_dir": str(run_dir),
                "intent": intent,
                "target_coverage": target_coverage,
                "tasks": [self._task_to_dict(t) for t in completed],
            }
        finally:
            self._close_run_debug_log()

    def _load_kb(self, kb_path: Path) -> dict[str, Any]:
        if not kb_path.exists():
            return {}
        return json.loads(kb_path.read_text(encoding="utf-8"))

    def _detect_available_tools(self) -> dict[str, bool]:
        tools = [
            "nvidia-smi",
            "ncu",
            "nsys",
            "python",
            "bandwidthTest",
            "gpu-burn",
            "cutlass_profiler",
            "nvbench",
        ]
        return {tool: shutil.which(tool) is not None for tool in tools}

    def _max_amendments(self, schema_contract: dict[str, Any]) -> int:
        negotiation = schema_contract.get("negotiation_policy", {}) if isinstance(schema_contract, dict) else {}
        if not isinstance(negotiation, dict):
            return 0
        try:
            return max(0, int(negotiation.get("max_amendment_rounds", 0)))
        except Exception:
            return 0

    def _append_codegen_history(
        self, kb_path: Path, iteration: int, amendment_round: int, implementation: dict[str, Any]
    ) -> None:
        kb = self._load_kb(kb_path)
        kb.setdefault("codegen_history", [])
        kb["codegen_history"].append(
            {
                "iteration": iteration,
                "amendment_round": amendment_round,
                "accepted_count": len(implementation.get("benchmarks", [])),
                "rejected_count": len(implementation.get("rejected_benchmarks", [])),
                "rejected": implementation.get("rejected_benchmarks", []),
                "policy": implementation.get("negotiation", {}).get("policy", {}),
                "artifact": implementation.get("artifact"),
                "timestamp": time.time(),
            }
        )
        write_json(kb_path, kb)

    def _append_research_history(self, kb_path: Path, iteration: int, research: dict[str, Any]) -> None:
        research = self._canonicalize_markdown_result("llm_research", research)
        kb = self._load_kb(kb_path)
        kb.setdefault("research_history", [])
        findings = research.get("findings", [])
        request_summary = str(research.get("request_summary", "")).strip()
        unanswered_questions = research.get("unanswered_questions", [])
        kb["research_history"].append(
            {
                "iteration": iteration,
                "planner": research.get("planner"),
                "reason": research.get("reason"),
                "request_summary": request_summary,
                "findings_count": len(findings) if isinstance(findings, list) else 0,
                "findings": findings if isinstance(findings, list) else [],
                "proposed_dimensions": research.get("proposed_dimensions", []),
                "unanswered_questions": unanswered_questions if isinstance(unanswered_questions, list) else [],
                "artifact": research.get("artifact"),
                "artifact_md": research.get("artifact_md"),
                "timestamp": time.time(),
            }
        )
        proposed = research.get("proposed_dimensions", [])
        if isinstance(proposed, list):
            target = kb.get("target_dimensions", [])
            if not isinstance(target, list):
                target = []
            for dim in proposed:
                item = str(dim).strip()
                if item and item not in target:
                    target.append(item)
            kb["target_dimensions"] = target
        kb["latest_research"] = {
            "iteration": iteration,
            "artifact": research.get("artifact"),
            "artifact_md": research.get("artifact_md"),
            "request_summary": request_summary,
            "findings_count": len(findings) if isinstance(findings, list) else 0,
            "proposed_dimensions": research.get("proposed_dimensions", []),
        }
        write_json(kb_path, kb)

    def _append_planner_research_outputs(self, kb_path: Path, iteration: int, result: dict[str, Any]) -> None:
        result = self._canonicalize_markdown_result("llm_plan_research", result)
        kb = self._load_kb(kb_path)
        kb.setdefault("proposal_history", [])
        research_request = result.get("research_request", {}) if isinstance(result.get("research_request", {}), dict) else {}
        kb["proposal_history"].append(
            {
                "iteration": iteration,
                "planner": result.get("planner"),
                "reason": result.get("reason"),
                "research_request_artifact": result.get("research_request_artifact"),
                "research_request_meta_artifact": result.get("research_request_meta_artifact"),
                "research_request": research_request,
                "proposal": None,
                "timestamp": time.time(),
            }
        )
        write_json(kb_path, kb)

    def _append_planner_outputs(self, kb_path: Path, iteration: int, plan: dict[str, Any]) -> None:
        plan = self._canonicalize_markdown_result("llm_plan_proposal", plan)
        kb = self._load_kb(kb_path)
        kb.setdefault("proposal_history", [])
        proposal = plan.get("proposal", {})
        kb["current_proposal"] = proposal if isinstance(proposal, dict) else {}
        kb["proposal_history"].append(
            {
                "iteration": iteration,
                "planner": plan.get("planner"),
                "reason": plan.get("reason"),
                "artifact": plan.get("proposal_artifact"),
                "artifact_md": plan.get("proposal_md_artifact"),
                "proposal": proposal if isinstance(proposal, dict) else {},
                "timestamp": time.time(),
            }
        )
        write_json(kb_path, kb)

    def _canonicalize_markdown_result(self, task_kind: str, result: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(result, dict):
            return {}
        normalized = dict(result)
        if task_kind == "llm_plan_research":
            parsed = parse_research_request_markdown(self._read_artifact_text(result.get("research_request_artifact")))
            if any(parsed.values()):
                normalized["research_request"] = parsed
            return normalized
        if task_kind == "llm_research":
            parsed = parse_research_markdown(self._read_artifact_text(result.get("artifact_md")))
            if parsed.get("request_summary", "").strip():
                normalized["request_summary"] = parsed["request_summary"]
            normalized["proposed_dimensions"] = parsed.get("proposed_dimensions", [])
            normalized["unanswered_questions"] = parsed.get("unanswered_questions", [])
            normalized["findings"] = parsed.get("findings", [])
            return normalized
        if task_kind == "llm_plan_proposal":
            parsed = parse_proposal_markdown(self._read_artifact_text(result.get("proposal_md_artifact")))
            if parsed.get("proposal_summary", "").strip() or parsed.get("proposals"):
                normalized["proposal"] = parsed
            return normalized
        if task_kind == "llm_analyze_update":
            parsed = parse_analysis_markdown(self._read_artifact_text(result.get("artifact_md")))
            if parsed.get("summary", "").strip():
                normalized["summary"] = parsed["summary"]
            if parsed.get("reason", "").strip():
                normalized["reason"] = parsed["reason"]
            normalized["covered_dimensions"] = parsed.get("covered_dimensions", [])
            normalized["required_observability"] = parsed.get("required_observability", [])
            normalized["claims"] = parsed.get("claims", [])
            normalized["claims_added"] = len(parsed.get("claims", []))
            normalized["stop"] = bool(parsed.get("stop", normalized.get("stop", False)))
            return normalized
        return normalized

    def _increment_run_counter(self, kb_path: Path, counter_name: str) -> None:
        kb = self._load_kb(kb_path)
        run_state = kb.setdefault("run_state", {})
        try:
            run_state[counter_name] = int(run_state.get(counter_name, 0)) + 1
        except Exception:
            run_state[counter_name] = 1
        write_json(kb_path, kb)

    def _update_run_state(self, kb_path: Path, status: str, reason: str, iteration: int) -> None:
        kb = self._load_kb(kb_path)
        run_state = kb.setdefault("run_state", {})
        run_state["status"] = status
        run_state["reason"] = reason
        run_state["iterations_completed"] = max(int(run_state.get("iterations_completed", 0)), int(iteration))
        run_state["updated_at"] = time.time()
        write_json(kb_path, kb)

    def _append_pending_contract_amendments(
        self, kb_path: Path, iteration: int, proposer: str, amendments: list[dict[str, Any]]
    ) -> None:
        if not isinstance(amendments, list) or not amendments:
            return
        kb = self._load_kb(kb_path)
        kb.setdefault("pending_contract_amendments", [])
        for item in amendments:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            change = str(item.get("change", "")).strip()
            if not path or not change:
                continue
            kb["pending_contract_amendments"].append(
                {
                    "iteration": iteration,
                    "proposer": proposer,
                    "path": path,
                    "change": change,
                    "rationale": str(item.get("rationale", "")).strip(),
                    "priority": str(item.get("priority", "medium")).strip().lower(),
                    "timestamp": time.time(),
                }
            )
        write_json(kb_path, kb)

    def _finalize_contract(self, kb_path: Path, iteration: int, finalized_by: str, reason: str) -> None:
        kb = self._load_kb(kb_path)
        contract = kb.get("schema_contract", {})
        if not isinstance(contract, dict):
            contract = {}
        pending = kb.get("pending_contract_amendments", [])
        if not isinstance(pending, list):
            pending = []
        kb.setdefault("contract_history", [])
        kb["contract_history"].append(
            {
                "iteration": iteration,
                "finalized_by": finalized_by,
                "reason": reason,
                "applied_amendments": pending,
                "contract": contract,
                "timestamp": time.time(),
            }
        )
        kb["pending_contract_amendments"] = []
        write_json(kb_path, kb)

    def _run_stage_parallel(self, tasks: list[Task], ctx: AgentContext) -> list[Task]:
        if not tasks:
            return []

        workers = max(1, len(tasks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._run_with_retry, task, ctx) for task in tasks]
            return [f.result() for f in futures]

    def _run_with_retry(self, task: Task, ctx: AgentContext) -> Task:
        agent = self._find_agent(task)
        if not agent:
            task.status = "failed"
            task.error = f"No agent for task kind={task.kind}"
            self._emit_task_trace(task, ctx=ctx, agent_name="unassigned", phase="failed")
            return task

        max_attempts = self.retry_policy.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            task.attempts = attempt
            self._emit_task_trace(task, ctx=ctx, agent_name=agent.name, phase="start")
            try:
                task.result = agent.run(task, ctx)
                task.status = "done"
                task.error = None
                self._emit_task_trace(task, ctx=ctx, agent_name=agent.name, phase="done")
                return task
            except Exception as exc:  # noqa: BLE001
                task.status = "failed"
                task.error = str(exc)
                self._emit_task_trace(task, ctx=ctx, agent_name=agent.name, phase="failed")
                if attempt < max_attempts:
                    time.sleep(self.retry_policy.retry_delay_sec)

        return task

    def _find_agent(self, task: Task) -> Agent | None:
        for agent in self.agents:
            if agent.can_handle(task):
                return agent
        return None

    def _find_monitor_agent(self) -> Agent | None:
        probe = Task(id="monitor-probe", kind="monitor_communications", payload={})
        for agent in self.agents:
            if agent.can_handle(probe):
                return agent
        return None

    def _attach_run_debug_log(self, run_dir: Path) -> None:
        if not self._trace_stream_is_default:
            return
        self._close_run_debug_log()
        log_path = run_dir / "debug.log"
        self._trace_log_handle = log_path.open("a", encoding="utf-8")
        self.trace_stream = self._trace_log_handle

    def _close_run_debug_log(self) -> None:
        if self._trace_log_handle is None:
            return
        try:
            self._trace_log_handle.close()
        finally:
            self._trace_log_handle = None
            self.trace_stream = sys.stderr

    def _default_conversation_stream(self) -> Any:
        try:
            if sys.stdout.isatty():
                return sys.stdout
        except Exception:
            pass
        return sys.stderr

    def _persist_run_log(self, run_dir: Path, tasks: list[Task]) -> None:
        run_log = run_dir / "run_log.json"
        write_json(run_log, [self._task_to_dict(t) for t in tasks])

    def _emit_task_trace(self, task: Task, ctx: AgentContext, agent_name: str, phase: str) -> None:
        if task.kind == "monitor_communications":
            return
        header = self._trace_header(task=task, agent_name=agent_name, phase=phase)
        detail = self._trace_task_detail(task=task, phase=phase)
        event = self._build_trace_event(task=task, agent_name=agent_name, phase=phase, header=header, detail=detail)
        self._record_communication_event(ctx=ctx, event=event)
        if not self.emit_live_trace:
            return
        sections = [header]
        if detail:
            sections.append(detail)
        print("\n".join(sections), file=self.trace_stream, flush=True)

    def _build_trace_event(self, task: Task, agent_name: str, phase: str, header: str, detail: str) -> dict[str, Any]:
        _ = (header, detail)
        iteration = task.payload.get("iteration")
        sender = "orchestrator" if phase == "start" else agent_name
        recipient = agent_name if phase == "start" else "orchestrator"
        return {
            "ts": time.time(),
            "phase": phase,
            "task_id": task.id,
            "task_kind": task.kind,
            "iteration": iteration,
            "attempt": task.attempts,
            "agent": agent_name,
            "sender": sender,
            "recipient": recipient,
            "summary": self._conversation_utterance(task=task, agent_name=agent_name, phase=phase),
            "message": self._conversation_detail(task=task, phase=phase),
        }

    def _conversation_utterance(self, task: Task, agent_name: str, phase: str) -> str:
        iteration = self._conversation_iteration_phrase(task.payload.get("iteration"))
        intent = str(task.payload.get("intent", "")).strip()
        if phase == "start":
            if task.kind == "collect_system_info":
                return "Please collect the basic system information for this run."
            if task.kind == "llm_plan_research":
                return f"Please decide what research we need next{iteration}.{self._intent_sentence(intent)}"
            if task.kind == "llm_research":
                return f"Please research the current open questions{iteration}."
            if task.kind == "llm_plan_proposal":
                return f"Please propose the next benchmark{iteration}.{self._intent_sentence(intent)}"
            if task.kind == "llm_generate_implementation":
                return f"Please turn the current proposal into an executable benchmark{iteration}."
            if task.kind == "execute_implementation":
                return f"Please execute the current benchmark plan{iteration}."
            if task.kind == "llm_analyze_update":
                return f"Please analyze the latest execution results and update the model{iteration}."
            if task.kind == "run_workload":
                command = str(task.payload.get("command", "")).strip()
                return f"Please run this command{': ' + command if command else '.'}"
            return f"Please handle `{task.kind}`{iteration}."

        result = task.result if isinstance(task.result, dict) else {}
        reason = str(result.get("reason", "")).strip()
        if phase == "failed":
            return f"I could not complete `{task.kind}` because {task.error or 'an unknown error occurred'}."
        if task.kind == "collect_system_info":
            return "I collected the system information."
        if task.kind == "llm_plan_research":
            return self._sentence_with_reason("I prepared the next research request.", reason)
        if task.kind == "llm_research":
            request_summary = str(result.get("request_summary", "")).strip()
            base = "I finished the research pass."
            if request_summary:
                base += f" I focused on: {request_summary}"
            return self._sentence_with_reason(base, reason)
        if task.kind == "llm_plan_proposal":
            return self._sentence_with_reason("I prepared the next benchmark proposal.", reason)
        if task.kind == "llm_generate_implementation":
            accepted = len(result.get("benchmarks", []))
            rejected = len(result.get("rejected_benchmarks", []))
            base = f"I generated an implementation draft with {accepted} accepted benchmark(s) and {rejected} rejected one(s)."
            return self._sentence_with_reason(base, reason)
        if task.kind == "execute_implementation":
            return f"I executed {int(result.get('benchmarks_run', 0))} benchmark(s)."
        if task.kind == "llm_analyze_update":
            summary = str(result.get("summary", "")).strip()
            return summary or self._sentence_with_reason("I analyzed the latest results and updated the model.", reason)
        if task.kind == "run_workload":
            return f"I ran the command and it exited with code {result.get('returncode', 'unknown')}."
        if task.kind in {"autonomous_report", "report"}:
            return "I wrote the report."
        return self._sentence_with_reason(f"I completed `{task.kind}`.", reason)

    def _conversation_detail(self, task: Task, phase: str) -> str:
        if phase == "start":
            detail = self._conversation_start_detail(task)
            return detail
        if phase == "failed":
            return ""
        result = task.result if isinstance(task.result, dict) else {}
        if not result:
            return ""
        parts: list[str] = []
        lines = self._conversation_result_lines(task.kind, result)
        if lines:
            parts.append("\n".join(lines))
        parts.extend(self._conversation_artifact_sections(task.kind, result))
        return "\n\n".join(part for part in parts if part)

    def _conversation_start_detail(self, task: Task) -> str:
        lines: list[str] = []
        intent = str(task.payload.get("intent", "")).strip()
        command = str(task.payload.get("command", "")).strip()
        if intent:
            lines.append(f"Intent: {intent}")
        if command:
            lines.append(f"Command: {command}")
        max_benchmarks = task.payload.get("max_benchmarks")
        if max_benchmarks not in {None, ""}:
            lines.append(f"Benchmark budget: {max_benchmarks}")
        return "\n".join(lines)

    def _conversation_result_lines(self, task_kind: str, result: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        if task_kind == "run_workload":
            lines.append(f"Return code: {result.get('returncode', 'unknown')}")
        summary = str(result.get("summary", "")).strip()
        if task_kind == "llm_analyze_update" and summary:
            lines.append(f"Summary: {summary}")
        request_summary = str(result.get("request_summary", "")).strip()
        if task_kind == "llm_research" and request_summary:
            lines.append(f"Research focus: {request_summary}")
        if task_kind == "execute_implementation":
            lines.append(f"Benchmarks executed: {int(result.get('benchmarks_run', 0))}")
        if task_kind == "llm_generate_implementation":
            lines.append(f"Accepted benchmarks: {len(result.get('benchmarks', []))}")
            lines.append(f"Rejected benchmarks: {len(result.get('rejected_benchmarks', []))}")
        stdout = str(result.get("stdout", "")).strip()
        stderr = str(result.get("stderr", "")).strip()
        if stdout:
            lines.append(f"Standard output:\n{stdout}")
        if stderr:
            lines.append(f"Standard error:\n{stderr}")
        return lines

    def _conversation_artifact_sections(self, task_kind: str, result: dict[str, Any]) -> list[str]:
        sections: list[str] = []
        for title, artifact_key in self._conversation_artifact_keys_for_task(task_kind):
            content = self._read_artifact_text(result.get(artifact_key))
            if not content:
                continue
            sections.append(f"{title}:\n{content}")
        return sections

    def _conversation_artifact_keys_for_task(self, task_kind: str) -> list[tuple[str, str]]:
        if task_kind == "llm_plan_research":
            return [("Research request", "research_request_artifact")]
        if task_kind == "llm_research":
            return [("Research notes", "artifact_md")]
        if task_kind == "llm_plan_proposal":
            return [("Proposal", "proposal_md_artifact")]
        if task_kind == "llm_generate_implementation":
            return [
                ("Prompt I used", "prompt_artifact"),
                ("Implementation memo", "raw_artifact"),
                ("Implementation notes", "artifact_md"),
            ]
        if task_kind == "execute_implementation":
            return [("Execution report", "artifact_md")]
        if task_kind == "llm_analyze_update":
            return [("Analysis update", "artifact_md")]
        return []

    def _conversation_iteration_phrase(self, iteration: Any) -> str:
        if iteration is None:
            return ""
        try:
            return f" for iteration {int(iteration)}"
        except Exception:
            return f" for iteration {iteration}"

    def _intent_sentence(self, intent: str) -> str:
        if not intent:
            return ""
        return f' The intent is "{intent}".'

    def _sentence_with_reason(self, base: str, reason: str) -> str:
        if not reason:
            return base
        if base.endswith("."):
            base = base[:-1]
        return f"{base} Reason: {reason}"

    def _record_communication_event(self, ctx: AgentContext, event: dict[str, Any]) -> None:
        if self.monitor_agent is None:
            return
        monitor_task = Task(
            id=f"monitor-{event.get('task_id', 'task')}-{event.get('phase', 'event')}",
            kind="monitor_communications",
            payload={"event": event},
        )
        try:
            monitor_result = self.monitor_agent.run(monitor_task, ctx)
            self._emit_live_conversation(event=event, monitor_result=monitor_result)
        except Exception as exc:  # noqa: BLE001
            if self.emit_live_trace:
                print(f"[trace] monitor-error {exc}", file=self.trace_stream, flush=True)

    def _emit_live_conversation(self, event: dict[str, Any], monitor_result: dict[str, Any] | None) -> None:
        if not self.emit_live_conversation:
            return
        screen_output = str((monitor_result or {}).get("screen_output", "")).strip()
        if not screen_output:
            return
        print(screen_output, file=self.conversation_stream, flush=True)

    def _trace_header(self, task: Task, agent_name: str, phase: str) -> str:
        iteration = task.payload.get("iteration")
        iteration_label = "iter=?"
        if iteration is not None:
            try:
                iteration_label = f"iter={int(iteration):02d}"
            except Exception:
                iteration_label = f"iter={iteration}"
        return (
            f"[trace] {phase.upper()} {iteration_label} "
            f"task={task.id} kind={task.kind} agent={agent_name} attempt={task.attempts}"
        )

    def _trace_task_detail(self, task: Task, phase: str) -> str:
        if phase == "start":
            payload_summary = self._summarize_payload(task.payload)
            return f"payload: {payload_summary}" if payload_summary else ""
        if phase == "failed":
            return f"error: {task.error or 'unknown task failure'}"
        result = task.result if isinstance(task.result, dict) else {}
        if not result:
            return "result: <empty>"

        summary_lines = self._result_summary_lines(task.kind, result)
        artifact_sections = self._result_artifact_sections(task.kind, result)
        detail_parts: list[str] = []
        if summary_lines:
            detail_parts.append("\n".join(summary_lines))
        detail_parts.extend(artifact_sections)
        return "\n\n".join(part for part in detail_parts if part)

    def _summarize_payload(self, payload: dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""
        parts: list[str] = []
        for key in (
            "intent",
            "command",
            "max_iterations",
            "max_benchmarks",
            "amendment_round",
        ):
            value = payload.get(key)
            if value is None or value == "" or value == []:
                continue
            parts.append(f"{key}={value}")
        return ", ".join(parts)

    def _result_summary_lines(self, task_kind: str, result: dict[str, Any]) -> list[str]:
        lines = [f"result_kind: {task_kind}"]
        for key in ("reason", "summary", "request_summary", "cwd", "artifact", "artifact_md"):
            value = result.get(key)
            if value is None or value == "" or value == []:
                continue
            lines.append(f"{key}: {value}")
        if task_kind == "run_workload":
            lines.extend(self._stdout_stderr_lines(result))
        if task_kind == "execute_implementation":
            lines.append(f"benchmarks_run: {result.get('benchmarks_run', 0)}")
        if task_kind == "llm_generate_implementation":
            lines.append(f"accepted_benchmarks: {len(result.get('benchmarks', []))}")
            lines.append(f"rejected_benchmarks: {len(result.get('rejected_benchmarks', []))}")
        return lines

    def _stdout_stderr_lines(self, result: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        stdout = str(result.get("stdout", "")).strip()
        stderr = str(result.get("stderr", "")).strip()
        if stdout:
            lines.append("stdout:")
            lines.append(stdout)
        if stderr:
            lines.append("stderr:")
            lines.append(stderr)
        return lines

    def _result_artifact_sections(self, task_kind: str, result: dict[str, Any]) -> list[str]:
        sections: list[str] = []
        for title, artifact_key in self._artifact_keys_for_task(task_kind):
            content = self._read_artifact_text(result.get(artifact_key))
            if not content:
                continue
            sections.append(f"{title}:\n{content}")
        return sections

    def _artifact_keys_for_task(self, task_kind: str) -> list[tuple[str, str]]:
        if task_kind == "llm_schema_contract":
            return [("schema-contract", "artifact_md"), ("schema-contract-json", "artifact")]
        if task_kind == "llm_plan_research":
            return [
                ("research-request-raw", "research_request_raw_artifact"),
                ("research-request", "research_request_artifact"),
            ]
        if task_kind == "llm_research":
            return [("research-raw", "raw_artifact"), ("research-memo", "artifact_md")]
        if task_kind == "llm_plan_proposal":
            return [("proposal-raw", "proposal_raw_artifact"), ("proposal-memo", "proposal_md_artifact")]
        if task_kind == "llm_generate_implementation":
            return [
                ("implementation-prompt", "prompt_artifact"),
                ("implementation-raw", "raw_artifact"),
                ("implementation-memo", "artifact_md"),
            ]
        if task_kind in {"execute_implementation", "llm_analyze_update", "autonomous_report", "report"}:
            return [("artifact", "artifact_md"), ("artifact", "artifact")]
        return []

    def _read_artifact_text(self, path_value: Any) -> str:
        path_text = str(path_value or "").strip()
        if not path_text:
            return ""
        path = Path(path_text)
        if not path.exists() or not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def _task_to_dict(self, task: Task) -> dict[str, Any]:
        return {
            "id": task.id,
            "kind": task.kind,
            "status": task.status,
            "payload": task.payload,
            "result": task.result,
            "error": task.error,
            "attempts": task.attempts,
        }


def build_default_orchestrator(retry_policy: RetryPolicy | None = None) -> Orchestrator:
    return Orchestrator(agents=default_agents(), retry_policy=retry_policy)


def build_orchestrator_with_planner(
    retry_policy: RetryPolicy | None = None,
    planner_backend: str = "heuristic",
    planner_model: str = "gpt-5.4",
) -> Orchestrator:
    workflow_backend: Any = HeuristicWorkflowBackend()
    if planner_backend == "openai":
        workflow_backend = ResilientWorkflowBackend(
            primary=OpenAIWorkflowBackend(model=planner_model),
            fallback=HeuristicWorkflowBackend(),
        )
    return Orchestrator(agents=default_agents(workflow_backend=workflow_backend), retry_policy=retry_policy)
