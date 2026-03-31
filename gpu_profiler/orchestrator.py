import json
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .agents import Agent, default_agents
from .llm import HeuristicWorkflowBackend, OpenAIWorkflowBackend, ResilientWorkflowBackend
from .models import AgentContext, RetryPolicy, Task
from .store import write_json

DEFAULT_TARGET_DIMENSIONS: list[str] = []


class Orchestrator:
    def __init__(self, agents: list[Agent], retry_policy: RetryPolicy | None = None):
        self.agents = agents
        self.retry_policy = retry_policy or RetryPolicy()

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

        dimensions = target_dimensions or DEFAULT_TARGET_DIMENSIONS
        knowledge_base: dict[str, Any] = {
            "intent": intent,
            "target_dimensions": dimensions,
            "covered_dimensions": [],
            "coverage_score": 0.0,
            "target_coverage": target_coverage,
            "available_tools": self._detect_available_tools(),
            "history": [],
            "claims": [],
            "research_history": [],
            "pending_contract_amendments": [],
            "contract_history": [],
        }
        kb_path = run_dir / "performance_model.json"
        write_json(kb_path, knowledge_base)

        completed: list[Task] = []

        sys_task = Task(id="boot-collect_system_info-0", kind="collect_system_info", payload={})
        completed.append(self._run_with_retry(sys_task, ctx))
        self._persist_run_log(run_dir, completed)

        for iteration in range(max_iterations):
            knowledge_base = self._load_kb(kb_path)
            contract_task = Task(
                id=f"iter{iteration}-llm-schema-0",
                kind="llm_schema_contract",
                payload={
                    "intent": intent,
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "knowledge_base": knowledge_base,
                },
            )
            completed.append(self._run_with_retry(contract_task, ctx))
            self._persist_run_log(run_dir, completed)
            if contract_task.status != "done":
                break
            schema_contract = (contract_task.result or {}).get("schema_contract", {})
            if isinstance(schema_contract, dict):
                knowledge_base["schema_contract"] = schema_contract
                write_json(kb_path, knowledge_base)
                self._finalize_contract(
                    kb_path=kb_path,
                    iteration=iteration,
                    finalized_by=(contract_task.result or {}).get("planner", "unknown-planner"),
                    reason=(contract_task.result or {}).get("reason", ""),
                )
                knowledge_base = self._load_kb(kb_path)

            research_task = Task(
                id=f"iter{iteration}-llm-research-0",
                kind="llm_research",
                payload={
                    "intent": intent,
                    "iteration": iteration,
                    "knowledge_base": knowledge_base,
                    "max_sources": 10,
                },
            )
            completed.append(self._run_with_retry(research_task, ctx))
            self._persist_run_log(run_dir, completed)
            if research_task.status != "done":
                break
            research = research_task.result or {}
            self._append_research_history(kb_path=kb_path, iteration=iteration, research=research)
            knowledge_base = self._load_kb(kb_path)

            plan_task = Task(
                id=f"iter{iteration}-llm-plan-0",
                kind="llm_plan",
                payload={
                    "intent": intent,
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "max_benchmarks": max_benchmarks,
                    "knowledge_base": knowledge_base,
                },
            )
            completed.append(self._run_with_retry(plan_task, ctx))
            self._persist_run_log(run_dir, completed)
            if plan_task.status != "done":
                break

            plan = plan_task.result or {}
            if plan.get("stop", False):
                break

            max_amendment_rounds = self._max_amendments(knowledge_base.get("schema_contract", {}))
            suite: dict[str, Any] = {}
            suite_task: Task | None = None
            amendment_feedback: list[dict[str, Any]] = []
            for amend_round in range(max_amendment_rounds + 1):
                suite_task = Task(
                    id=f"iter{iteration}-llm-suite-{amend_round}",
                    kind="llm_generate_suite",
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
                completed.append(self._run_with_retry(suite_task, ctx))
                self._persist_run_log(run_dir, completed)
                if suite_task.status != "done":
                    break
                suite = suite_task.result or {}
                benchmarks = suite.get("benchmarks", [])
                if benchmarks:
                    break
                amendment_feedback = suite.get("rejected_benchmarks", [])
                self._append_negotiation_history(
                    kb_path=kb_path,
                    iteration=iteration,
                    amendment_round=amend_round,
                    suite=suite,
                )
            if not suite_task or suite_task.status != "done":
                break
            benchmarks = suite.get("benchmarks", [])
            if not benchmarks:
                break
            self._append_negotiation_history(
                kb_path=kb_path,
                iteration=iteration,
                amendment_round=int(suite.get("negotiation", {}).get("amendment_round", 0)),
                suite=suite,
            )
            self._append_pending_contract_amendments(
                kb_path=kb_path,
                iteration=iteration,
                proposer=(suite.get("planner") or "llm-suite"),
                amendments=suite.get("contract_amendments", []),
            )

            execute_task = Task(
                id=f"iter{iteration}-execute-suite-0",
                kind="execute_suite",
                payload={
                    "iteration": iteration,
                    "suite": suite,
                    "samples": samples,
                    "interval_sec": interval_sec,
                },
            )
            completed.append(self._run_with_retry(execute_task, ctx))
            self._persist_run_log(run_dir, completed)
            if execute_task.status != "done":
                break

            analysis_task = Task(
                id=f"iter{iteration}-llm-analysis-0",
                kind="llm_analyze_update",
                payload={
                    "intent": intent,
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "kb_path": str(kb_path),
                    "plan": plan,
                    "suite_results": (execute_task.result or {}).get("results", []),
                },
            )
            completed.append(self._run_with_retry(analysis_task, ctx))
            self._persist_run_log(run_dir, completed)
            if analysis_task.status != "done":
                break

            analysis_result = analysis_task.result or {}
            self._append_pending_contract_amendments(
                kb_path=kb_path,
                iteration=iteration,
                proposer=(analysis_result.get("planner") or "llm-analysis"),
                amendments=analysis_result.get("contract_amendments", []),
            )
            veto_next_plan = bool(analysis_result.get("veto_next_plan", False))
            if float(analysis_result.get("coverage_score", 0.0)) >= target_coverage and not veto_next_plan:
                break
            if bool(analysis_result.get("stop", False)) and not veto_next_plan:
                break

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

    def _append_negotiation_history(self, kb_path: Path, iteration: int, amendment_round: int, suite: dict[str, Any]) -> None:
        kb = self._load_kb(kb_path)
        kb.setdefault("negotiation_history", [])
        kb["negotiation_history"].append(
            {
                "iteration": iteration,
                "amendment_round": amendment_round,
                "accepted_count": len(suite.get("benchmarks", [])),
                "rejected_count": len(suite.get("rejected_benchmarks", [])),
                "rejected": suite.get("rejected_benchmarks", []),
                "policy": suite.get("negotiation", {}).get("policy", {}),
                "timestamp": time.time(),
            }
        )
        write_json(kb_path, kb)

    def _append_research_history(self, kb_path: Path, iteration: int, research: dict[str, Any]) -> None:
        kb = self._load_kb(kb_path)
        kb.setdefault("research_history", [])
        findings = research.get("findings", [])
        kb["research_history"].append(
            {
                "iteration": iteration,
                "planner": research.get("planner"),
                "reason": research.get("reason"),
                "findings_count": len(findings) if isinstance(findings, list) else 0,
                "findings": findings if isinstance(findings, list) else [],
                "proposed_dimensions": research.get("proposed_dimensions", []),
                "artifact": research.get("artifact"),
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
            "findings_count": len(findings) if isinstance(findings, list) else 0,
        }
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
            return task

        max_attempts = self.retry_policy.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            task.attempts = attempt
            try:
                task.result = agent.run(task, ctx)
                task.status = "done"
                task.error = None
                return task
            except Exception as exc:  # noqa: BLE001
                task.status = "failed"
                task.error = str(exc)
                if attempt < max_attempts:
                    time.sleep(self.retry_policy.retry_delay_sec)

        return task

    def _find_agent(self, task: Task) -> Agent | None:
        for agent in self.agents:
            if agent.can_handle(task):
                return agent
        return None

    def _persist_run_log(self, run_dir: Path, tasks: list[Task]) -> None:
        run_log = run_dir / "run_log.json"
        write_json(run_log, [self._task_to_dict(t) for t in tasks])

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
