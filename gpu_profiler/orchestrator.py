import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .agents import Agent, default_agents
from .models import AgentContext, RetryPolicy, Task
from .store import write_json


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
            "run_id": run_id,
            "run_dir": str(run_dir),
            "tasks": [self._task_to_dict(t) for t in completed],
        }

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
