import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from gpu_profiler.agents import Agent, WorkloadRunnerAgent
from gpu_profiler.models import AgentContext, RetryPolicy, Task
from gpu_profiler.orchestrator import Orchestrator


def run_cli(tmp_path: Path, *args: str) -> dict:
    cmd = [sys.executable, "gpu_autoprofile.py", *args]
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def copy_app(tmp_path: Path) -> None:
    root = Path(__file__).parent
    (tmp_path / "gpu_autoprofile.py").write_text((root / "gpu_autoprofile.py").read_text(encoding="utf-8"), encoding="utf-8")
    shutil.copytree(root / "gpu_profiler", tmp_path / "gpu_profiler")


def test_run_produces_artifacts(tmp_path):
    copy_app(tmp_path)

    result = run_cli(
        tmp_path,
        "run",
        "--workload",
        "python -c \"print('hello')\"",
        "--out",
        "runs",
        "--samples",
        "1",
        "--interval",
        "0",
    )

    run_dir = tmp_path / result["run_dir"]
    assert run_dir.exists()
    assert (run_dir / "run_log.json").exists()
    assert (run_dir / "analysis.json").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "system_info.json").exists()

    run_log = json.loads((run_dir / "run_log.json").read_text(encoding="utf-8"))
    kinds = [item["kind"] for item in run_log]
    assert kinds[0] == "plan"
    assert "collect_metrics" in kinds
    assert "collect_system_info" in kinds
    assert "run_workload" in kinds
    assert kinds[-2:] == ["analyze", "report"]


def test_workload_result_recorded(tmp_path):
    copy_app(tmp_path)

    result = run_cli(
        tmp_path,
        "run",
        "--workload",
        "python -c \"print(123)\"",
        "--out",
        "runs",
        "--samples",
        "1",
        "--interval",
        "0",
    )

    run_dir = tmp_path / result["run_dir"]
    workload = json.loads((run_dir / "workload_result.json").read_text(encoding="utf-8"))
    assert workload["returncode"] == 0
    assert "123" in workload["stdout"]


def test_autonomous_run_produces_model_artifacts(tmp_path):
    copy_app(tmp_path)

    result = run_cli(
        tmp_path,
        "autonomous",
        "--intent",
        "Develop a performance model for the local GPU",
        "--out",
        "runs",
        "--samples",
        "1",
        "--interval",
        "0",
        "--max-iterations",
        "1",
        "--max-benchmarks",
        "1",
        "--target-coverage",
        "1.0",
    )

    run_dir = tmp_path / result["run_dir"]
    assert run_dir.exists()
    assert result["mode"] == "autonomous"

    assert (run_dir / "performance_model.json").exists()
    assert (run_dir / "autonomous_report.md").exists()
    assert (run_dir / "iterations" / "iter_00" / "research_request.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "research.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "knowledge_model.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "proposal.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "proposal.md").exists()
    assert (run_dir / "iterations" / "iter_00" / "implementation.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "feasibility_report.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "execution_results.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "execution.md").exists()
    assert (run_dir / "iterations" / "iter_00" / "analysis_update.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "analysis.md").exists()

    run_log = json.loads((run_dir / "run_log.json").read_text(encoding="utf-8"))
    kinds = [item["kind"] for item in run_log]
    assert kinds[0] == "collect_system_info"
    assert "llm_research" in kinds
    assert "llm_plan" in kinds
    assert kinds.count("llm_plan") >= 1
    assert "llm_generate_implementation" in kinds
    assert "execute_implementation" in kinds
    assert "llm_analyze_update" in kinds
    assert kinds[-1] == "autonomous_report"


def test_autonomous_openai_backend_falls_back_to_heuristic(tmp_path):
    copy_app(tmp_path)

    result = run_cli(
        tmp_path,
        "autonomous",
        "--intent",
        "Develop a performance model for the local GPU",
        "--planner-backend",
        "openai",
        "--out",
        "runs",
        "--samples",
        "1",
        "--interval",
        "0",
        "--max-iterations",
        "1",
        "--max-benchmarks",
        "1",
        "--target-coverage",
        "1.0",
    )

    run_dir = tmp_path / result["run_dir"]
    run_log = json.loads((run_dir / "run_log.json").read_text(encoding="utf-8"))
    plan_task = next(item for item in run_log if item["kind"] == "llm_plan")
    assert plan_task["status"] == "done"
    assert "fallback" in plan_task["result"]["planner"]
    assert "fallback used" in plan_task["result"]["reason"]
    assert "knowledge_model" in plan_task["result"]
    assert "proposal" in plan_task["result"]
    assert "research_request_artifact" in plan_task["result"]


def test_workload_skip_detection(tmp_path):
    runner = WorkloadRunnerAgent()
    ctx = AgentContext(run_id="skip-test", run_dir=tmp_path)
    task = Task(
        id="skip-0",
        kind="run_workload",
        payload={"command": "python -c \"print('SKIP: missing benchmark binary')\""},
    )
    result = runner.run(task, ctx)
    assert result["returncode"] == 0
    assert result["skipped"] is True


class FlakyAgent(Agent):
    def __init__(self):
        self.calls = 0

    def can_handle(self, task: Task) -> bool:
        return task.kind == "flaky"

    def run(self, task: Task, ctx: AgentContext) -> dict:
        _ = (task, ctx)
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient")
        return {"ok": True}


class SlowAgent(Agent):
    def can_handle(self, task: Task) -> bool:
        return task.kind.startswith("slow")

    def run(self, task: Task, ctx: AgentContext) -> dict:
        _ = ctx
        time.sleep(float(task.payload.get("sleep", 0.2)))
        return {"task": task.kind}


class OrchestratorHarness(Orchestrator):
    def run_custom(self, stages: list[list[dict]], out_dir: str) -> list[Task]:
        run_dir = Path(out_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        ctx = AgentContext(run_id="test-run", run_dir=run_dir)
        completed = []
        for stage_idx, stage in enumerate(stages):
            tasks = [
                Task(id=f"s{stage_idx}-{i}", kind=item["kind"], payload=item.get("payload", {}))
                for i, item in enumerate(stage)
            ]
            completed.extend(self._run_stage_parallel(tasks, ctx))
        return completed


def test_retry_policy_recovers_from_transient_error(tmp_path):
    flaky = FlakyAgent()
    orch = OrchestratorHarness(agents=[flaky], retry_policy=RetryPolicy(max_retries=1, retry_delay_sec=0))

    tasks = orch.run_custom(stages=[[{"kind": "flaky", "payload": {}}]], out_dir=str(tmp_path / "run"))
    assert tasks[0].status == "done"
    assert tasks[0].attempts == 2


def test_parallel_stage_runs_concurrently(tmp_path):
    orch = OrchestratorHarness(agents=[SlowAgent()], retry_policy=RetryPolicy(max_retries=0, retry_delay_sec=0))

    start = time.time()
    tasks = orch.run_custom(
        stages=[[{"kind": "slow-a", "payload": {"sleep": 0.25}}, {"kind": "slow-b", "payload": {"sleep": 0.25}}]],
        out_dir=str(tmp_path / "run"),
    )
    elapsed = time.time() - start

    assert all(t.status == "done" for t in tasks)
    assert elapsed < 0.45
