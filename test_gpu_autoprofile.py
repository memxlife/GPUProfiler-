import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from openai_agent_probe import build_agent_fixture, evaluate_agent_result, summarize_agent_attempts
from openai_research_probe import evaluate_research_result, load_queries, parse_timeout_values, summarize_attempts
from gpu_profiler.agents import (
    Agent,
    CommunicationMonitorAgent,
    LLMCodegenAgent,
    WorkloadRunnerAgent,
    _extract_compile_preflight,
    _preflight_single_benchmark,
)
from gpu_profiler.llm import (
    ANALYSIS_TIMEOUT_SEC,
    CODEGEN_INPUT_HARD_CAP_CHARS,
    PLANNER_INPUT_HARD_CAP_CHARS,
    RESEARCH_TIMEOUT_SEC,
    AnalysisDecision,
    HeuristicWorkflowBackend,
    OpenAIWorkflowBackend,
    ResearchDecision,
    ResilientWorkflowBackend,
    _diagnostic_environment,
    _timeout_diagnostic_payload,
    _benchmark_from_memo,
    _merge_schema_contract,
    _compact_codegen_kb,
    _compact_codegen_plan,
    _compact_planner_kb,
    _enforce_payload_budget,
)
from gpu_profiler.models import AgentContext, RetryPolicy, Task
from gpu_profiler.orchestrator import Orchestrator


def run_cli(tmp_path: Path, *args: str) -> dict:
    cmd = [sys.executable, "gpu_autoprofile.py", *args]
    subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, check=True)
    out_dir = "profiling_runs"
    for index, arg in enumerate(args):
        if arg == "--out" and index + 1 < len(args):
            out_dir = args[index + 1]
            break
    latest_run = sorted((tmp_path / out_dir).glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return json.loads((latest_run / "final_result.json").read_text(encoding="utf-8"))


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
    assert (run_dir / "debug.log").exists()
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
    assert (run_dir / "knowledge_model.json").exists()
    assert (run_dir / "autonomous_report.md").exists()
    assert not (run_dir / "iterations" / "iter_00" / "schema_contract.json").exists()
    assert not (run_dir / "iterations" / "iter_00" / "schema_contract.md").exists()
    assert (run_dir / "iterations" / "iter_00" / "research_request.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "research.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "knowledge_model.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "proposal.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "proposal.md").exists()
    assert (run_dir / "iterations" / "iter_00" / "implementation.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "implementation_prompt.md").exists()
    assert (run_dir / "iterations" / "iter_00" / "feasibility_report.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "execution_results.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "execution.md").exists()
    assert (run_dir / "iterations" / "iter_00" / "analysis_update.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "analysis.md").exists()

    run_log = json.loads((run_dir / "run_log.json").read_text(encoding="utf-8"))
    kinds = [item["kind"] for item in run_log]
    assert kinds[0] == "collect_system_info"
    assert "llm_schema_contract" not in kinds
    assert "llm_research" in kinds
    assert "llm_plan_research" in kinds
    assert "llm_plan_proposal" in kinds
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
    plan_task = next(item for item in run_log if item["kind"] == "llm_plan_proposal")
    assert plan_task["status"] == "done"
    assert "fallback" in plan_task["result"]["planner"]
    assert "fallback used" in plan_task["result"]["reason"]
    assert "proposal" in plan_task["result"]
    research_task = next(item for item in run_log if item["kind"] == "llm_plan_research")
    assert "research_request_artifact" in research_task["result"]


def test_schema_contract_uses_split_planner_outputs():
    contract = _merge_schema_contract({})

    assert contract["research_request_output"]["required_keys"] == ["reason", "research_request"]
    assert contract["proposal_output"]["required_keys"] == ["reason", "proposal"]
    assert "planner_output" not in contract


def test_resilient_workflow_falls_back_when_openai_plan_times_out():
    backend = OpenAIWorkflowBackend(model="gpt-5.4", request_timeout_sec=1.0)

    def _timeout(*_args, **_kwargs):
        raise TimeoutError("OpenAI completion timed out")

    backend._json_completion = _timeout  # type: ignore[method-assign]
    resilient = ResilientWorkflowBackend(primary=backend, fallback=HeuristicWorkflowBackend())

    result = resilient.propose_plan(
        intent="Develop a performance model for the local GPU",
        kb={},
        iteration=0,
        max_iterations=1,
        max_benchmarks=1,
    )

    assert "fallback used" in result.reason
    assert "timed out" in result.reason
    assert "fallback" in result.planner
    assert isinstance(result.knowledge_model, dict)
    assert isinstance(result.proposal, dict)


class SlowPrimaryBackend(HeuristicWorkflowBackend):
    name = "slow-primary"
    request_timeout_sec = 0.2

    def propose_plan(
        self,
        intent: str,
        kb: dict,
        iteration: int,
        max_iterations: int,
        max_benchmarks: int,
    ):
        time.sleep(20)
        return super().propose_plan(intent, kb, iteration, max_iterations, max_benchmarks)


def test_resilient_workflow_terminates_hung_primary_backend():
    resilient = ResilientWorkflowBackend(primary=SlowPrimaryBackend(), fallback=HeuristicWorkflowBackend())

    start = time.time()
    result = resilient.propose_plan(
        intent="Develop a performance model for the local GPU",
        kb={},
        iteration=0,
        max_iterations=1,
        max_benchmarks=1,
    )
    elapsed = time.time() - start

    assert elapsed < 5.0
    assert "fallback used" in result.reason
    assert "timed out" in result.reason
    assert "fallback" in result.planner


def test_resilient_research_retries_once_after_timeout_then_succeeds():
    resilient = ResilientWorkflowBackend(primary=HeuristicWorkflowBackend(), fallback=HeuristicWorkflowBackend())
    calls: list[str] = []

    def _call_primary(method_name: str, *_args):
        calls.append(method_name)
        if len(calls) == 1:
            raise TimeoutError("Primary backend openai:gpt-5.4:research_context timed out after 17.0s")
        return ResearchDecision(
            reason="Collected local GPU facts.",
            request_summary="Measure local capabilities.",
            unanswered_questions=[],
            findings=[{"title": "GPU model", "summary": "Detected local GPU."}],
            proposed_dimensions=["dram_bandwidth"],
            planner="openai:gpt-5.4",
            raw_response="",
        )

    resilient._call_primary = _call_primary  # type: ignore[method-assign]

    result = resilient.research_context(
        intent="Develop a performance model for the local GPU",
        kb={},
        iteration=0,
        research_request={"request_summary": "Measure local capabilities."},
    )

    assert calls == ["research_context", "research_context"]
    assert result.planner == "openai:gpt-5.4"
    assert result.reason.startswith("Primary research succeeded after 1 timeout retry attempt(s).")
    assert result.findings


def test_resilient_research_falls_back_after_retry_exhausted():
    resilient = ResilientWorkflowBackend(primary=HeuristicWorkflowBackend(), fallback=HeuristicWorkflowBackend())
    calls: list[str] = []

    def _call_primary(method_name: str, *_args):
        calls.append(method_name)
        raise TimeoutError("Primary backend openai:gpt-5.4:research_context timed out after 17.0s")

    resilient._call_primary = _call_primary  # type: ignore[method-assign]

    result = resilient.research_context(
        intent="Develop a performance model for the local GPU",
        kb={},
        iteration=0,
        research_request={"request_summary": "Measure local capabilities."},
    )

    assert calls == ["research_context", "research_context"]
    assert "fallback used" in result.reason
    assert "timeout retry exhausted after 1 additional attempt(s)" in result.reason
    assert "fallback" in result.planner


def test_resilient_timeout_exhaustion_launches_diagnostic(monkeypatch):
    resilient = ResilientWorkflowBackend(primary=HeuristicWorkflowBackend(), fallback=HeuristicWorkflowBackend())
    launched: list[tuple[str, tuple]] = []

    def _call_primary(method_name: str, *_args):
        raise TimeoutError("Primary backend openai:gpt-5.4:research_context timed out after 30.0s")

    def _launch(primary, method_name, args, exc):  # noqa: ARG001
        launched.append((method_name, args))

    resilient._call_primary = _call_primary  # type: ignore[method-assign]
    monkeypatch.setattr("gpu_profiler.llm._launch_openai_timeout_diagnostic", _launch)

    result = resilient.research_context(
        intent="Develop a performance model for the local GPU",
        kb={},
        iteration=0,
        research_request={"request_summary": "Find the right roofline profiler metrics."},
    )

    assert "fallback used" in result.reason
    assert launched
    assert launched[0][0] == "research_context"


def test_resilient_openai_research_does_not_add_outer_retry(monkeypatch):
    resilient = ResilientWorkflowBackend(
        primary=OpenAIWorkflowBackend(model="gpt-5.4", request_timeout_sec=15.0),
        fallback=HeuristicWorkflowBackend(),
    )
    calls: list[str] = []
    launched: list[tuple[str, tuple]] = []

    def _call_primary(method_name: str, *_args):
        calls.append(method_name)
        raise TimeoutError("Primary backend openai:gpt-5.4:research_context timed out after 35.0s")

    def _launch(primary, method_name, args, exc):  # noqa: ARG001
        launched.append((method_name, args))

    resilient._call_primary = _call_primary  # type: ignore[method-assign]
    monkeypatch.setattr("gpu_profiler.llm._launch_openai_timeout_diagnostic", _launch)

    result = resilient.research_context(
        intent="Develop a performance model for the local GPU",
        kb={},
        iteration=0,
        research_request={"request_summary": "Measure local capabilities."},
    )

    assert calls == ["research_context"]
    assert "fallback used" in result.reason
    assert launched
    assert launched[0][0] == "research_context"


def test_resilient_analysis_retries_once_after_timeout_then_succeeds():
    resilient = ResilientWorkflowBackend(primary=HeuristicWorkflowBackend(), fallback=HeuristicWorkflowBackend())
    calls: list[str] = []

    def _call_primary(method_name: str, *_args):
        calls.append(method_name)
        if len(calls) == 1:
            raise TimeoutError("Primary backend openai:gpt-5.4:analyze_results timed out after 30.0s")
        return AnalysisDecision(
            summary="Validated the measured bandwidth baseline.",
            claims=[],
            covered_dimensions=["dram_bandwidth"],
            stop=False,
            reason="Applied the new benchmark evidence.",
            veto_next_plan=False,
            veto_reason="",
            required_observability=[],
            contract_amendments=[],
            planner="openai:gpt-5.4",
        )

    resilient._call_primary = _call_primary  # type: ignore[method-assign]

    result = resilient.analyze_results(
        intent="Develop a performance model for the local GPU",
        kb={},
        plan={},
        execution_results=[],
        iteration=0,
        max_iterations=1,
    )

    assert calls == ["analyze_results", "analyze_results"]
    assert result.planner == "openai:gpt-5.4"
    assert result.reason.startswith("Primary analysis succeeded after 1 timeout retry attempt(s).")
    assert result.covered_dimensions == ["dram_bandwidth"]


def test_resilient_backend_extends_research_and_analysis_timeouts():
    resilient = ResilientWorkflowBackend(
        primary=OpenAIWorkflowBackend(model="gpt-5.4", request_timeout_sec=15.0),
        fallback=HeuristicWorkflowBackend(),
    )

    assert resilient._primary_timeout_sec("research_context") >= RESEARCH_TIMEOUT_SEC
    assert resilient._primary_timeout_sec("analyze_results") >= ANALYSIS_TIMEOUT_SEC
    assert resilient._primary_timeout_sec("research_context") > resilient._primary_timeout_sec("propose_plan")


def test_diagnostic_environment_preserves_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "  sk-test-key  ")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    env = _diagnostic_environment(OpenAIWorkflowBackend(model="gpt-5.4"))

    assert env["OPENAI_API_KEY"] == "sk-test-key"
    assert env["OPENAI_BASE_URL"] == "https://example.test/v1"
    assert env["GPU_PROFILER_OPENAI_MODEL"] == "gpt-5.4"


def test_load_queries_merges_inline_and_file_queries(tmp_path):
    queries_file = tmp_path / "queries.txt"
    queries_file.write_text(
        "# comment\nWhat is the GPU memory bandwidth?\n\nWhat profiler metrics matter?\nWhat is the GPU memory bandwidth?\n",
        encoding="utf-8",
    )

    result = load_queries(
        ["How do I estimate launch overhead?", "What profiler metrics matter?"],
        str(queries_file),
    )

    assert result == [
        "How do I estimate launch overhead?",
        "What profiler metrics matter?",
        "What is the GPU memory bandwidth?",
    ]


def test_parse_timeout_values_supports_sensitivity_sweep():
    result = parse_timeout_values(30.0, ["20", "30", "20", "45"])

    assert result == [20.0, 30.0, 45.0]


def test_evaluate_research_result_marks_meaningful_response():
    decision = ResearchDecision(
        reason="Collected relevant roofline references.",
        request_summary="Find the best roofline references for local GPU modeling.",
        unanswered_questions=["What is the exact memory clock under load?"],
        findings=[
            {
                "title": "Roofline paper",
                "summary": "Defines the roofline model.",
                "relevance": "Baseline methodology.",
                "source_url": "https://example.com/roofline-paper",
            }
        ],
        proposed_dimensions=["dram_bandwidth"],
        planner="openai:gpt-5.4",
    )

    evaluation = evaluate_research_result(decision, min_findings=1, min_unique_sources=1)

    assert evaluation["meaningful"] is True
    assert evaluation["findings_count"] == 1
    assert evaluation["unique_source_count"] == 1


def test_summarize_attempts_reports_per_query_counts():
    summary = summarize_attempts(
        [
            {
                "query": "q1",
                "status": "ok",
                "elapsed_sec": 2.0,
                "evaluation": {"meaningful": True},
            },
            {
                "query": "q1",
                "status": "error",
                "elapsed_sec": 5.0,
            },
            {
                "query": "q2",
                "status": "ok",
                "elapsed_sec": 3.0,
                "evaluation": {"meaningful": False},
            },
        ]
    )

    assert summary["attempt_count"] == 3
    assert summary["ok_count"] == 2
    assert summary["meaningful_count"] == 1
    per_query = {item["query"]: item for item in summary["per_query"]}
    assert per_query["q1"]["attempts"] == 2
    assert per_query["q1"]["meaningful"] == 1
    assert per_query["q2"]["ok"] == 1
    per_timeout = {item["timeout_sec"]: item for item in summary["per_timeout"]}
    assert per_timeout[0.0]["attempts"] == 3


def test_timeout_diagnostic_payload_extracts_research_query():
    payload = _timeout_diagnostic_payload(
        "research_context",
        (
            "Develop a performance model for the local GPU",
            {},
            2,
            {
                "request_summary": "Find the right roofline profiler metrics.",
                "target_questions": ["Which metrics map to roofline?"],
            },
            "",
            6,
        ),
    )

    assert payload["intent"] == "Develop a performance model for the local GPU"
    assert payload["iteration"] == 2
    assert payload["query"] == "Find the right roofline profiler metrics."
    assert payload["max_sources"] == 6


def test_build_agent_fixture_creates_codegen_probe_inputs():
    fixture = build_agent_fixture(
        "generate_implementation",
        "Develop a performance model for the local GPU",
        "Measure sustained DRAM bandwidth with a bounded CUDA benchmark.",
        8,
    )

    assert fixture["intent"] == "Develop a performance model for the local GPU"
    assert fixture["plan"]["proposal"]["proposals"]
    assert fixture["proposal_memo"] == "Measure sustained DRAM bandwidth with a bounded CUDA benchmark."
    assert fixture["kb"]["target_dimensions"]


def test_evaluate_agent_result_marks_meaningful_codegen_output():
    class _Result:
        reason = "Generated one runnable benchmark."
        benchmarks = [{"command": "nvcc -O3 bench.cu -o bench && ./bench"}]

    evaluation = evaluate_agent_result("generate_implementation", _Result())

    assert evaluation["meaningful"] is True
    assert evaluation["runnable_benchmark_count"] == 1


def test_summarize_agent_attempts_groups_by_agent_and_timeout():
    summary = summarize_agent_attempts(
        [
            {
                "agent": "research_context",
                "query": "q1",
                "status": "ok",
                "elapsed_sec": 10.0,
                "request_timeout_sec": 30.0,
                "evaluation": {"meaningful": True},
            },
            {
                "agent": "research_context",
                "query": "q1",
                "status": "error",
                "elapsed_sec": 35.0,
                "request_timeout_sec": 45.0,
            },
            {
                "agent": "plan_proposal",
                "query": "q2",
                "status": "ok",
                "elapsed_sec": 6.0,
                "request_timeout_sec": 30.0,
                "evaluation": {"meaningful": True},
            },
        ]
    )

    assert summary["attempt_count"] == 3
    assert summary["ok_count"] == 2
    per_agent = {item["agent"]: item for item in summary["per_agent"]}
    assert per_agent["research_context"]["attempts"] == 2
    assert per_agent["plan_proposal"]["meaningful"] == 1


def test_compact_planner_kb_stays_within_budget():
    kb = {
        "intent": "Develop a performance model for the local GPU",
        "target_dimensions": [f"dim_{i}" for i in range(20)],
        "covered_dimensions": [f"dim_{i}" for i in range(10)],
        "coverage_score": 0.3,
        "target_coverage": 0.9,
        "available_tools": {"nvidia-smi": True, "ncu": True, "nsys": True, "python": True},
        "current_knowledge_model": {
            "focus_nodes": [f"node_{i}" for i in range(12)],
            "domain_hierarchy": [
                {
                    "id": f"node_{i}",
                    "name": f"Node {i}",
                    "description": "x" * 400,
                    "node_type": "feature",
                    "status": "unknown",
                    "open_gaps": ["gap one", "gap two", "gap three"],
                }
                for i in range(20)
            ],
        },
        "current_proposal": {
            "target_nodes": [f"node_{i}" for i in range(12)],
            "proposals": [
                {
                    "id": f"proposal_{i}",
                    "title": "t" * 200,
                    "objective": "o" * 400,
                    "target_node_ids": [f"node_{i}"],
                    "benchmark_role": "baseline",
                }
                for i in range(12)
            ],
        },
        "history": [{"iteration": i, "summary": "s" * 500, "coverage_score": 0.1, "claims_added": i} for i in range(6)],
        "research_history": [
            {
                "iteration": i,
                "request_summary": "r" * 300,
                "proposed_dimensions": [f"dim_{j}" for j in range(8)],
                "findings": [{"title": "t" * 120, "summary": "f" * 400, "source_url": "https://example.com"} for _ in range(6)],
            }
            for i in range(4)
        ],
        "pending_contract_amendments": [{"path": "p" * 80, "change": "c" * 300, "priority": "high"} for _ in range(5)],
    }
    compact = _compact_planner_kb(kb)
    payload = _enforce_payload_budget(
        {"intent": kb["intent"], "knowledge_base": compact, "iteration": 0, "max_iterations": 1, "max_benchmarks": 1},
        target_chars=6000,
        hard_cap_chars=PLANNER_INPUT_HARD_CAP_CHARS,
        trimmers=[],
    )
    assert len(json.dumps(payload)) <= PLANNER_INPUT_HARD_CAP_CHARS
    assert len(payload["knowledge_base"]["history"]) <= 2
    assert len(payload["knowledge_base"]["research_history"]) <= 2


def test_compact_codegen_payload_stays_within_budget():
    plan = {
        "iteration": 0,
        "planner": "planner",
        "proposal": {
            "intent_summary": "Develop a performance model for the local GPU",
            "proposal_summary": "summary" * 100,
            "target_nodes": ["feature_0", "feature_1", "feature_2"],
            "proposals": [
                {
                    "id": "proposal_0",
                    "title": "Baseline for dimension_1",
                    "objective": "Measure dimension_1 with a simple benchmark",
                    "target_node_ids": ["feature_0"],
                    "benchmark_role": "baseline",
                    "hypothesis": "h" * 300,
                    "required_evidence": ["e1", "e2", "e3"],
                    "rationale": "r" * 300,
                },
                {
                    "id": "proposal_1",
                    "title": "Sweep for dimension_2",
                    "objective": "Measure dimension_2 with another benchmark",
                    "target_node_ids": ["feature_1"],
                    "benchmark_role": "sweep",
                    "hypothesis": "h" * 300,
                    "required_evidence": ["e1", "e2", "e3"],
                    "rationale": "r" * 300,
                },
            ],
        },
        "knowledge_model": {
            "focus_nodes": ["feature_0", "feature_1"],
            "domain_hierarchy": [
                {"id": "feature_0", "name": "dimension_1", "description": "d" * 400, "status": "unknown", "open_gaps": ["g1"]},
                {"id": "feature_1", "name": "dimension_2", "description": "d" * 400, "status": "unknown", "open_gaps": ["g2"]},
            ],
        },
    }
    kb = {
        "intent": "Develop a performance model for the local GPU",
        "target_dimensions": ["feature_0", "feature_1"],
        "available_tools": {"nvidia-smi": True, "ncu": True, "nsys": True, "python": True},
        "current_knowledge_model": plan["knowledge_model"],
        "schema_contract": {"negotiation_policy": {"thresholds": {"utility_min": 0.5}, "weights": {"coverage_gain_score": 0.35}}},
    }
    payload = {
        "intent": kb["intent"],
        "knowledge_base": _compact_codegen_kb(kb, "dimension_1"),
        "plan": _compact_codegen_plan(plan, "dimension_1"),
        "iteration": 0,
        "dimension": "dimension_1",
    }
    payload = _enforce_payload_budget(
        payload,
        target_chars=8000,
        hard_cap_chars=CODEGEN_INPUT_HARD_CAP_CHARS,
        trimmers=[],
    )
    assert len(json.dumps(payload)) <= CODEGEN_INPUT_HARD_CAP_CHARS
    assert len(payload["plan"]["proposal"]["proposals"]) == 1


def test_benchmark_from_memo_extracts_cuda_file_and_command():
    memo = """1. Implementation Summary
A streaming-memory CUDA benchmark that measures achieved bandwidth for a large linear copy loop.

2. CUDA Source File
Path: generated/iter_00/dram_bandwidth.cu
```cuda
#include <cstdio>
int main() { std::puts("bandwidth_gbps=123.4"); return 0; }
```

3. Build and Run Command
```bash
nvcc -O3 generated/iter_00/dram_bandwidth.cu -o ./generated/iter_00/dram_bandwidth && ./generated/iter_00/dram_bandwidth
```

4. Validation Checks
- stdout prints bandwidth_gbps
- runtime stays under 45 seconds
- nvidia-smi shows active memory traffic

5. Feasibility and Risks
- Feasibility: feasible
- Complexity: low
- Risks: may need larger working sets on GPUs with large cache.
"""
    benchmark, amendments = _benchmark_from_memo(
        memo,
        intent="Develop a performance model for the local GPU",
        dimension="dram_bandwidth",
        iteration=0,
        benchmark_index=0,
        proposal={"proposals": [{"id": "proposal_0", "title": "Bandwidth baseline", "hypothesis": "Measure streaming bandwidth."}]},
    )

    assert benchmark["id"] == "proposal_0"
    assert benchmark["files"][0]["path"].endswith(".cu")
    assert "bandwidth_gbps" in benchmark["files"][0]["content"]
    assert "nvcc -O3" in benchmark["command"]
    assert benchmark["scores"]["implementability_score"] >= 0.8
    assert amendments == []


def test_benchmark_from_markdown_heading_memo_extracts_backticked_path():
    memo = """## 1. Implementation Summary
One CUDA roofline anchor benchmark.

## 2. CUDA Source File
Path: `bench/roofline_stream_fma.cu`
```cuda
#include <cstdio>
int main() { std::puts("ok"); return 0; }
```

## 3. Build and Run Command
```bash
mkdir -p bench && nvcc -O3 bench/roofline_stream_fma.cu -o bench/roofline_stream_fma && ./bench/roofline_stream_fma
```

## 4. Validation Checks
- stdout prints ok

## 5. Feasibility and Risks
- Feasibility: feasible
- Complexity: medium
"""
    benchmark, amendments = _benchmark_from_memo(
        memo,
        intent="Develop a performance model for the local GPU",
        dimension="roofline_anchor",
        iteration=0,
        benchmark_index=0,
        proposal={"proposals": [{"id": "proposal_0", "title": "Roofline anchor", "hypothesis": "Measure DRAM and FMA throughput."}]},
    )

    assert benchmark["files"][0]["path"] == "bench/roofline_stream_fma.cu"
    assert "roofline_stream_fma.cu" in benchmark["command"]
    assert amendments == []


def test_openai_codegen_can_parse_memo_without_json():
    backend = OpenAIWorkflowBackend(model="gpt-5.4", request_timeout_sec=1.0)

    def _memo(*_args, **_kwargs):
        return """1. Implementation Summary
A CUDA benchmark that reports sustained global-memory bandwidth from a simple streaming kernel.

2. CUDA Source File
Path: generated/iter_00/stream_bandwidth.cu
```cuda
#include <cstdio>
int main() { std::puts("bandwidth_gbps=200.0"); return 0; }
```

3. Build and Run Command
```bash
nvcc -O3 generated/iter_00/stream_bandwidth.cu -o ./generated/iter_00/stream_bandwidth && ./generated/iter_00/stream_bandwidth
```

4. Validation Checks
- stdout prints bandwidth_gbps
- binary exits with return code 0

5. Feasibility and Risks
- Feasibility: feasible
- Complexity: low
"""

    backend._text_completion = _memo  # type: ignore[method-assign]

    result = backend.generate_implementation(
        intent="Develop a performance model for the local GPU",
        kb={"target_dimensions": ["dram_bandwidth"], "schema_contract": {}},
        plan={
            "proposal": {
                "target_nodes": ["dram_bandwidth"],
                "proposals": [
                    {
                        "id": "proposal_0",
                        "title": "DRAM bandwidth baseline",
                        "objective": "Measure sustained DRAM bandwidth",
                        "target_node_ids": ["dram_bandwidth"],
                        "benchmark_role": "baseline",
                    }
                ],
            },
            "proposal_md_artifact": "",
        },
        iteration=0,
        max_benchmarks=1,
        proposal_memo="Measure a simple streaming bandwidth baseline.",
    )

    assert result.benchmarks
    assert result.benchmarks[0]["files"][0]["path"].endswith(".cu")
    assert "nvcc -O3" in result.benchmarks[0]["command"]
    assert "1. Implementation Summary" in result.raw_response


def test_codegen_truncated_cuda_block_is_rejected():
    memo = """1. Implementation Summary
Short summary.

2. CUDA Source File
Path: generated/iter_00/broken.cu
```cuda
int main() {

3. Build and Run Command
```bash
nvcc -O3 generated/iter_00/broken.cu -o ./generated/iter_00/broken && ./generated/iter_00/broken
```

4. Validation Checks
- return code 0

5. Feasibility and Risks
- Feasibility: feasible
- Complexity: low
"""
    benchmark, amendments = _benchmark_from_memo(
        memo,
        intent="Develop a performance model for the local GPU",
        dimension="dram_bandwidth",
        iteration=0,
        benchmark_index=0,
        proposal={"proposals": [{"id": "proposal_0", "title": "Bandwidth baseline", "hypothesis": "Measure streaming bandwidth."}]},
    )
    assert benchmark["command"] == ""
    assert benchmark["files"] == []
    assert any(item["path"] == "implementation.cuda_source" for item in amendments)


def test_generate_implementation_repairs_incomplete_first_memo():
    backend = OpenAIWorkflowBackend(model="gpt-5.4")
    responses = [
        """1. Implementation Summary
Implement one CUDA census benchmark that:

2. CUDA Source File
Path: generated/iter_00/broken.cu
```cuda
int main() {

3. Build and Run Command
```bash
nvcc -O3 generated/iter_00/broken.cu -o ./generated/iter_00/broken && ./generated/iter_00/broken
```

4. Validation Checks
- return code 0

5. Feasibility and Risks
- Feasibility: feasible
- Complexity: low
""",
        """1. Implementation Summary
Implement one CUDA census benchmark that prints the local GPU identity.

2. CUDA Source File
Path: generated/iter_00/gpu_identity.cu
```cuda
#include <cstdio>
int main() { std::puts("gpu_identity_ok"); return 0; }
```

3. Build and Run Command
```bash
nvcc -O3 generated/iter_00/gpu_identity.cu -o ./generated/iter_00/gpu_identity && ./generated/iter_00/gpu_identity
```

4. Validation Checks
- stdout prints gpu_identity_ok
- return code 0

5. Feasibility and Risks
- Feasibility: feasible
- Complexity: low
""",
    ]
    prompts: list[str] = []

    def _memo(*, user, **_kwargs):
        prompts.append(user)
        return responses.pop(0)

    backend._text_completion = _memo  # type: ignore[method-assign]

    result = backend.generate_implementation(
        intent="Develop a performance model for the local GPU",
        kb={"target_dimensions": ["gpu_identity"], "schema_contract": {}},
        plan={
            "proposal": {
                "target_nodes": ["gpu_identity"],
                "proposals": [
                    {
                        "id": "proposal_0",
                        "title": "GPU identity baseline",
                        "objective": "Print the local GPU identity",
                        "target_node_ids": ["gpu_identity"],
                        "benchmark_role": "baseline",
                    }
                ],
            },
            "proposal_md_artifact": "",
        },
        iteration=0,
        max_benchmarks=1,
        proposal_memo="Identify the local GPU with a minimal CUDA executable.",
    )

    assert len(prompts) == 2
    assert "could not be converted into a runnable benchmark" in prompts[1]
    assert result.benchmarks
    assert "gpu_identity.cu" in result.benchmarks[0]["files"][0]["path"]
    assert "[repair_attempt_used]" in result.reason
    assert "CODEGEN REPAIR ATTEMPT" in result.raw_response


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


def test_workload_runner_supports_shell_chaining(tmp_path):
    runner = WorkloadRunnerAgent()
    ctx = AgentContext(run_id="shell-test", run_dir=tmp_path)
    task = Task(
        id="shell-0",
        kind="run_workload",
        payload={"command": "python -c \"print('a')\" && python -c \"print('b')\""},
    )
    result = runner.run(task, ctx)
    assert result["returncode"] == 0
    assert "a" in result["stdout"]
    assert "b" in result["stdout"]


def test_workload_runner_respects_cwd_path(tmp_path):
    runner = WorkloadRunnerAgent()
    ctx = AgentContext(run_id="cwd-test", run_dir=tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "hello.txt").write_text("cwd-ok\n", encoding="utf-8")
    task = Task(
        id="cwd-0",
        kind="run_workload",
        payload={
            "command": "python -c \"from pathlib import Path; print(Path('hello.txt').read_text().strip())\"",
            "cwd_path": str(workdir),
        },
    )
    result = runner.run(task, ctx)
    assert result["returncode"] == 0
    assert result["cwd"] == str(workdir)
    assert "cwd-ok" in result["stdout"]


def test_workload_runner_retries_ncu_with_passwordless_sudo(tmp_path, monkeypatch):
    runner = WorkloadRunnerAgent()
    ctx = AgentContext(run_id="ncu-sudo-test", run_dir=tmp_path)
    calls: list[list[str]] = []

    class _Proc:
        def __init__(self, returncode: int, stdout: str, stderr: str = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(args, capture_output=True, text=True, cwd=None):  # noqa: ARG001
        calls.append(list(args))
        shell_command = args[-1]
        if shell_command.startswith("ncu --set full"):
            return _Proc(1, "==ERROR== ERR_NVGPUCTRPERM\n")
        if shell_command.startswith("sudo -n /usr/local/cuda/bin/ncu --set full"):
            return _Proc(0, "profile ok\n")
        raise AssertionError(f"unexpected subprocess args: {args}")

    monkeypatch.setattr("gpu_profiler.agents._find_passwordless_sudo_ncu_path", lambda: "/usr/local/cuda/bin/ncu")
    monkeypatch.setattr(subprocess, "run", _fake_run)

    task = Task(
        id="ncu-0",
        kind="run_workload",
        payload={"command": "ncu --set full ./benchmarks/dram_stream_roofline"},
    )
    result = runner.run(task, ctx)

    assert result["returncode"] == 0
    assert result["retried_with_sudo_ncu"] is True
    assert result["original_command"] == "ncu --set full ./benchmarks/dram_stream_roofline"
    assert result["command"].startswith("sudo -n /usr/local/cuda/bin/ncu --set full")
    assert len(calls) == 2


def test_extract_compile_preflight_stops_before_execution():
    command = (
        "mkdir -p benchmarks && nvcc -O3 benchmarks/dram_bandwidth.cu "
        "-o benchmarks/dram_bandwidth && ./benchmarks/dram_bandwidth"
    )
    preflight_command, expected_output = _extract_compile_preflight(command)

    assert preflight_command == "mkdir -p benchmarks && nvcc -O3 benchmarks/dram_bandwidth.cu -o benchmarks/dram_bandwidth"
    assert expected_output == "benchmarks/dram_bandwidth"


def test_preflight_single_benchmark_reports_compile_failure(tmp_path, monkeypatch):
    bench_dir = tmp_path / "proposal_0_0" / "benchmarks"
    bench_dir.mkdir(parents=True)
    (bench_dir / "dram_bandwidth.cu").write_text("int main(){return 0;}\n", encoding="utf-8")

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "compile failed: missing symbol\n"

    def _fake_run(*_args, **_kwargs):
        return _Proc()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    report = _preflight_single_benchmark(
        tmp_path,
        {
            "id": "proposal_0_0",
            "command": "mkdir -p benchmarks && nvcc -O3 benchmarks/dram_bandwidth.cu -o benchmarks/dram_bandwidth && ./benchmarks/dram_bandwidth",
            "files": [{"path": "benchmarks/dram_bandwidth.cu", "type": "cu", "content": "int main(){return 0;}\n"}],
        },
    )

    assert report["checked"] is True
    assert report["ok"] is False
    assert report["reason"] == "compile preflight failed"
    assert "compile failed" in report["stderr"]


class BrokenCompileBackend(HeuristicWorkflowBackend):
    def generate_implementation(
        self,
        intent: str,
        kb: dict,
        plan: dict,
        iteration: int,
        max_benchmarks: int,
        proposal_memo: str = "",
    ):
        _ = (intent, kb, plan, iteration, max_benchmarks, proposal_memo)
        return type("Decision", (), {
            "reason": "Generated one benchmark.",
            "benchmarks": [
                {
                    "id": "proposal_0_0",
                    "command": "mkdir -p benchmarks && nvcc -O3 benchmarks/dram_bandwidth.cu -o benchmarks/dram_bandwidth && ./benchmarks/dram_bandwidth",
                    "hypothesis": "Measure DRAM bandwidth.",
                    "dimensions": ["dram_bandwidth"],
                    "analysis_method": {"summary": "Check compile and runtime.", "metrics": ["returncode"]},
                    "scores": {
                        "coverage_gain_score": 0.8,
                        "implementability_score": 0.8,
                        "observability_score": 0.8,
                        "rationale": "test",
                    },
                    "files": [
                        {
                            "path": "benchmarks/dram_bandwidth.cu",
                            "type": "cu",
                            "content": "int main(){return 0;}\n",
                        }
                    ],
                }
            ],
            "negotiation": {"accepted": [], "rejected": [], "policy": {}},
            "contract_amendments": [],
            "planner": "broken-compile-backend",
            "raw_response": "",
        })()


def test_codegen_agent_rejects_preflight_failures_before_execution(tmp_path, monkeypatch):
    agent = LLMCodegenAgent(workflow_backend=BrokenCompileBackend())
    ctx = AgentContext(run_id="preflight-test", run_dir=tmp_path)
    task = Task(
        id="codegen-0",
        kind="llm_generate_implementation",
        payload={
            "intent": "Develop a performance model for the local GPU",
            "iteration": 0,
            "max_benchmarks": 1,
            "knowledge_base": {},
            "plan": {
                "proposal": {
                    "target_nodes": ["dram_bandwidth"],
                    "proposals": [
                        {
                            "id": "proposal_0_0",
                            "title": "DRAM benchmark",
                            "objective": "Compile and run one CUDA benchmark.",
                            "target_node_ids": ["dram_bandwidth"],
                            "benchmark_role": "baseline",
                        }
                    ],
                },
                "proposal_md_artifact": "",
            },
        },
    )

    def _fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "compile failed\n")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = agent.run(task, ctx)

    assert result["benchmarks"] == []
    assert result["rejected_benchmarks"]
    assert result["preflight_artifact"]


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


class TraceArtifactAgent(Agent):
    name = "trace-artifact-agent"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "llm_generate_implementation"

    def run(self, task: Task, ctx: AgentContext) -> dict:
        iter_dir = ctx.run_dir / "iterations" / "iter_00"
        iter_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = iter_dir / "implementation_prompt.md"
        raw_path = iter_dir / "implementation_raw.md"
        prompt_path.write_text("# Prompt\nPlan a bandwidth benchmark.", encoding="utf-8")
        raw_path.write_text("1. Implementation Summary\nA good benchmark memo.", encoding="utf-8")
        return {
            "planner": "trace-artifact-agent",
            "reason": "Generated implementation memo.",
            "benchmarks": [],
            "rejected_benchmarks": [],
            "prompt_artifact": str(prompt_path),
            "raw_artifact": str(raw_path),
        }


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


def test_orchestrator_emits_live_trace_with_prompt_and_raw_memo(tmp_path):
    trace_stream = io.StringIO()
    orch = OrchestratorHarness(
        agents=[CommunicationMonitorAgent(), TraceArtifactAgent()],
        retry_policy=RetryPolicy(max_retries=0, retry_delay_sec=0),
        trace_stream=trace_stream,
        conversation_stream=trace_stream,
    )

    tasks = orch.run_custom(
        stages=[[{"kind": "llm_generate_implementation", "payload": {"iteration": 0, "intent": "debug trace"}}]],
        out_dir=str(tmp_path / "run"),
    )
    trace = trace_stream.getvalue()

    assert tasks[0].status == "done"
    assert "[trace] START iter=00" in trace
    assert "[trace] DONE iter=00" in trace
    assert "Round #1, orchestrator -> trace-artifact-agent: message: Please turn the current proposal into an executable benchmark for iteration 0." in trace
    assert "Round #2, trace-artifact-agent -> orchestrator: message: I generated an implementation draft" in trace


def test_communication_monitor_writes_conversation_transcript(tmp_path):
    trace_stream = io.StringIO()
    run_dir = tmp_path / "run"
    orch = OrchestratorHarness(
        agents=[CommunicationMonitorAgent(), TraceArtifactAgent()],
        retry_policy=RetryPolicy(max_retries=0, retry_delay_sec=0),
        trace_stream=trace_stream,
    )

    tasks = orch.run_custom(
        stages=[[{"kind": "llm_generate_implementation", "payload": {"iteration": 0, "intent": "debug trace"}}]],
        out_dir=str(run_dir),
    )

    assert tasks[0].status == "done"
    global_transcript = (run_dir / "agent_conversation.md").read_text(encoding="utf-8")
    iter_transcript = (run_dir / "iterations" / "iter_00" / "conversation.md").read_text(encoding="utf-8")

    assert "## Turn 1" in global_transcript
    assert "orchestrator to trace-artifact-agent:" in global_transcript
    assert "## Turn 2" in global_transcript
    assert "trace-artifact-agent to orchestrator:" in global_transcript
    assert "Prompt I used:" in global_transcript
    assert "A good benchmark memo." in global_transcript
    assert "iter 00" in iter_transcript


def test_live_conversation_stream_remains_active_without_raw_trace(tmp_path):
    conversation_stream = io.StringIO()
    orch = OrchestratorHarness(
        agents=[CommunicationMonitorAgent(), TraceArtifactAgent()],
        retry_policy=RetryPolicy(max_retries=0, retry_delay_sec=0),
        emit_live_trace=False,
        emit_live_conversation=True,
        conversation_stream=conversation_stream,
    )

    tasks = orch.run_custom(
        stages=[[{"kind": "llm_generate_implementation", "payload": {"iteration": 0, "intent": "debug trace"}}]],
        out_dir=str(tmp_path / "run"),
    )
    conversation = conversation_stream.getvalue()

    assert tasks[0].status == "done"
    assert "[trace]" not in conversation
    assert "Round #1, orchestrator -> trace-artifact-agent: message: Please turn the current proposal into an executable benchmark for iteration 0." in conversation
    assert "Round #2, trace-artifact-agent -> orchestrator: message: I generated an implementation draft" in conversation
