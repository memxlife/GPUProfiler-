import shutil
import subprocess
import sys
import os
from pathlib import Path

from gpu_profiler.core.models import AgentContext, Task
from gpu_profiler.core.store import read_data_artifact, write_data_artifact
from gpu_profiler.runtime.agents import BookBuilderAgent, _render_research_md
from gpu_profiler.knowledge.knowledge_base import (
    answer_question,
    consolidate_markdown_knowledge_base,
    initialize_markdown_knowledge_base,
    load_markdown_knowledge_base_memos,
    update_markdown_knowledge_base,
    update_question_context,
)
from gpu_profiler.workflow.llm import (
    BookBuildDecision,
    HeuristicWorkflowBackend,
    ResilientWorkflowBackend,
    _compact_codegen_kb,
    _compact_codegen_plan,
    _next_frontier_question,
    _render_codegen_prompt,
    _heuristic_initial_book_markdown,
)


def run_cli(tmp_path: Path, *args: str) -> dict:
    cmd = [sys.executable, "gpu_autoprofile.py", *args]
    env = dict(os.environ)
    env["GPU_PROFILER_ENABLE_INIT_QUESTION_SEED"] = "0"
    env["GPU_PROFILER_ENABLE_LIVE_INIT_BOOK_BUILDER"] = "0"
    subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, check=True, env=env)
    out_dir = "profiling_runs"
    for index, arg in enumerate(args):
        if arg == "--out" and index + 1 < len(args):
            out_dir = args[index + 1]
            break
    latest_run = sorted((tmp_path / out_dir).glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return read_data_artifact(latest_run / "final_result.md", {})


def copy_app(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    (tmp_path / "gpu_autoprofile.py").write_text((root / "gpu_autoprofile.py").read_text(encoding="utf-8"), encoding="utf-8")
    shutil.copytree(root / "gpu_profiler", tmp_path / "gpu_profiler")


def test_autonomous_run_produces_kb_centric_artifacts(tmp_path):
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
    assert (run_dir / "run_state.md").exists()
    assert (run_dir / "knowledge_base" / "knowledge_book.md").exists()
    assert (run_dir / "knowledge_base" / "frontier.md").exists()
    assert (run_dir / "knowledge_base" / "local_findings.md").exists()
    assert (run_dir / "iterations" / "iter_00" / "research.md").exists()
    assert (run_dir / "iterations" / "iter_00" / "implementation.md").exists()
    assert (run_dir / "iterations" / "iter_00" / "analysis.md").exists()
    assert (run_dir / "iterations" / "iter_00" / "book_builder.md").exists()
    assert (run_dir / "iterations" / "iter_00" / "feasibility_report.md").exists()
    assert (run_dir / "iterations" / "iter_00" / "implementation_preflight.md").exists()

    assert not (run_dir / "iterations" / "iter_00" / "proposal.json").exists()
    assert not (run_dir / "iterations" / "iter_00" / "proposal.md").exists()
    assert not (run_dir / "iterations" / "iter_00" / "research_request.json").exists()
    assert not (run_dir / "iterations" / "iter_00" / "research_request.md").exists()
    assert not list(run_dir.rglob("*.json"))

    run_log = (run_dir / "run_log.md").read_text(encoding="utf-8")
    assert "kind: `llm_plan_research`" in run_log
    assert "kind: `llm_research`" in run_log
    assert "kind: `llm_plan_benchmark`" in run_log
    assert "kind: `llm_generate_implementation`" in run_log
    assert "kind: `execute_implementation`" in run_log
    assert "kind: `llm_analyze_update`" in run_log


def test_init_only_run_authors_book_from_intent(tmp_path):
    copy_app(tmp_path)

    intent = """# Initial Intent

Build a profiling-driven book of GPU microarchitecture behavior focused on execution pipelines, warp scheduling, memory hierarchy, occupancy limits, and synchronization costs.
"""
    result = run_cli(
        tmp_path,
        "autonomous",
        "--intent",
        intent,
        "--out",
        "runs_init",
        "--samples",
        "1",
        "--interval",
        "0",
        "--max-iterations",
        "0",
        "--max-benchmarks",
        "1",
    )

    run_dir = tmp_path / result["run_dir"]
    book_text = (run_dir / "knowledge_base" / "knowledge_book.md").read_text(encoding="utf-8")
    run_state = read_data_artifact(run_dir / "run_state.md", {})

    assert "Chapter 2. Execution Pipelines and Warp Scheduling" in book_text
    assert "Chapter 1. Foundations" not in book_text
    assert "Which instruction classes saturate distinct execution pipelines" in book_text
    assert not str(run_state.get("intent", "")).startswith("# Initial Intent")


def test_book_builder_initialize_generates_section_questions(monkeypatch, tmp_path):
    kb_files = initialize_markdown_knowledge_base(tmp_path, "test intent")
    kb = {"intent": "test intent", **kb_files}
    kb_path = tmp_path / "run_state.md"
    write_data_artifact(kb_path, kb, title="Run State")

    class RecordingBackend:
        def build_book(
            self,
            intent: str,
            kb: dict,
            iteration: int,
            book_markdown: str,
            mode: str,
            proposed_questions=None,
            initial_question_seed: str = "",
        ) -> BookBuildDecision:
            return BookBuildDecision(
                reason="recorded structure pass",
                book_markdown=(
                    "# GPU Architecture Knowledge Book\n\n"
                    "## Chapter 1. Test\n\n"
                    "### 1.1 Example Section\n\n"
                    "Summary\n"
                    "This section studies a simple mechanism.\n\n"
                    "Mechanism\n"
                    "The mechanism depends on scheduler readiness and arithmetic throughput.\n\n"
                    "Evidence\n"
                    "- Initial intent only.\n\n"
                    "Current Understanding\n"
                    "A structure pass should create the section before questions are added.\n\n"
                    "Uncertainty\n"
                    "The exact quantitative behavior is still unknown.\n\n"
                    "Questions\n"
                ),
                planner="recording-backend",
                raw_response="",
            )

    monkeypatch.setattr(
        "gpu_profiler.runtime.agents.generate_book_builder_section_questions",
        lambda **kwargs: (
            "- Question: Which instruction classes saturate distinct pipelines?\n"
            "  Why It Matters: It defines the first compute ceiling.\n"
            "  Context: The section focuses on arithmetic throughput.\n"
        ),
    )
    monkeypatch.setattr("gpu_profiler.runtime.agents.openai_completion_available", lambda: True)

    backend = RecordingBackend()
    agent = BookBuilderAgent(workflow_backend=backend)
    result = agent.run(
        Task(
            id="book-init",
            kind="book_build_update",
            payload={"intent": "test intent", "iteration": 0, "kb_path": str(kb_path), "mode": "initialize"},
        ),
        AgentContext(run_id="test-run", run_dir=tmp_path),
    )

    question_path = tmp_path / "iterations" / "iter_00" / "book_builder_questions.md"
    summary_path = tmp_path / "iterations" / "iter_00" / "book_builder.md"
    book_text = Path(kb_files["knowledge_base_book_artifact"]).read_text(encoding="utf-8")

    assert question_path.exists()
    assert "Which instruction classes saturate distinct pipelines" in question_path.read_text(encoding="utf-8")
    assert "Which instruction classes saturate distinct pipelines" in book_text
    assert result["initial_question_seed_artifact"] == str(question_path)
    assert result["initial_question_seed_status"] == "completed"
    assert "book_builder_questions.md" in summary_path.read_text(encoding="utf-8")


def test_book_builder_initialize_prefers_openai_authoring_for_structure_pass(monkeypatch, tmp_path):
    kb_files = initialize_markdown_knowledge_base(tmp_path, "test intent")
    kb = {"intent": "test intent", **kb_files}
    kb_path = tmp_path / "run_state.md"
    write_data_artifact(kb_path, kb, title="Run State")

    monkeypatch.setattr("gpu_profiler.runtime.agents.openai_completion_available", lambda: True)
    monkeypatch.setattr(
        "gpu_profiler.runtime.agents.generate_book_builder_section_questions",
        lambda **kwargs: "",
    )

    captured = {}

    class FakeOpenAIBackend:
        def __init__(self, model: str = "gpt-5.4") -> None:
            self.name = f"fake-openai:{model}"

        def build_book(
            self,
            intent: str,
            kb: dict,
            iteration: int,
            book_markdown: str,
            mode: str,
            proposed_questions=None,
            initial_question_seed: str = "",
        ) -> BookBuildDecision:
            captured["planner"] = self.name
            captured["mode"] = mode
            return BookBuildDecision(
                reason="fake openai build",
                book_markdown=(
                    "# GPU Architecture Knowledge Book\n\n"
                    "## Chapter 1. Test\n\n"
                    "### 1.1 Example Section\n\n"
                    "Summary\nOne summary.\n\nMechanism\nOne mechanism.\n\nEvidence\n- Initial intent.\n\n"
                    "Current Understanding\nOne understanding.\n\nUncertainty\nOne uncertainty.\n\nQuestions\n"
                ),
                planner=self.name,
                raw_response="",
            )

    class FakeResilientBackend:
        def __init__(self, primary, fallback) -> None:
            self.primary = primary
            self.fallback = fallback

        def build_book(self, *args, **kwargs):
            return self.primary.build_book(*args, **kwargs)

    monkeypatch.setattr("gpu_profiler.runtime.agents.OpenAIWorkflowBackend", FakeOpenAIBackend)
    monkeypatch.setattr("gpu_profiler.runtime.agents.ResilientWorkflowBackend", FakeResilientBackend)

    agent = BookBuilderAgent(workflow_backend=HeuristicWorkflowBackend())
    result = agent.run(
        Task(
            id="book-init",
            kind="book_build_update",
            payload={"intent": "test intent", "iteration": 0, "kb_path": str(kb_path), "mode": "initialize"},
        ),
        AgentContext(run_id="test-run", run_dir=tmp_path),
    )

    assert captured["planner"].startswith("fake-openai:")
    assert captured["mode"] == "initialize_structure"
    assert result["planner"].startswith("fake-openai:")


def test_heuristic_initial_book_authors_real_structure():
    text = _heuristic_initial_book_markdown("test intent")

    assert "# GPU Architecture Knowledge Book" in text
    assert "Chapter 2. Execution Pipelines and Warp Scheduling" in text


def test_resilient_workflow_backend_build_book_falls_back_cleanly():
    class BrokenPrimary:
        name = "broken-primary"

        def build_book(self, *args, **kwargs):
            raise RuntimeError("boom")

    backend = ResilientWorkflowBackend(primary=BrokenPrimary(), fallback=HeuristicWorkflowBackend())
    decision = backend.build_book(
        intent="test intent",
        kb={},
        iteration=0,
        book_markdown="",
        mode="initialize",
        proposed_questions=[],
        initial_question_seed="",
    )

    assert decision.planner.startswith("broken-primary->fallback:")
    assert "Primary book building failed" in decision.reason
    assert "# GPU Architecture Knowledge Book" in decision.book_markdown


def test_frontier_question_tracks_next_unresolved_book_question(tmp_path):
    kb_files = initialize_markdown_knowledge_base(tmp_path, "test intent")
    book_path = Path(kb_files["knowledge_base_book_artifact"])
    backend = HeuristicWorkflowBackend()
    book_path.write_text(
        backend.build_book(
            intent="test intent",
            kb={},
            iteration=0,
            book_markdown=book_path.read_text(encoding="utf-8"),
            mode="initialize",
            proposed_questions=[],
        ).book_markdown,
        encoding="utf-8",
    )
    kb = dict(kb_files)
    kb.update(
        {
            "intent": "test intent",
            "covered_dimensions": ["dram_bandwidth", "global_memory"],
            "coverage_score": 0.25,
            "latest_research": {},
        }
    )
    knowledge_model = {"domain_hierarchy": [], "focus_nodes": ["memory"]}

    kb.update(
        update_markdown_knowledge_base(
            tmp_path,
            intent="test intent",
            kb=kb,
            knowledge_model=knowledge_model,
            iteration=1,
            analysis={
                "summary": "Measured sequential global-memory throughput.",
                "covered_dimensions": ["dram_bandwidth", "global_memory"],
                "required_observability": [],
                "claims": [
                    {
                        "claim": "Sequential global-memory accesses sustain roughly 640 GB/s in the bounded benchmark.",
                        "dimensions": ["dram_bandwidth", "global_memory"],
                        "confidence": "medium",
                        "evidence": {"analysis_artifact": "iterations/iter_01/analysis.md"},
                    }
                ],
            },
        )
    )
    kb.update(load_markdown_knowledge_base_memos(kb))

    question = _next_frontier_question(kb, ["memory"])
    assert "Which local profiling measurements are necessary" in question


def test_question_context_answer_and_consolidation_update_book(tmp_path):
    kb_files = initialize_markdown_knowledge_base(tmp_path, "test intent")
    book_path = Path(kb_files["knowledge_base_book_artifact"])
    backend = HeuristicWorkflowBackend()
    book_path.write_text(
        backend.build_book(
            intent="test intent",
            kb={},
            iteration=0,
            book_markdown=book_path.read_text(encoding="utf-8"),
            mode="initialize",
            proposed_questions=[],
        ).book_markdown,
        encoding="utf-8",
    )
    kb = {"intent": "test intent", **kb_files}

    update_question_context(
        kb,
        question_text="What sustained global-memory bandwidth and effective latency can the local GPU deliver under simple, well-controlled access patterns?",
        context="Use a streaming benchmark with minimal arithmetic and clearly documented launch conditions.",
    )
    answer_question(
        kb,
        question_text="What sustained global-memory bandwidth and effective latency can the local GPU deliver under simple, well-controlled access patterns?",
        answer="A bounded sequential-load benchmark reached about 640 GB/s, suggesting the observed throughput is primarily limited by sustained DRAM transfer rather than arithmetic work.",
        evidence=["iterations/iter_00/execution.md", "iterations/iter_00/analysis.md"],
        resolved=True,
    )
    consolidate_markdown_knowledge_base(tmp_path, intent="test intent", kb=kb, iteration=0, proposed_questions=[])

    memory_text = (tmp_path / "knowledge_base" / "knowledge_book.md").read_text(encoding="utf-8")
    assert "640 GB/s" in memory_text
    assert "streaming benchmark with minimal arithmetic" in memory_text
    assert "What sustained DRAM bandwidth can this GPU deliver" not in memory_text


def test_research_markdown_records_current_question():
    rendered = _render_research_md(
        {
            "iteration": 0,
            "planner": "heuristic",
            "reason": "Need external context before code generation.",
            "current_question": "What evidence is still needed to document the benchmark conditions clearly enough for reuse?",
            "request_summary": "Gather external context for documenting benchmark conditions.",
            "proposed_dimensions": ["dram_bandwidth"],
            "unanswered_questions": ["What launch parameters are typically reported?"],
            "findings": [],
        }
    )

    assert "## Current Question" in rendered
    assert "document the benchmark conditions clearly enough for reuse" in rendered


def test_codegen_prompt_uses_current_question_as_primary_context():
    kb = {
        "intent": "Develop a performance model for the local GPU",
        "available_tools": {"python": True},
        "target_dimensions": ["dram_bandwidth"],
        "current_knowledge_model": {
            "domain_hierarchy": [
                {
                    "id": "dram_bandwidth",
                    "name": "dram_bandwidth",
                    "description": "Measure sustained global-memory bandwidth.",
                    "status": "frontier",
                    "open_gaps": ["Need reusable benchmark conditions."],
                }
            ]
        },
    }
    plan = {
        "iteration": 0,
        "planner": "heuristic",
        "current_question": "For 2.1 Global Memory Access, what evidence is still needed to satisfy this frontier criterion: The benchmark conditions are documented clearly enough for reuse?",
        "benchmark_plan": {
            "target_nodes": ["dram_bandwidth"],
            "benchmarks": [
                {
                    "id": "benchmark_plan_0",
                    "title": "Baseline benchmark for dram_bandwidth",
                    "objective": "Measure sustained global-memory bandwidth.",
                    "target_node_ids": ["dram_bandwidth"],
                    "benchmark_role": "baseline",
                    "required_evidence": ["successful run", "timing artifact"],
                    "rationale": "Need a bounded baseline before more complex memory experiments.",
                }
            ],
        },
    }
    payload = {
        "intent": kb["intent"],
        "iteration": 0,
        "dimension": "dram_bandwidth",
        "knowledge_base": _compact_codegen_kb(kb, "dram_bandwidth"),
        "plan": _compact_codegen_plan(plan, "dram_bandwidth"),
        "planning_memo": "# Planning Context\n\n## Current Question\nNeed reusable documented benchmark conditions.\n",
    }

    prompt = _render_codegen_prompt(payload)
    assert "Current selected question:" in prompt
    assert "documented clearly enough for reuse" in prompt
    assert "Planning context:" in prompt
