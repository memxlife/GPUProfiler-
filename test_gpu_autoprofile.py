import json
import shutil
import subprocess
import sys
from pathlib import Path

from gpu_profiler.agents import _render_research_md
from gpu_profiler.knowledge_base import (
    initialize_markdown_knowledge_base,
    load_markdown_knowledge_base_memos,
    update_markdown_knowledge_base,
)
from gpu_profiler.llm import _compact_codegen_kb, _compact_codegen_plan, _next_frontier_question, _render_codegen_prompt


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
    assert (run_dir / "run_state.json").exists()
    assert (run_dir / "knowledge_base" / "README.md").exists()
    assert (run_dir / "knowledge_base" / "frontier.md").exists()
    assert (run_dir / "knowledge_base" / "local_findings.md").exists()
    assert (run_dir / "iterations" / "iter_00" / "research.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "research.md").exists()
    assert (run_dir / "iterations" / "iter_00" / "implementation.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "implementation.md").exists()
    assert (run_dir / "iterations" / "iter_00" / "analysis_update.json").exists()
    assert (run_dir / "iterations" / "iter_00" / "analysis.md").exists()

    assert not (run_dir / "iterations" / "iter_00" / "proposal.json").exists()
    assert not (run_dir / "iterations" / "iter_00" / "proposal.md").exists()
    assert not (run_dir / "iterations" / "iter_00" / "research_request.json").exists()
    assert not (run_dir / "iterations" / "iter_00" / "research_request.md").exists()

    run_log = json.loads((run_dir / "run_log.json").read_text(encoding="utf-8"))
    kinds = [item["kind"] for item in run_log]
    assert "llm_plan_research" in kinds
    assert "llm_research" in kinds
    assert "llm_plan_proposal" in kinds
    assert "llm_generate_implementation" in kinds
    assert "execute_implementation" in kinds
    assert "llm_analyze_update" in kinds


def test_frontier_question_prefers_unmet_criteria(tmp_path):
    kb_files = initialize_markdown_knowledge_base(tmp_path, "test intent")
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
                    }
                ],
            },
        )
    )
    kb.update(load_markdown_knowledge_base_memos(kb))

    question = _next_frontier_question(kb, ["memory"])
    assert "2.1 Global Memory Access" in question
    assert "frontier criterion" in question.lower()
    assert "documented clearly enough for reuse" in question


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
        "proposal": {
            "target_nodes": ["dram_bandwidth"],
            "proposals": [
                {
                    "id": "proposal_0",
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
        "proposal_memo": "# Planning Context\n\n## Current Question\nNeed reusable documented benchmark conditions.\n",
    }

    prompt = _render_codegen_prompt(payload)
    assert "Current frontier question:" in prompt
    assert "documented clearly enough for reuse" in prompt
    assert "Planning context:" in prompt
