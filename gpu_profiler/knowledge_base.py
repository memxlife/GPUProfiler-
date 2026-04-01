import time
from pathlib import Path
from typing import Any

from .store import write_text


def initialize_markdown_knowledge_base(run_dir: Path, intent: str) -> dict[str, str]:
    kb_dir = run_dir / "knowledge_base"
    updates_dir = kb_dir / "updates"
    kb_dir.mkdir(parents=True, exist_ok=True)
    updates_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "knowledge_base_dir": str(kb_dir),
        "knowledge_base_index_artifact": str(kb_dir / "README.md"),
        "knowledge_base_frontier_artifact": str(kb_dir / "frontier.md"),
        "knowledge_base_findings_artifact": str(kb_dir / "local_findings.md"),
        "knowledge_base_foundations_artifact": str(kb_dir / "01_foundations.md"),
        "knowledge_base_memory_artifact": str(kb_dir / "02_memory_system.md"),
        "knowledge_base_resources_artifact": str(kb_dir / "03_resource_constraints.md"),
        "knowledge_base_modeling_artifact": str(kb_dir / "04_performance_modeling.md"),
    }

    _write_if_missing(Path(files["knowledge_base_foundations_artifact"]), _foundations_chapter())
    _write_if_missing(Path(files["knowledge_base_memory_artifact"]), _memory_chapter())
    _write_if_missing(Path(files["knowledge_base_resources_artifact"]), _resources_chapter())
    _write_if_missing(Path(files["knowledge_base_modeling_artifact"]), _modeling_chapter())
    _write_if_missing(Path(files["knowledge_base_findings_artifact"]), _initial_findings_md())
    write_text(Path(files["knowledge_base_index_artifact"]), _render_kb_index(intent=intent, files=files))
    write_text(Path(files["knowledge_base_frontier_artifact"]), _render_frontier_md(intent=intent, frontier_questions=[]))
    return files


def update_markdown_knowledge_base(
    run_dir: Path,
    *,
    intent: str,
    kb: dict[str, Any],
    knowledge_model: dict[str, Any],
    iteration: int,
    analysis: dict[str, Any],
) -> dict[str, str]:
    files = initialize_markdown_knowledge_base(run_dir, intent)
    frontier_questions = _frontier_questions(kb=kb, knowledge_model=knowledge_model, analysis=analysis)
    write_text(
        Path(files["knowledge_base_index_artifact"]),
        _render_kb_index(intent=intent, files=files, kb=kb, knowledge_model=knowledge_model),
    )
    write_text(
        Path(files["knowledge_base_frontier_artifact"]),
        _render_frontier_md(
            intent=intent,
            frontier_questions=frontier_questions,
            covered_dimensions=kb.get("covered_dimensions", []),
            coverage_score=kb.get("coverage_score", 0.0),
        ),
    )
    update_path = Path(files["knowledge_base_dir"]) / "updates" / f"iter_{iteration:02d}.md"
    write_text(update_path, _render_iteration_update_md(iteration=iteration, analysis=analysis))
    write_text(
        Path(files["knowledge_base_findings_artifact"]),
        _render_local_findings_md(kb=kb),
    )
    files["knowledge_base_latest_update_artifact"] = str(update_path)
    return files


def load_markdown_knowledge_base_memos(kb: dict[str, Any]) -> dict[str, str]:
    index_path = str(kb.get("knowledge_base_index_artifact", "")).strip()
    frontier_path = str(kb.get("knowledge_base_frontier_artifact", "")).strip()
    return {
        "knowledge_base_book_memo": _read_text(index_path),
        "knowledge_base_frontier_memo": _read_text(frontier_path),
    }


def _write_if_missing(path: Path, text: str) -> None:
    if not path.exists():
        write_text(path, text)


def _read_text(path_text: str) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _render_kb_index(
    *,
    intent: str,
    files: dict[str, str],
    kb: dict[str, Any] | None = None,
    knowledge_model: dict[str, Any] | None = None,
) -> str:
    covered = kb.get("covered_dimensions", []) if isinstance(kb, dict) else []
    coverage_score = kb.get("coverage_score", 0.0) if isinstance(kb, dict) else 0.0
    focus_nodes = knowledge_model.get("focus_nodes", []) if isinstance(knowledge_model, dict) else []
    lines = [
        "# GPU Architecture Knowledge Base",
        "",
        "## Intent",
        intent,
        "",
        "## Book Structure",
        "- [01 Foundations](01_foundations.md)",
        "- [02 Memory System](02_memory_system.md)",
        "- [03 Resource Constraints](03_resource_constraints.md)",
        "- [04 Performance Modeling](04_performance_modeling.md)",
        "- [Frontier](frontier.md)",
        "- [Local Findings](local_findings.md)",
    ]
    if covered:
        lines.extend(
            [
                "",
                "## Current Run Snapshot",
                f"- coverage_score: `{coverage_score}`",
                f"- covered_dimensions: `{covered}`",
                f"- focus_nodes: `{focus_nodes}`",
            ]
        )
    return "\n".join(lines)


def _render_frontier_md(
    *,
    intent: str,
    frontier_questions: list[str],
    covered_dimensions: list[Any] | None = None,
    coverage_score: float | int | None = None,
) -> str:
    lines = [
        "# Frontier",
        "",
        "## Intent",
        intent,
    ]
    if covered_dimensions is not None:
        lines.extend(
            [
                "",
                "## Current Coverage",
                f"- coverage_score: `{coverage_score}`",
                f"- covered_dimensions: `{covered_dimensions}`",
            ]
        )
    lines.extend(["", "## Active Frontier Questions"])
    if frontier_questions:
        for index, question in enumerate(frontier_questions, start=1):
            lines.append(f"{index}. {question}")
    else:
        lines.append("1. No explicit frontier question has been recorded yet.")
    return "\n".join(lines)


def _render_iteration_update_md(*, iteration: int, analysis: dict[str, Any]) -> str:
    lines = [
        f"# Iteration {iteration} Knowledge Update",
        "",
        "## Summary",
        str(analysis.get("summary", "")).strip() or "No analyzer summary recorded.",
        "",
        "## Covered Dimensions",
    ]
    covered = [str(x).strip() for x in analysis.get("covered_dimensions", []) if str(x).strip()]
    if covered:
        for item in covered:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.extend(["", "## Claims"])
    claims = analysis.get("claims", []) if isinstance(analysis.get("claims", []), list) else []
    if claims:
        for index, item in enumerate(claims, start=1):
            lines.extend(
                [
                    f"### Claim {index}",
                    f"- claim: {item.get('claim', '')}",
                    f"- confidence: `{item.get('confidence', '')}`",
                    f"- dimensions: `{item.get('dimensions', [])}`",
                    "",
                ]
            )
    else:
        lines.append("No claims recorded.")
    return "\n".join(lines)


def _render_local_findings_md(*, kb: dict[str, Any]) -> str:
    lines = [
        "# Local Findings",
        "",
        "## Iteration History",
    ]
    history = kb.get("history", []) if isinstance(kb.get("history", []), list) else []
    if history:
        for item in history:
            lines.append(
                f"- iter {item.get('iteration')}: coverage={item.get('coverage_score')}, claims_added={item.get('claims_added')}, summary={item.get('summary', '')}"
            )
    else:
        lines.append("- No completed analysis iterations yet.")
    return "\n".join(lines)


def _frontier_questions(kb: dict[str, Any], knowledge_model: dict[str, Any], analysis: dict[str, Any]) -> list[str]:
    questions: list[str] = []
    latest_research = kb.get("latest_research", {}) if isinstance(kb.get("latest_research", {}), dict) else {}
    for item in latest_research.get("unanswered_questions", []):
        text = str(item).strip()
        if text and text not in questions:
            questions.append(text)
    for item in analysis.get("required_observability", []):
        text = str(item).strip()
        if text and text not in questions:
            questions.append(text)
    hierarchy = knowledge_model.get("domain_hierarchy", []) if isinstance(knowledge_model.get("domain_hierarchy", []), list) else []
    for node in hierarchy:
        if not isinstance(node, dict):
            continue
        name = str(node.get("name", node.get("id", ""))).strip()
        for gap in node.get("open_gaps", []):
            text = str(gap).strip()
            if not text:
                continue
            question = f"{name}: {text}" if name else text
            if question not in questions:
                questions.append(question)
    return questions[:12]


def _initial_findings_md() -> str:
    return "# Local Findings\n\n## Iteration History\n- No completed analysis iterations yet.\n"


def _foundations_chapter() -> str:
    return """# Part I. Foundations

## Chapter 1. Execution and Throughput Model

#### 1.1 Threads, Warps, and Thread Blocks

Summary  
The warp is the central execution unit for many performance-sensitive GPU behaviors.

Mechanism  
Thread blocks map onto streaming multiprocessors, and warps are the units exposed to warp scheduling and many stall-hiding effects.

Quantitative Understanding  
No trusted local quantitative characterization has yet been established for this section.

Evidence  
- No accepted local evidence yet

Open Questions  
- Which warp-level effects dominate throughput for simple kernels on this GPU?

Cross References  
- [1.2 Warp Scheduling]
- [2.1 Global Memory Access]

Status  
Frontier
"""


def _memory_chapter() -> str:
    return """# Part II. Memory System

## Chapter 2. Global Memory, Cache, and Bandwidth

#### 2.1 Global Memory Access

Summary  
Global memory is the primary backing store for large working sets and a major limiter for throughput-sensitive kernels.

Mechanism  
Effective throughput depends on transaction efficiency, caching, and the ability to overlap memory latency with useful work.

Quantitative Understanding  
No trusted local DRAM bandwidth estimate has yet been established in this chapter.

Evidence  
- No accepted local evidence yet

Open Questions  
- What sustained DRAM bandwidth can this GPU deliver under a simple sequential-load benchmark?

Cross References  
- [2.2 Coalescing]
- [4.1 Roofline Interpretation]

Status  
Frontier

#### 2.2 Coalescing

Summary  
Coalescing strongly affects effective global-memory throughput.

Mechanism  
Accesses that align well across a warp usually require fewer transactions and therefore achieve higher throughput.

Quantitative Understanding  
The stride-to-bandwidth relationship for this GPU is not yet locally characterized.

Evidence  
- No accepted local evidence yet

Open Questions  
- How does achieved bandwidth vary with stride on this GPU?

Cross References  
- [2.1 Global Memory Access]
- [2.3 L2 Cache]

Status  
Frontier

#### 2.3 L2 Cache

Summary  
The L2 cache mediates traffic between kernels and backing memory and can materially change effective access cost.

Mechanism  
When reuse falls within the effective L2 regime, traffic pressure on DRAM can be reduced substantially.

Quantitative Understanding  
The effective L2-resident regime for this GPU is not yet locally characterized.

Evidence  
- No accepted local evidence yet

Open Questions  
- At what working-set size does behavior transition beyond effective L2 reuse?

Cross References  
- [2.1 Global Memory Access]
- [4.1 Roofline Interpretation]

Status  
Frontier
"""


def _resources_chapter() -> str:
    return """# Part III. Resource Constraints

## Chapter 3. On-Chip Resource Limits

#### 3.1 Register Pressure

Summary  
Register usage per thread constrains occupancy and therefore can limit throughput indirectly.

Mechanism  
Higher register allocation reduces the number of resident warps when the per-SM register budget becomes limiting.

Quantitative Understanding  
The performance sensitivity to register pressure on this GPU is not yet calibrated.

Evidence  
- No accepted local evidence yet

Open Questions  
- When does register pressure become the dominant occupancy limiter on this GPU?

Cross References  
- [1.1 Threads, Warps, and Thread Blocks]
- [1.2 Warp Scheduling]

Status  
Frontier
"""


def _modeling_chapter() -> str:
    return """# Part IV. Quantitative Performance Modeling

## Chapter 4. Performance Limits and Interpretation

#### 4.1 Roofline Interpretation

Summary  
A roofline-style model requires trustworthy local ceilings before it can support quantitative reasoning.

Mechanism  
Useful roofline analysis depends on measured or well-justified compute and bandwidth ceilings and clear conditions of applicability.

Quantitative Understanding  
No trusted local roofline has yet been established.

Evidence  
- No accepted local evidence yet

Open Questions  
- What is the best first benchmark to establish a trustworthy bandwidth ceiling?

Cross References  
- [2.1 Global Memory Access]
- [2.2 Coalescing]

Status  
Frontier
"""
