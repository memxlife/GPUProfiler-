import time
import re
from pathlib import Path
from typing import Any

from ..core.store import write_text


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
    update_path = Path(files["knowledge_base_dir"]) / "updates" / f"iter_{iteration:02d}.md"
    analysis_ref = f"../iterations/iter_{iteration:02d}/analysis.md"
    write_text(update_path, _render_iteration_update_md(iteration=iteration, analysis=analysis))
    _apply_analysis_to_chapters(
        files=files,
        analysis=analysis,
        covered_dimensions=kb.get("covered_dimensions", []),
        update_ref=f"updates/iter_{iteration:02d}.md",
        analysis_ref=analysis_ref,
    )
    frontier_questions = _frontier_questions(
        kb=kb,
        knowledge_model=knowledge_model,
        analysis=analysis,
        files=files,
    )
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
    write_text(
        Path(files["knowledge_base_findings_artifact"]),
        _render_local_findings_md(kb=kb),
    )
    files["knowledge_base_latest_update_artifact"] = str(update_path)
    return files


def load_markdown_knowledge_base_memos(kb: dict[str, Any]) -> dict[str, str]:
    index_path = str(kb.get("knowledge_base_index_artifact", "")).strip()
    frontier_path = str(kb.get("knowledge_base_frontier_artifact", "")).strip()
    chapter_paths = [
        str(kb.get("knowledge_base_foundations_artifact", "")).strip(),
        str(kb.get("knowledge_base_memory_artifact", "")).strip(),
        str(kb.get("knowledge_base_resources_artifact", "")).strip(),
        str(kb.get("knowledge_base_modeling_artifact", "")).strip(),
    ]
    chapter_texts = [_read_text(path) for path in chapter_paths if path]
    frontier_candidates = extract_frontier_candidates(kb)
    frontier_questions = [str(item.get("question", "")).strip() for item in frontier_candidates if str(item.get("question", "")).strip()]
    return {
        "knowledge_base_book_memo": "\n\n".join(part for part in [_read_text(index_path), *chapter_texts] if part),
        "knowledge_base_frontier_memo": _read_text(frontier_path),
        "knowledge_base_frontier_questions": frontier_questions,
        "knowledge_base_frontier_candidates": frontier_candidates,
    }


def extract_frontier_questions(kb: dict[str, Any]) -> list[str]:
    return [str(item.get("question", "")).strip() for item in extract_frontier_candidates(kb) if str(item.get("question", "")).strip()]


def extract_frontier_candidates(kb: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    section_index: dict[str, dict[str, Any]] = {}
    frontier_path = str(kb.get("knowledge_base_frontier_artifact", "")).strip()
    candidates.extend(_parse_frontier_markdown(_read_text(frontier_path)))
    for artifact_key in (
        "knowledge_base_foundations_artifact",
        "knowledge_base_memory_artifact",
        "knowledge_base_resources_artifact",
        "knowledge_base_modeling_artifact",
    ):
        path_text = str(kb.get(artifact_key, "")).strip()
        if not path_text:
            continue
        chapter_sections = _parse_chapter_sections(_read_text(path_text))
        for section in chapter_sections:
            section_name = str(section.get("section", "")).strip()
            if section_name:
                section_index[section_name] = section
        candidates.extend(_parse_open_questions_from_chapter(_read_text(path_text)))
    for item in candidates:
        refs = [str(ref).strip() for ref in item.get("section_refs", []) if str(ref).strip()]
        section_meta = None
        for ref in refs:
            if ref in section_index:
                section_meta = section_index[ref]
                break
        if section_meta is None:
            continue
        item["status"] = section_meta.get("status", "")
        item["prerequisites"] = section_meta.get("prerequisites", [])
        item["frontier_criteria"] = section_meta.get("frontier_criteria", [])
        item["unmet_frontier_criteria"] = _unmet_frontier_criteria(section_meta)
        item["unsatisfied_prerequisites"] = _unsatisfied_prerequisites(
            section_meta.get("prerequisites", []),
            section_index,
        )
    deduped: dict[str, dict[str, Any]] = {}
    for item in candidates:
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        score = _candidate_sort_key(item)
        existing = deduped.get(question)
        if existing is None or score < _candidate_sort_key(existing):
            deduped[question] = item
    ordered = sorted(deduped.values(), key=_candidate_sort_key)
    return ordered[:16]


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


def _apply_analysis_to_chapters(
    *,
    files: dict[str, str],
    analysis: dict[str, Any],
    covered_dimensions: list[Any],
    update_ref: str,
    analysis_ref: str,
) -> None:
    for artifact_key in (
        "knowledge_base_foundations_artifact",
        "knowledge_base_memory_artifact",
        "knowledge_base_resources_artifact",
        "knowledge_base_modeling_artifact",
    ):
        path_text = str(files.get(artifact_key, "")).strip()
        if not path_text:
            continue
        path = Path(path_text)
        text = _read_text(path_text)
        if not text:
            continue
        header, sections = _parse_chapter_document(text)
        updated_sections = [
            _update_section_from_analysis(
                section=section,
                analysis=analysis,
                covered_dimensions=covered_dimensions,
                update_ref=update_ref,
                analysis_ref=analysis_ref,
            )
            for section in sections
        ]
        write_text(path, _render_chapter_document(header, updated_sections))


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


def _frontier_questions(
    kb: dict[str, Any],
    knowledge_model: dict[str, Any],
    analysis: dict[str, Any],
    files: dict[str, str] | None = None,
) -> list[str]:
    questions: list[str] = []
    if isinstance(files, dict):
        for artifact_key in (
            "knowledge_base_foundations_artifact",
            "knowledge_base_memory_artifact",
            "knowledge_base_resources_artifact",
            "knowledge_base_modeling_artifact",
        ):
            path_text = str(files.get(artifact_key, "")).strip()
            if not path_text:
                continue
            for item in _parse_open_questions_from_chapter(_read_text(path_text)):
                text = str(item.get("question", "")).strip()
                if text and text not in questions:
                    questions.append(text)
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


def _parse_frontier_markdown(text: str) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    in_frontier = False
    current_index = -1
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if line == "## Active Frontier Questions":
            in_frontier = True
            continue
        if in_frontier and line.startswith("## "):
            break
        if not in_frontier:
            continue
        if line and line[0].isdigit() and ". " in line:
            current_index += 1
            question_text = line.split(". ", 1)[1].strip()
            questions.append(
                {
                    "question": question_text,
                    "source": "frontier",
                    "section_refs": [],
                    "frontier_rank": current_index,
                    "section_order": (999, 999),
                }
            )
            continue
        if current_index >= 0 and line.startswith("- [") and "]" in line:
            ref = line[2:].strip()
            questions[current_index]["section_refs"].append(ref)
            section_order = _section_order_from_reference(ref)
            if section_order is not None and questions[current_index].get("section_order") == (999, 999):
                questions[current_index]["section_order"] = section_order
    return questions


def _parse_open_questions_from_chapter(text: str) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for section in _parse_chapter_sections(text):
        status = str(section.get("status", "")).strip().lower()
        if status not in {"frontier", "unknown"}:
            continue
        for index, question in enumerate(section.get("open_questions", [])):
            question = str(question).strip()
            if question:
                questions.append(
                    {
                        "question": f"{section.get('section')}: {question}" if section.get("section") else question,
                        "source": "chapter",
                        "section_refs": [section.get("section")] if section.get("section") else [],
                        "frontier_rank": index,
                        "section_order": section.get("section_order", (999, 999)),
                        "status": section.get("status", ""),
                        "prerequisites": section.get("prerequisites", []),
                        "frontier_criteria": section.get("frontier_criteria", []),
                    }
                )
    return questions


def _parse_chapter_sections(text: str) -> list[dict[str, Any]]:
    return _parse_chapter_document(text)[1]


def _parse_chapter_document(text: str) -> tuple[list[str], list[dict[str, Any]]]:
    header_lines: list[str] = []
    sections: list[dict[str, Any]] = []
    current_heading = ""
    current_lines: list[str] = []
    in_sections = False
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#### "):
            in_sections = True
            if current_heading:
                sections.append(_parse_section_block(current_heading, current_lines))
            current_heading = stripped[5:].strip()
            current_lines = []
            continue
        if in_sections:
            current_lines.append(raw_line)
        else:
            header_lines.append(raw_line)
    if current_heading:
        sections.append(_parse_section_block(current_heading, current_lines))
    return header_lines, sections


def _parse_section_block(heading: str, lines: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "section": heading,
        "section_order": _section_order_from_heading(heading),
        "summary": "",
        "mechanism": "",
        "quantitative_understanding": "",
        "local_findings": "",
        "evidence": [],
        "cross_references": [],
        "status": "",
        "open_questions": [],
        "prerequisites": [],
        "frontier_criteria": [],
    }
    current_field = ""
    paragraph_fields = {
        "Summary": "summary",
        "Mechanism": "mechanism",
        "Quantitative Understanding": "quantitative_understanding",
        "Local Findings": "local_findings",
    }
    list_fields = {
        "Evidence": "evidence",
        "Open Questions": "open_questions",
        "Cross References": "cross_references",
        "Prerequisites": "prerequisites",
        "Frontier Criteria": "frontier_criteria",
    }
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped in {
            "Summary",
            "Mechanism",
            "Quantitative Understanding",
            "Local Findings",
            "Evidence",
            "Open Questions",
            "Cross References",
            "Status",
            "Prerequisites",
            "Frontier Criteria",
        }:
            current_field = stripped
            continue
        if not current_field:
            continue
        if current_field in list_fields and stripped.startswith("- "):
            item = stripped[2:].strip()
            if item:
                key = list_fields[current_field]
                parsed[key].append(item)
            continue
        if current_field == "Status" and stripped:
            parsed["status"] = stripped
            continue
        if current_field in paragraph_fields and stripped:
            key = paragraph_fields[current_field]
            parsed[key] = f"{parsed[key]} {stripped}".strip()
    return parsed


def _render_chapter_document(header_lines: list[str], sections: list[dict[str, Any]]) -> str:
    lines = list(header_lines)
    if lines and lines[-1].strip():
        lines.append("")
    for index, section in enumerate(sections):
        if index:
            lines.append("")
        lines.extend(_render_section_block(section))
    return "\n".join(lines).rstrip() + "\n"


def _render_section_block(section: dict[str, Any]) -> list[str]:
    lines = [
        f"#### {section.get('section', '')}",
        "",
        "Summary  ",
        str(section.get("summary", "")).strip() or "No summary recorded.",
        "",
        "Mechanism  ",
        str(section.get("mechanism", "")).strip() or "No mechanism recorded.",
        "",
        "Quantitative Understanding  ",
        str(section.get("quantitative_understanding", "")).strip() or "No quantitative understanding recorded.",
        "",
        "Local Findings  ",
        str(section.get("local_findings", "")).strip() or "No local findings recorded yet.",
        "",
        "Evidence  ",
    ]
    evidence = [str(x).strip() for x in section.get("evidence", []) if str(x).strip()]
    if evidence:
        for item in evidence:
            lines.append(f"- {item}")
    else:
        lines.append("- No evidence recorded.")
    lines.extend(["", "Open Questions  "])
    open_questions = [str(x).strip() for x in section.get("open_questions", []) if str(x).strip()]
    if open_questions:
        for item in open_questions:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    lines.extend(["", "Cross References  "])
    refs = [str(x).strip() for x in section.get("cross_references", []) if str(x).strip()]
    if refs:
        for item in refs:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "Status  ",
            str(section.get("status", "")).strip() or "Unknown",
            "",
            "Prerequisites  ",
        ]
    )
    prereqs = [str(x).strip() for x in section.get("prerequisites", []) if str(x).strip()]
    if prereqs:
        for item in prereqs:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    lines.extend(["", "Frontier Criteria  "])
    criteria = [str(x).strip() for x in section.get("frontier_criteria", []) if str(x).strip()]
    if criteria:
        for item in criteria:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    return lines


def _update_section_from_analysis(
    *,
    section: dict[str, Any],
    analysis: dict[str, Any],
    covered_dimensions: list[Any],
    update_ref: str,
    analysis_ref: str,
) -> dict[str, Any]:
    updated = dict(section)
    matches = _claims_for_section(section, analysis.get("claims", []))
    matched_dimensions = _matched_dimensions_for_section(section, covered_dimensions)
    if not matches and not matched_dimensions:
        return updated
    criteria_met, unmet_criteria = _assess_frontier_criteria(
        section=updated,
        matches=matches,
        matched_dimensions=matched_dimensions,
        analysis=analysis,
    )
    if matches and criteria_met and str(updated.get("status", "")).strip().lower() in {"frontier", "unknown"}:
        updated["status"] = "Known"
    evidence = [str(x).strip() for x in updated.get("evidence", []) if str(x).strip()]
    if update_ref not in evidence:
        evidence.append(update_ref)
    if analysis_ref not in evidence:
        evidence.append(analysis_ref)
    for claim in matches[:3]:
        claim_text = str(claim.get("claim", "")).strip()
        if claim_text:
            claim_ref = f"Claim: {claim_text}"
            if claim_ref not in evidence:
                evidence.append(claim_ref)
    updated["evidence"] = evidence[:10]
    quantitative_update = _synthesize_quantitative_update(
        section=updated,
        matches=matches,
        matched_dimensions=matched_dimensions,
        analysis=analysis,
        update_ref=update_ref,
    )
    if quantitative_update:
        updated["quantitative_understanding"] = _merge_text_block(
            str(updated.get("quantitative_understanding", "")).strip(),
            quantitative_update,
        )
    local_finding = _synthesize_local_finding(
        section=updated,
        matches=matches,
        matched_dimensions=matched_dimensions,
        analysis=analysis,
        update_ref=update_ref,
        criteria_met=criteria_met,
        unmet_criteria=unmet_criteria,
    )
    if local_finding:
        updated["local_findings"] = _merge_text_block(
            str(updated.get("local_findings", "")).strip(),
            local_finding,
        )
    if matches and criteria_met:
        updated["open_questions"] = [
            item
            for item in updated.get("open_questions", [])
            if not _question_answered_by_claims(str(item), matches, matched_dimensions)
        ]
    return updated


def _claims_for_section(section: dict[str, Any], claims: list[Any]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    tags = _section_tags(section)
    for item in claims if isinstance(claims, list) else []:
        if not isinstance(item, dict):
            continue
        dims = {str(x).strip().lower() for x in item.get("dimensions", []) if str(x).strip()}
        claim_text = str(item.get("claim", "")).strip().lower()
        if dims.intersection(tags) or any(tag in claim_text for tag in tags if len(tag) > 2):
            matches.append(item)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in matches:
        key = str(item.get("claim", "")).strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _matched_dimensions_for_section(section: dict[str, Any], covered_dimensions: list[Any]) -> list[str]:
    tags = _section_tags(section)
    covered = {str(x).strip().lower() for x in covered_dimensions if str(x).strip()}
    return sorted(covered.intersection(tags))


def _synthesize_quantitative_update(
    *,
    section: dict[str, Any],
    matches: list[dict[str, Any]],
    matched_dimensions: list[str],
    analysis: dict[str, Any],
    update_ref: str,
) -> str:
    claim_summaries = [str(item.get("claim", "")).strip().rstrip(".") for item in matches if str(item.get("claim", "")).strip()]
    if claim_summaries:
        joined = "; ".join(claim_summaries[:3])
        return f"Local evidence update from {update_ref}: {joined}."
    summary = str(analysis.get("summary", "")).strip().rstrip(".")
    if matched_dimensions and summary:
        dims_text = ", ".join(matched_dimensions[:4])
        return f"Local evidence update from {update_ref}: analyzer summary relevant to {dims_text}: {summary}."
    return ""


def _synthesize_local_finding(
    *,
    section: dict[str, Any],
    matches: list[dict[str, Any]],
    matched_dimensions: list[str],
    analysis: dict[str, Any],
    update_ref: str,
    criteria_met: bool,
    unmet_criteria: list[str],
) -> str:
    heading = str(section.get("section", "")).strip()
    claim_summaries = [str(item.get("claim", "")).strip().rstrip(".") for item in matches if str(item.get("claim", "")).strip()]
    if claim_summaries:
        lead = heading or "This section"
        joined = "; ".join(claim_summaries[:2])
        if criteria_met:
            return f"{lead} gained new local support in {update_ref}: {joined}. Frontier criteria were satisfied for this update."
        if unmet_criteria:
            unmet = "; ".join(unmet_criteria[:2])
            return f"{lead} gained partial local support in {update_ref}: {joined}. Frontier criteria remain incomplete: {unmet}."
        return f"{lead} gained new local support in {update_ref}: {joined}."
    summary = str(analysis.get("summary", "")).strip().rstrip(".")
    if summary and matched_dimensions:
        dims_text = ", ".join(matched_dimensions[:3])
        lead = heading or "This section"
        if unmet_criteria:
            unmet = "; ".join(unmet_criteria[:2])
            return f"{lead} received a local analysis update in {update_ref} for {dims_text}: {summary}. Frontier criteria remain incomplete: {unmet}."
        return f"{lead} received a local analysis update in {update_ref} for {dims_text}: {summary}."
    return ""


def _assess_frontier_criteria(
    *,
    section: dict[str, Any],
    matches: list[dict[str, Any]],
    matched_dimensions: list[str],
    analysis: dict[str, Any],
) -> tuple[bool, list[str]]:
    criteria = [str(x).strip() for x in section.get("frontier_criteria", []) if str(x).strip()]
    if not criteria:
        return (bool(matches), [])
    signal_text = _analysis_signal_text(matches=matches, matched_dimensions=matched_dimensions, analysis=analysis)
    unmet = [criterion for criterion in criteria if not _criterion_satisfied(criterion, signal_text)]
    return (not unmet and bool(matches), unmet)


def _analysis_signal_text(
    *,
    matches: list[dict[str, Any]],
    matched_dimensions: list[str],
    analysis: dict[str, Any],
) -> str:
    parts: list[str] = [str(analysis.get("summary", "")).strip().lower()]
    parts.extend(str(item).strip().lower() for item in matched_dimensions if str(item).strip())
    for claim in matches:
        if not isinstance(claim, dict):
            continue
        parts.append(str(claim.get("claim", "")).strip().lower())
        parts.extend(str(item).strip().lower() for item in claim.get("dimensions", []) if str(item).strip())
        parts.append(str(claim.get("method_summary", "")).strip().lower())
    return " ".join(part for part in parts if part)


def _criterion_satisfied(criterion: str, signal_text: str) -> bool:
    text = str(criterion or "").strip().lower()
    signal = str(signal_text or "").strip().lower()
    if not text:
        return True
    if not signal:
        return False
    if "bandwidth estimate" in text or ("bandwidth" in text and "estimate" in text):
        return _contains_any(signal, ["bandwidth", "throughput", "gb/s", "gib/s"]) and (
            _contains_any(signal, ["benchmark", "measured", "measurement", "sustain", "estimate"])
            or _contains_number(signal)
        )
    if "conditions" in text or "reuse" in text or "documented" in text:
        return _contains_any(signal, ["condition", "configuration", "config", "launch", "block", "grid", "documented", "reuse"])
    if "stride" in text or "access pattern" in text:
        return _contains_any(signal, ["stride", "coalescing", "coalesced", "uncoalesced", "access pattern"]) and _contains_any(
            signal, ["throughput", "bandwidth", "trend", "curve", "degrad", "change"]
        )
    if "stable enough" in text or "describe quantitatively" in text:
        return _contains_any(signal, ["stable", "trend", "curve", "threshold", "slope", "quantitative", "regime"])
    if "working-set sweep" in text or "cache-to-dram transition" in text:
        return _contains_any(signal, ["working_set", "working-set", "cache", "l2", "dram"]) and _contains_any(
            signal, ["transition", "regime", "latency", "bandwidth", "sweep"]
        )
    if "register-pressure" in text or "register pressure" in text:
        return _contains_any(signal, ["register", "register_pressure", "registers_per_thread"]) and _contains_any(
            signal, ["occupancy", "throughput", "resident", "warp", "warps"]
        )
    if "occupancy" in text and "throughput" in text:
        return _contains_any(signal, ["occupancy", "warp", "warps"]) and _contains_any(
            signal, ["throughput", "latency", "hide", "hiding"]
        )
    if "ceiling" in text:
        return _contains_any(signal, ["ceiling", "roofline", "bandwidth", "compute"])
    criterion_tokens = [token for token in _question_tokens(text) if token not in {"local", "evidence", "clear", "clearly", "enough", "resulting"}]
    overlaps = [token for token in criterion_tokens if token in signal]
    threshold = 2 if len(criterion_tokens) >= 3 else 1
    return len(overlaps) >= threshold


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(str(item).lower() in text for item in needles)


def _contains_number(text: str) -> bool:
    return re.search(r"\d", str(text or "")) is not None


def _merge_text_block(existing: str, new_text: str) -> str:
    existing = str(existing or "").strip()
    new_text = str(new_text or "").strip()
    if not new_text:
        return existing
    if not existing or existing in {
        "No quantitative understanding recorded.",
        "No local findings recorded yet.",
    }:
        return new_text
    if new_text in existing:
        return existing
    return f"{existing} {new_text}".strip()


def _question_answered_by_claims(question: str, matches: list[dict[str, Any]], matched_dimensions: list[str]) -> bool:
    text = str(question).strip().lower()
    if not text:
        return False
    claim_text = " ".join(str(item.get("claim", "")).strip().lower() for item in matches)
    if any(dim in text for dim in matched_dimensions):
        return True
    for token in _question_tokens(text):
        if token in claim_text:
            return True
    return False


def _question_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in str(text or "").replace("?", " ").replace(",", " ").split():
        token = raw.strip(" .:;()[]`").lower().replace("-", "_")
        if len(token) > 3:
            tokens.append(token)
    return tokens


def _section_tags(section: dict[str, Any]) -> set[str]:
    heading = str(section.get("section", "")).lower()
    tags: set[str] = set()
    if "global memory" in heading or "bandwidth" in heading:
        tags.update({"dram_bandwidth", "global_memory", "memory_bandwidth", "sequential_load", "bandwidth"})
    if "coalescing" in heading:
        tags.update({"coalescing", "stride", "coalesced", "uncoalesced", "transaction_efficiency"})
    if "l2" in heading or "cache" in heading:
        tags.update({"l2", "l2_cache", "cache", "working_set", "cache_hit"})
    if "register" in heading:
        tags.update({"register", "register_pressure", "registers_per_thread"})
    if "warp scheduling" in heading:
        tags.update({"warp_scheduling", "scheduler", "eligible_warps", "issue"})
    if "threads, warps, and thread blocks" in heading:
        tags.update({"warp", "thread_block", "execution_model"})
    if "roofline" in heading:
        tags.update({"roofline", "bandwidth_ceiling", "compute_ceiling", "performance_model"})
    tokens = [token.strip(".,:;()[]`") for token in heading.split()]
    for token in tokens:
        if len(token) > 2:
            tags.add(token.replace("-", "_"))
    return tags


def _section_order_from_heading(heading: str) -> tuple[int, int]:
    text = str(heading or "").strip()
    token = text.split(" ", 1)[0]
    if "." not in token:
        return (999, 999)
    parts = token.split(".")
    if len(parts) < 2:
        return (999, 999)
    try:
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return (999, 999)


def _section_order_from_reference(reference: str) -> tuple[int, int] | None:
    text = str(reference or "").strip()
    if text.startswith("[") and "]" in text:
        text = text[1:text.index("]")]
    order = _section_order_from_heading(text)
    return None if order == (999, 999) else order


def _candidate_sort_key(item: dict[str, Any]) -> tuple[int, int, int, int]:
    source = str(item.get("source", "")).strip()
    source_priority = 0 if source == "chapter" else 1
    section_order = item.get("section_order", (999, 999))
    if not isinstance(section_order, tuple) or len(section_order) != 2:
        section_order = (999, 999)
    frontier_rank = int(item.get("frontier_rank", 999))
    unsatisfied = item.get("unsatisfied_prerequisites", [])
    unsatisfied_count = len(unsatisfied) if isinstance(unsatisfied, list) else 999
    return (unsatisfied_count, section_order[0], section_order[1], source_priority, frontier_rank)


def _unsatisfied_prerequisites(prerequisites: list[Any], section_index: dict[str, dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for item in prerequisites if isinstance(prerequisites, list) else []:
        text = str(item).strip()
        if not text:
            continue
        section = section_index.get(text)
        status = str((section or {}).get("status", "")).strip().lower()
        if status != "known":
            missing.append(text)
    return missing


def _unmet_frontier_criteria(section: dict[str, Any]) -> list[str]:
    criteria = [str(x).strip() for x in section.get("frontier_criteria", []) if str(x).strip()]
    status = str(section.get("status", "")).strip().lower()
    if status == "known":
        return []
    local_findings = str(section.get("local_findings", "")).strip()
    marker = "Frontier criteria remain incomplete:"
    if marker in local_findings:
        tail = local_findings.split(marker, 1)[1].strip()
        tail = tail.rstrip(".")
        parsed = [item.strip() for item in tail.split(";") if item.strip()]
        if parsed:
            return parsed
    return criteria


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

Local Findings  
No local findings recorded yet.

Evidence  
- No accepted local evidence yet

Open Questions  
- Which warp-level effects dominate throughput for simple kernels on this GPU?

Cross References  
- [1.2 Warp Scheduling]
- [2.1 Global Memory Access]

Status  
Known

Prerequisites  
- None

Frontier Criteria  
- Maintain consistency with later locally validated execution evidence
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

Local Findings  
No local findings recorded yet.

Evidence  
- No accepted local evidence yet

Open Questions  
- What sustained DRAM bandwidth can this GPU deliver under a simple sequential-load benchmark?

Cross References  
- [2.2 Coalescing]
- [4.1 Roofline Interpretation]

Status  
Frontier

Prerequisites  
- 1.1 Threads, Warps, and Thread Blocks

Frontier Criteria  
- A bounded local benchmark produces a trustworthy sustained bandwidth estimate
- The benchmark conditions are documented clearly enough for reuse

#### 2.2 Coalescing

Summary  
Coalescing strongly affects effective global-memory throughput.

Mechanism  
Accesses that align well across a warp usually require fewer transactions and therefore achieve higher throughput.

Quantitative Understanding  
The stride-to-bandwidth relationship for this GPU is not yet locally characterized.

Local Findings  
No local findings recorded yet.

Evidence  
- No accepted local evidence yet

Open Questions  
- How does achieved bandwidth vary with stride on this GPU?

Cross References  
- [2.1 Global Memory Access]
- [2.3 L2 Cache]

Status  
Frontier

Prerequisites  
- 2.1 Global Memory Access

Frontier Criteria  
- A stride-controlled benchmark shows how throughput changes with access pattern
- The observed trend is stable enough to describe quantitatively

#### 2.3 L2 Cache

Summary  
The L2 cache mediates traffic between kernels and backing memory and can materially change effective access cost.

Mechanism  
When reuse falls within the effective L2 regime, traffic pressure on DRAM can be reduced substantially.

Quantitative Understanding  
The effective L2-resident regime for this GPU is not yet locally characterized.

Local Findings  
No local findings recorded yet.

Evidence  
- No accepted local evidence yet

Open Questions  
- At what working-set size does behavior transition beyond effective L2 reuse?

Cross References  
- [2.1 Global Memory Access]
- [4.1 Roofline Interpretation]

Status  
Frontier

Prerequisites  
- 2.1 Global Memory Access

Frontier Criteria  
- A working-set sweep identifies a credible cache-to-DRAM transition regime
- The transition can be expressed with clear conditions
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

Local Findings  
No local findings recorded yet.

Evidence  
- No accepted local evidence yet

Open Questions  
- When does register pressure become the dominant occupancy limiter on this GPU?

Cross References  
- [1.1 Threads, Warps, and Thread Blocks]
- [1.2 Warp Scheduling]

Status  
Frontier

Prerequisites  
- 1.1 Threads, Warps, and Thread Blocks

Frontier Criteria  
- A controlled benchmark or analysis isolates register-pressure effects on occupancy or throughput
- The resulting dependence can be stated more precisely than general prior knowledge
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

Local Findings  
No local findings recorded yet.

Evidence  
- No accepted local evidence yet

Open Questions  
- What is the best first benchmark to establish a trustworthy bandwidth ceiling?

Cross References  
- [2.1 Global Memory Access]
- [2.2 Coalescing]

Status  
Frontier

Prerequisites  
- 2.1 Global Memory Access
- 2.2 Coalescing

Frontier Criteria  
- At least one trustworthy local bandwidth ceiling is established
- The relationship between the ceiling and kernel observations is interpretable quantitatively
"""
