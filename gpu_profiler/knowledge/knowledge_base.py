from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.store import write_text


def initialize_markdown_knowledge_base(run_dir: Path, intent: str) -> dict[str, str]:
    kb_dir = run_dir / "knowledge_base"
    updates_dir = kb_dir / "updates"
    kb_dir.mkdir(parents=True, exist_ok=True)
    updates_dir.mkdir(parents=True, exist_ok=True)

    book_path = kb_dir / "knowledge_book.md"
    files = {
        "knowledge_base_dir": str(kb_dir),
        "knowledge_base_book_artifact": str(book_path),
        "knowledge_base_index_artifact": str(book_path),
        "knowledge_base_frontier_artifact": str(kb_dir / "frontier.md"),
        "knowledge_base_findings_artifact": str(kb_dir / "local_findings.md"),
    }
    _write_if_missing(book_path, _empty_book_md(intent))
    write_text(Path(files["knowledge_base_frontier_artifact"]), _render_frontier_md(intent=intent, candidates=[]))
    _write_if_missing(Path(files["knowledge_base_findings_artifact"]), _initial_findings_md())
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
    write_text(update_path, _render_iteration_update_md(iteration=iteration, analysis=analysis))
    proposed_questions = analysis.get("proposed_follow_up_questions", [])
    if not isinstance(proposed_questions, list):
        proposed_questions = []
    _insert_follow_up_questions(files["knowledge_base_book_artifact"], proposed_questions)
    write_text(
        Path(files["knowledge_base_frontier_artifact"]),
        _render_frontier_md(
            intent=intent,
            candidates=extract_frontier_candidates({**kb, **files}),
            covered_dimensions=kb.get("covered_dimensions", []),
            coverage_score=kb.get("coverage_score", 0.0),
        ),
    )
    write_text(Path(files["knowledge_base_findings_artifact"]), _render_local_findings_md(kb=kb))
    files["knowledge_base_latest_update_artifact"] = str(update_path)
    return files


def update_question_context(kb: dict[str, Any], *, question_text: str, context: str) -> dict[str, str]:
    files = _files_from_kb(kb)
    book_path = files.get("knowledge_base_book_artifact", "")
    if not book_path:
        return {}
    title, chapters = _parse_book_document(_read_text(book_path))
    changed = False
    for chapter in chapters:
        for section in chapter["sections"]:
            for question in section["questions"]:
                if _question_matches(question, question_text):
                    question["context"] = _merge_text(question.get("context", ""), context)
                    changed = True
    if changed:
        write_text(Path(book_path), _render_book_document(title, chapters))
    return files


def answer_question(
    kb: dict[str, Any],
    *,
    question_text: str,
    answer: str,
    evidence: list[str],
    resolved: bool,
    why_it_matters: str = "",
    context: str = "",
) -> dict[str, str]:
    files = _files_from_kb(kb)
    book_path = files.get("knowledge_base_book_artifact", "")
    if not book_path:
        return {}
    title, chapters = _parse_book_document(_read_text(book_path))
    changed = False
    for chapter in chapters:
        for section in chapter["sections"]:
            for question in section["questions"]:
                if not _question_matches(question, question_text):
                    continue
                if why_it_matters:
                    question["why_it_matters"] = _merge_text(question.get("why_it_matters", ""), why_it_matters)
                if context:
                    question["context"] = _merge_text(question.get("context", ""), context)
                question["answer"] = _merge_text(question.get("answer", ""), answer)
                question["evidence"] = _merge_evidence(question.get("evidence", []), evidence)
                question["resolved"] = "Yes" if resolved else "No"
                changed = True
    if changed:
        write_text(Path(book_path), _render_book_document(title, chapters))
    return files


def consolidate_markdown_knowledge_base(
    run_dir: Path,
    *,
    intent: str,
    kb: dict[str, Any],
    iteration: int,
    proposed_questions: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    files = initialize_markdown_knowledge_base(run_dir, intent)
    book_path = files["knowledge_base_book_artifact"]
    title, chapters = _parse_book_document(_read_text(book_path))
    for chapter in chapters:
        for section in chapter["sections"]:
            _consolidate_answered_questions(section)
    _insert_follow_up_questions_to_chapters(chapters, proposed_questions if isinstance(proposed_questions, list) else [])
    write_text(Path(book_path), _render_book_document(title, chapters))
    write_text(
        Path(files["knowledge_base_frontier_artifact"]),
        _render_frontier_md(
            intent=intent,
            candidates=extract_frontier_candidates({**kb, **files}),
            covered_dimensions=kb.get("covered_dimensions", []),
            coverage_score=kb.get("coverage_score", 0.0),
        ),
    )
    write_text(Path(files["knowledge_base_findings_artifact"]), _render_local_findings_md(kb=kb))
    files["knowledge_base_latest_update_artifact"] = str(
        Path(files["knowledge_base_dir"]) / "updates" / f"iter_{iteration:02d}.md"
    )
    return files


def load_markdown_knowledge_base_memos(kb: dict[str, Any]) -> dict[str, str]:
    files = _files_from_kb(kb)
    book_path = str(files.get("knowledge_base_book_artifact", "")).strip()
    frontier_path = str(files.get("knowledge_base_frontier_artifact", "")).strip()
    frontier_candidates = extract_frontier_candidates({**kb, **files})
    return {
        "knowledge_base_book_memo": _read_text(book_path),
        "knowledge_base_frontier_memo": _read_text(frontier_path),
        "knowledge_base_frontier_questions": [_candidate_display_question(item) for item in frontier_candidates],
        "knowledge_base_frontier_candidates": frontier_candidates,
    }


def list_book_sections(kb: dict[str, Any]) -> list[dict[str, Any]]:
    files = _files_from_kb(kb)
    book_path = str(files.get("knowledge_base_book_artifact", "")).strip()
    _, chapters = _parse_book_document(_read_text(book_path))
    out: list[dict[str, Any]] = []
    for chapter in chapters:
        chapter_name = str(chapter.get("chapter", "")).strip()
        for section in chapter.get("sections", []):
            out.append(
                {
                    "chapter": chapter_name,
                    "section": str(section.get("section", "")).strip(),
                    "summary": str(section.get("summary", "")).strip(),
                    "mechanism": str(section.get("mechanism", "")).strip(),
                    "evidence": [str(x).strip() for x in section.get("evidence", []) if str(x).strip()],
                    "current_understanding": str(section.get("current_understanding", "")).strip(),
                    "uncertainty": str(section.get("uncertainty", "")).strip(),
                    "questions": [_normalize_question(item) for item in section.get("questions", [])],
                }
            )
    return out


def set_section_questions(kb: dict[str, Any], *, section_heading: str, questions: list[dict[str, Any]]) -> dict[str, str]:
    files = _files_from_kb(kb)
    book_path = files.get("knowledge_base_book_artifact", "")
    if not book_path:
        return {}
    title, chapters = _parse_book_document(_read_text(book_path))
    normalized_questions = [_normalize_question(item) for item in questions if isinstance(item, dict)]
    changed = False
    for chapter in chapters:
        for section in chapter["sections"]:
            if str(section.get("section", "")).strip() != str(section_heading or "").strip():
                continue
            section["questions"] = normalized_questions
            changed = True
    if changed:
        write_text(Path(book_path), _render_book_document(title, chapters))
    return files


def extract_frontier_questions(kb: dict[str, Any]) -> list[str]:
    return [_candidate_display_question(item) for item in extract_frontier_candidates(kb)]


def extract_frontier_candidates(kb: dict[str, Any]) -> list[dict[str, Any]]:
    files = _files_from_kb(kb)
    book_path = str(files.get("knowledge_base_book_artifact", "")).strip()
    title, chapters = _parse_book_document(_read_text(book_path))
    _ = title
    candidates: list[dict[str, Any]] = []
    for chapter in chapters:
        for section in chapter["sections"]:
            for index, question in enumerate(section["questions"]):
                q = _normalize_question(question)
                if q["resolved"] == "Yes":
                    continue
                candidates.append(
                    {
                        "question": q["question"],
                        "section_refs": [section["section"]],
                        "why_it_matters": q["why_it_matters"],
                        "context": q["context"],
                        "source": "book",
                        "question_order": index,
                        "section_order": section["section_order"],
                    }
                )
    return sorted(
        candidates,
        key=lambda item: (
            item.get("section_order", (999, 999))[0],
            item.get("section_order", (999, 999))[1],
            item.get("question_order", 999),
        ),
    )[:24]


def _files_from_kb(kb: dict[str, Any]) -> dict[str, str]:
    files = {
        "knowledge_base_dir": str(kb.get("knowledge_base_dir", "")).strip(),
        "knowledge_base_book_artifact": str(kb.get("knowledge_base_book_artifact", "")).strip(),
        "knowledge_base_index_artifact": str(kb.get("knowledge_base_index_artifact", "")).strip(),
        "knowledge_base_frontier_artifact": str(kb.get("knowledge_base_frontier_artifact", "")).strip(),
        "knowledge_base_findings_artifact": str(kb.get("knowledge_base_findings_artifact", "")).strip(),
    }
    if files["knowledge_base_book_artifact"] and not files["knowledge_base_index_artifact"]:
        files["knowledge_base_index_artifact"] = files["knowledge_base_book_artifact"]
    return {k: v for k, v in files.items() if v}


def _parse_book_document(text: str) -> tuple[str, list[dict[str, Any]]]:
    title = "GPU Architecture Knowledge Book"
    chapters: list[dict[str, Any]] = []
    current_chapter: dict[str, Any] | None = None
    current_section_heading = ""
    current_section_lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip() or title
            continue
        if stripped.startswith("## "):
            if current_chapter is not None and current_section_heading:
                current_chapter["sections"].append(_parse_section_block(current_section_heading, current_section_lines))
                current_section_heading = ""
                current_section_lines = []
            if current_chapter is not None:
                chapters.append(current_chapter)
            current_chapter = {"chapter": stripped[3:].strip(), "sections": []}
            continue
        if stripped.startswith("### "):
            if current_chapter is None:
                current_chapter = {"chapter": "Chapter", "sections": []}
            if current_section_heading:
                current_chapter["sections"].append(_parse_section_block(current_section_heading, current_section_lines))
            current_section_heading = stripped[4:].strip()
            current_section_lines = []
            continue
        if current_section_heading:
            current_section_lines.append(raw_line)
    if current_chapter is not None and current_section_heading:
        current_chapter["sections"].append(_parse_section_block(current_section_heading, current_section_lines))
    if current_chapter is not None:
        chapters.append(current_chapter)
    return title, chapters


def _parse_section_block(heading: str, lines: list[str]) -> dict[str, Any]:
    section = {
        "section": heading,
        "section_order": _section_order_from_heading(heading),
        "summary": "",
        "mechanism": "",
        "evidence": [],
        "current_understanding": "",
        "uncertainty": "",
        "questions": [],
    }
    current_field = ""
    current_question: dict[str, Any] | None = None

    def flush_question() -> None:
        nonlocal current_question
        if current_question is not None:
            section["questions"].append(_normalize_question(current_question))
        current_question = None

    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped in {"Summary", "Mechanism", "Evidence", "Current Understanding", "Uncertainty", "Questions"}:
            if current_field == "Questions":
                flush_question()
            current_field = stripped
            continue
        if not current_field:
            continue
        if current_field == "Evidence":
            if stripped.startswith("- "):
                item = stripped[2:].strip()
                if item:
                    section["evidence"].append(item)
            elif stripped:
                section["evidence"].append(stripped)
            continue
        if current_field == "Questions":
            if stripped.startswith("- Question:"):
                flush_question()
                current_question = {"question": stripped.split(":", 1)[1].strip()}
                continue
            if current_question is None and stripped:
                current_question = {"question": stripped}
                continue
            if current_question is None:
                continue
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                mapped = _question_field_name(key.strip())
                if mapped == "evidence":
                    current_question["evidence"] = [item.strip() for item in value.split(" | ") if item.strip()]
                elif mapped:
                    current_question[mapped] = value.strip()
                continue
            if stripped:
                current_question["answer"] = _merge_text(current_question.get("answer", ""), stripped)
            continue
        if stripped:
            key = {
                "Summary": "summary",
                "Mechanism": "mechanism",
                "Current Understanding": "current_understanding",
                "Uncertainty": "uncertainty",
            }[current_field]
            section[key] = _merge_text(section[key], stripped)
    if current_field == "Questions":
        flush_question()
    return section


def _render_book_document(title: str, chapters: list[dict[str, Any]]) -> str:
    lines = [f"# {title}", ""]
    for chapter in chapters:
        lines.append(f"## {chapter.get('chapter', '')}")
        lines.append("")
        for index, section in enumerate(chapter.get("sections", [])):
            if index:
                lines.append("")
            lines.extend(_render_section_block(section))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_section_block(section: dict[str, Any]) -> list[str]:
    lines = [
        f"### {section.get('section', '')}",
        "",
        "Summary",
        section.get("summary", "").strip() or "No summary recorded yet.",
        "",
        "Mechanism",
        section.get("mechanism", "").strip() or "No mechanism recorded yet.",
        "",
        "Evidence",
    ]
    evidence = [str(x).strip() for x in section.get("evidence", []) if str(x).strip()]
    if evidence:
        for item in evidence:
            lines.append(f"- {item}")
    else:
        lines.append("- No evidence recorded yet.")
    lines.extend(
        [
            "",
            "Current Understanding",
            section.get("current_understanding", "").strip() or "No current understanding recorded yet.",
            "",
            "Uncertainty",
            section.get("uncertainty", "").strip() or "No explicit uncertainty recorded yet.",
            "",
            "Questions",
        ]
    )
    questions = section.get("questions", [])
    if not questions:
        lines.append("- No unresolved question recorded at this time.")
        return lines
    for question in questions:
        q = _normalize_question(question)
        lines.extend(
            [
                f"- Question: {q['question']}",
                f"  Why It Matters: {q['why_it_matters'] or 'No reason recorded yet.'}",
                f"  Context: {q['context'] or 'No context recorded yet.'}",
                f"  Answer: {q['answer'] or 'Not answered yet.'}",
                f"  Evidence: {' | '.join(q['evidence']) if q['evidence'] else 'No supporting evidence recorded yet.'}",
                f"  Resolved: {q['resolved']}",
            ]
        )
    return lines


def _normalize_question(question: dict[str, Any]) -> dict[str, Any]:
    evidence = question.get("evidence", [])
    if isinstance(evidence, str):
        evidence_list = [item.strip() for item in evidence.split(" | ") if item.strip()]
    else:
        evidence_list = [str(item).strip() for item in evidence if str(item).strip()]
    return {
        "question": str(question.get("question", "")).strip(),
        "why_it_matters": str(question.get("why_it_matters", "")).strip(),
        "context": str(question.get("context", "")).strip(),
        "answer": str(question.get("answer", "")).strip(),
        "evidence": evidence_list,
        "resolved": "Yes" if str(question.get("resolved", "No")).strip().lower() in {"yes", "true"} else "No",
    }


def _question_field_name(label: str) -> str:
    normalized = " ".join(label.split()).lower()
    return {
        "question": "question",
        "why it matters": "why_it_matters",
        "context": "context",
        "answer": "answer",
        "evidence": "evidence",
        "resolved": "resolved",
    }.get(normalized, "")


def _candidate_display_question(item: dict[str, Any]) -> str:
    refs = [str(ref).strip() for ref in item.get("section_refs", []) if str(ref).strip()]
    prefix = f"{refs[0]}: " if refs else ""
    return f"{prefix}{str(item.get('question', '')).strip()}"


def _question_matches(question: dict[str, Any], question_text: str) -> bool:
    candidate = str(question.get("question", "")).strip()
    target = str(question_text or "").strip()
    return bool(candidate and target and (candidate == target or candidate.endswith(target) or target.endswith(candidate)))


def _merge_text(existing: Any, new_text: Any) -> str:
    left = str(existing or "").strip()
    right = str(new_text or "").strip()
    if not right:
        return left
    if not left:
        return right
    if right in left:
        return left
    return f"{left} {right}".strip()


def _merge_evidence(existing: list[Any], new_items: list[Any]) -> list[str]:
    merged = [str(item).strip() for item in existing if str(item).strip()]
    for item in new_items:
        text = str(item).strip()
        if text and text not in merged:
            merged.append(text)
    return merged[:12]


def _consolidate_answered_questions(section: dict[str, Any]) -> None:
    remaining: list[dict[str, Any]] = []
    for question in section.get("questions", []):
        q = _normalize_question(question)
        if q["resolved"] != "Yes":
            remaining.append(q)
            continue
        if q["why_it_matters"]:
            section["summary"] = _merge_text(section["summary"], q["why_it_matters"])
            section["current_understanding"] = _merge_text(section["current_understanding"], q["why_it_matters"])
        if q["context"]:
            section["mechanism"] = _merge_text(section["mechanism"], q["context"])
            section["uncertainty"] = _merge_text(section["uncertainty"], q["context"])
        if q["answer"]:
            section["current_understanding"] = _merge_text(section["current_understanding"], q["answer"])
        section["evidence"] = _merge_evidence(section["evidence"], q["evidence"])
    section["questions"] = remaining


def _insert_follow_up_questions(book_path: str, proposed_questions: list[dict[str, Any]]) -> None:
    if not proposed_questions:
        return
    title, chapters = _parse_book_document(_read_text(book_path))
    _insert_follow_up_questions_to_chapters(chapters, proposed_questions)
    write_text(Path(book_path), _render_book_document(title, chapters))


def _insert_follow_up_questions_to_chapters(chapters: list[dict[str, Any]], proposed_questions: list[dict[str, Any]]) -> None:
    if not proposed_questions:
        return
    sections = [section for chapter in chapters for section in chapter.get("sections", [])]
    for item in proposed_questions:
        if not isinstance(item, dict):
            continue
        question_text = str(item.get("question", "")).strip()
        if not question_text:
            continue
        target_section = str(item.get("section", "")).strip()
        target = None
        if target_section:
            for section in sections:
                if str(section.get("section", "")).strip() == target_section:
                    target = section
                    break
        if target is None and sections:
            target = sections[0]
        if target is None:
            continue
        if any(_question_matches(question, question_text) for question in target.get("questions", [])):
            continue
        target.setdefault("questions", []).append(
            _normalize_question(
                {
                    "question": question_text,
                    "why_it_matters": str(item.get("why_it_matters", "")).strip(),
                    "context": str(item.get("context", "")).strip(),
                    "answer": "",
                    "evidence": [],
                    "resolved": "No",
                }
            )
        )


def _section_order_from_heading(heading: str) -> tuple[int, int]:
    token = str(heading or "").strip().split(" ", 1)[0]
    if "." not in token:
        return (999, 999)
    parts = token.split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return (999, 999)


def _render_frontier_md(
    *,
    intent: str,
    candidates: list[dict[str, Any]],
    covered_dimensions: list[Any] | None = None,
    coverage_score: float | int | None = None,
) -> str:
    lines = ["# Frontier", "", "## Intent", intent]
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
    if not candidates:
        lines.append("1. No explicit frontier question has been recorded yet.")
        return "\n".join(lines)
    for index, item in enumerate(candidates, start=1):
        lines.append(f"{index}. {_candidate_display_question(item)}")
        why = str(item.get("why_it_matters", "")).strip()
        if why:
            lines.append(f"   Why it matters: {why}")
    return "\n".join(lines)


def _render_iteration_update_md(*, iteration: int, analysis: dict[str, Any]) -> str:
    lines = [
        f"# Iteration {iteration} Book Update",
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
    lines = ["# Local Findings", "", "## Iteration History"]
    history = kb.get("history", []) if isinstance(kb.get("history", []), list) else []
    if not history:
        lines.append("- No completed analysis iterations yet.")
        return "\n".join(lines)
    for item in history:
        lines.append(
            f"- iter {item.get('iteration')}: coverage={item.get('coverage_score')}, claims_added={item.get('claims_added')}, summary={item.get('summary', '')}"
        )
    return "\n".join(lines)


def _empty_book_md(intent: str) -> str:
    return f"<!-- intent: {intent} -->\n\n# GPU Architecture Knowledge Book\n"


def _initial_findings_md() -> str:
    return "# Local Findings\n\n## Iteration History\n- No completed analysis iterations yet.\n"


def _write_if_missing(path: Path, text: str) -> None:
    if not path.exists():
        write_text(path, text)


def _read_text(path_text: str) -> str:
    path = Path(path_text)
    if not path_text or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""
