import re
from typing import Any


_HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$")


def _normalize_heading(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _strip_ticks(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("`") and value.endswith("`") and len(value) >= 2:
        return value[1:-1].strip()
    return value


def split_markdown_sections(text: str, level: int = 2) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    expected_prefix = "#" * max(1, level)
    for raw_line in str(text or "").splitlines():
        match = _HEADING_RE.match(raw_line)
        if match and len(match.group(1)) == level:
            current = _normalize_heading(match.group(2))
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(raw_line)
    return {key: "\n".join(lines).strip() for key, lines in sections.items()}


def markdown_section(text: str, heading: str, level: int = 2) -> str:
    return split_markdown_sections(text, level=level).get(_normalize_heading(heading), "")


def markdown_bullets(text: str) -> list[str]:
    items: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        items.append(_strip_ticks(line[2:].strip()))
    return items


def markdown_key_values(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in markdown_bullets(text):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        fields[_normalize_heading(key)] = _strip_ticks(value.strip())
    return fields


def markdown_scalar(text: str) -> str:
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            return _strip_ticks(line)
    return ""


def parse_research_request_markdown(text: str) -> dict[str, Any]:
    return {
        "request_summary": markdown_scalar(markdown_section(text, "Objective")),
        "target_nodes": markdown_bullets(markdown_section(text, "Target Nodes")),
        "target_questions": markdown_bullets(markdown_section(text, "Questions")),
        "search_topics": markdown_bullets(markdown_section(text, "Search Topics")),
        "expected_outputs": markdown_bullets(markdown_section(text, "Expected Outputs")),
    }


def parse_research_markdown(text: str) -> dict[str, Any]:
    findings_section = markdown_section(text, "Findings")
    finding_sections = split_markdown_sections(findings_section, level=3)
    findings: list[dict[str, Any]] = []
    for title, body in finding_sections.items():
        fields = markdown_key_values(body)
        findings.append(
            {
                "title": fields.get("title", title.strip()),
                "relevance": fields.get("relevance", ""),
                "source_url": fields.get("source", ""),
                "summary": fields.get("summary", ""),
            }
        )
    return {
        "request_summary": markdown_scalar(markdown_section(text, "Request Summary")),
        "proposed_dimensions": markdown_bullets(markdown_section(text, "Proposed Dimensions")),
        "unanswered_questions": markdown_bullets(markdown_section(text, "Unanswered Questions")),
        "findings": findings,
    }


def parse_analysis_markdown(text: str) -> dict[str, Any]:
    metadata = markdown_key_values(markdown_section(text, "Metadata"))
    claims_section = markdown_section(text, "Claims")
    claim_sections = split_markdown_sections(claims_section, level=3)
    claims: list[dict[str, Any]] = []
    for title, body in claim_sections.items():
        fields = markdown_key_values(body)
        evidence_refs = markdown_bullets(markdown_section(body, "Evidence", level=4))
        dimensions = markdown_bullets(markdown_section(body, "Dimensions", level=4))
        claims.append(
            {
                "claim": fields.get("claim", title.strip()),
                "claim_type": fields.get("claim type", ""),
                "confidence": fields.get("confidence", ""),
                "status": fields.get("status", ""),
                "method_summary": fields.get("method summary", ""),
                "dimensions": dimensions,
                "evidence_refs": evidence_refs,
            }
        )
    return {
        "summary": metadata.get("summary", ""),
        "reason": metadata.get("reason", ""),
        "stop": metadata.get("stop", "").lower() == "true",
        "covered_dimensions": markdown_bullets(markdown_section(text, "Covered Dimensions")),
        "required_observability": markdown_bullets(markdown_section(text, "Required Observability")),
        "claims": claims,
    }
