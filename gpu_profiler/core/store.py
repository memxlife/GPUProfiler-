import json
import re
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_data_artifact(path: Path, data: Any, *, title: str = "Data Artifact") -> None:
    lines = [
        f"# {title}",
        "",
        "```json",
        json.dumps(data, indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def read_data_artifact(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if not match:
        return default
    try:
        return json.loads(match.group(1))
    except Exception:
        return default
