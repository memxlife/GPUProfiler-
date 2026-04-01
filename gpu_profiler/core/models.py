from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Task:
    id: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 0


@dataclass
class AgentContext:
    run_id: str
    run_dir: Path


@dataclass
class RetryPolicy:
    max_retries: int = 1
    retry_delay_sec: float = 0.1
