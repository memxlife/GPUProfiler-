#!/usr/bin/env python3
import argparse
import json
import os
import platform
import sys
import time
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal OpenAI API connectivity diagnostic.")
    parser.add_argument("--model", default="gpt-5.4", help="Model name to call.")
    parser.add_argument("--timeout", type=float, default=20.0, help="Client timeout in seconds.")
    parser.add_argument("--retries", type=int, default=1, help="Number of attempts.")
    parser.add_argument("--message", default="Say hello in one short sentence.", help="Tiny user prompt.")
    return parser


def masked_env() -> dict[str, Any]:
    return {
        "OPENAI_API_KEY_present": bool(os.environ.get("OPENAI_API_KEY")),
        "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", ""),
        "HTTPS_PROXY": os.environ.get("HTTPS_PROXY", ""),
        "HTTP_PROXY": os.environ.get("HTTP_PROXY", ""),
        "ALL_PROXY": os.environ.get("ALL_PROXY", ""),
        "NO_PROXY": os.environ.get("NO_PROXY", ""),
    }


def run_attempt(model: str, timeout_sec: float, message: str) -> dict[str, Any]:
    start = time.time()
    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "elapsed_sec": time.time() - start,
            "error_type": type(exc).__name__,
            "error": repr(exc),
        }

    try:
        client = OpenAI(timeout=timeout_sec, api_key=os.environ.get("OPENAI_API_KEY") or None)
        response = client.responses.create(
            model=model,
            max_output_tokens=120,
            input=[
                {"role": "system", "content": "Answer briefly in plain text."},
                {"role": "user", "content": message},
            ],
        )
        return {
            "status": "ok",
            "elapsed_sec": time.time() - start,
            "output_text": (response.output_text or "").strip(),
            "response_id": getattr(response, "id", ""),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "elapsed_sec": time.time() - start,
            "error_type": type(exc).__name__,
            "error": repr(exc),
        }


def main() -> int:
    args = build_parser().parse_args()
    report: dict[str, Any] = {
        "tool": "openai_ping",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "model": args.model,
        "timeout_sec": args.timeout,
        "retries": args.retries,
        "env": masked_env(),
        "attempts": [],
    }

    for attempt in range(1, max(1, args.retries) + 1):
        result = run_attempt(model=args.model, timeout_sec=args.timeout, message=args.message)
        result["attempt"] = attempt
        report["attempts"].append(result)
        if result["status"] == "ok":
            break

    print(json.dumps(report, indent=2))
    return 0 if any(item.get("status") == "ok" for item in report["attempts"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
