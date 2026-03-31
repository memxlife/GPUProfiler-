import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from .models import AgentContext, Task
from .store import read_json, write_json, write_text


class Agent:
    name = "agent"

    def can_handle(self, task: Task) -> bool:
        raise NotImplementedError

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        raise NotImplementedError


class PlannerAgent(Agent):
    name = "planner"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "plan"

    def run(self, task: Task, _ctx: AgentContext) -> dict[str, Any]:
        workload = task.payload.get("workload")
        samples = int(task.payload.get("samples", 3))
        interval_sec = float(task.payload.get("interval_sec", 0.5))

        # Stages define dependency order; tasks in the same stage can run in parallel.
        return {
            "stages": [
                [
                    {
                        "kind": "collect_metrics",
                        "payload": {
                            "phase": "baseline",
                            "samples": samples,
                            "interval_sec": interval_sec,
                        },
                    },
                    {"kind": "collect_system_info", "payload": {}},
                ],
                [
                    {
                        "kind": "run_workload",
                        "payload": {"command": workload},
                    }
                ],
                [
                    {
                        "kind": "collect_metrics",
                        "payload": {
                            "phase": "post_workload",
                            "samples": samples,
                            "interval_sec": interval_sec,
                        },
                    }
                ],
                [{"kind": "analyze", "payload": {}}],
                [{"kind": "report", "payload": {}}],
            ]
        }


class MetricsCollectorAgent(Agent):
    name = "collector"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "collect_metrics"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        samples = int(task.payload.get("samples", 3))
        interval_sec = float(task.payload.get("interval_sec", 0.5))
        phase = task.payload.get("phase", "unknown")

        records: list[dict[str, Any]] = []
        for _ in range(samples):
            records.append(self._sample_gpu())
            time.sleep(interval_sec)

        out = ctx.run_dir / f"metrics_{phase}.json"
        write_json(out, records)
        return {"phase": phase, "samples": samples, "artifact": str(out), "records": records}

    def _sample_gpu(self) -> dict[str, Any]:
        now = time.time()
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            line = result.stdout.strip().splitlines()[0]
            gpu_util, mem_util, mem_used, mem_total, temp_c, power_w = [v.strip() for v in line.split(",")]
            return {
                "ts": now,
                "source": "nvidia-smi",
                "available": True,
                "gpu_util_pct": float(gpu_util),
                "mem_util_pct": float(mem_util),
                "mem_used_mib": float(mem_used),
                "mem_total_mib": float(mem_total),
                "temp_c": float(temp_c),
                "power_w": float(power_w),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ts": now,
                "source": "nvidia-smi",
                "available": False,
                "error": str(exc),
            }


class SystemInfoAgent(Agent):
    name = "system-info"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "collect_system_info"

    def run(self, _task: Task, ctx: AgentContext) -> dict[str, Any]:
        info = {
            "ts": time.time(),
            "python_version": subprocess.run(
                ["python", "--version"], capture_output=True, text=True
            ).stdout.strip()
            or subprocess.run(["python", "--version"], capture_output=True, text=True).stderr.strip(),
        }
        out = ctx.run_dir / "system_info.json"
        write_json(out, info)
        return {"artifact": str(out), "info": info}


class WorkloadRunnerAgent(Agent):
    name = "runner"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "run_workload"

    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        command = task.payload.get("command")
        if not command:
            raise ValueError("run_workload task missing command")

        start = time.time()
        proc = subprocess.run(shlex.split(command), capture_output=True, text=True)
        elapsed = time.time() - start

        out = {
            "command": command,
            "returncode": proc.returncode,
            "elapsed_sec": elapsed,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        artifact = ctx.run_dir / "workload_result.json"
        write_json(artifact, out)
        out["artifact"] = str(artifact)
        return out


class AnalyzerAgent(Agent):
    name = "analyzer"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "analyze"

    def run(self, _task: Task, ctx: AgentContext) -> dict[str, Any]:
        baseline = read_json(ctx.run_dir / "metrics_baseline.json", [])
        post = read_json(ctx.run_dir / "metrics_post_workload.json", [])

        summary = {
            "baseline": self._summarize(baseline),
            "post_workload": self._summarize(post),
        }

        analysis_file = ctx.run_dir / "analysis.json"
        write_json(analysis_file, summary)
        return {"artifact": str(analysis_file), "summary": summary}

    def _summarize(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [r for r in records if r.get("available")]
        if not valid:
            return {"available": False, "samples": len(records)}

        return {
            "available": True,
            "samples": len(records),
            "gpu_util_avg": sum(r["gpu_util_pct"] for r in valid) / len(valid),
            "mem_util_avg": sum(r["mem_util_pct"] for r in valid) / len(valid),
            "power_avg_w": sum(r["power_w"] for r in valid) / len(valid),
            "temp_avg_c": sum(r["temp_c"] for r in valid) / len(valid),
        }


class ReporterAgent(Agent):
    name = "reporter"

    def can_handle(self, task: Task) -> bool:
        return task.kind == "report"

    def run(self, _task: Task, ctx: AgentContext) -> dict[str, Any]:
        analysis = read_json(ctx.run_dir / "analysis.json", {})
        workload = read_json(ctx.run_dir / "workload_result.json", {})

        lines = [
            f"# GPU Auto Profiling Report ({ctx.run_id})",
            "",
            "## Workload",
            f"- command: `{workload.get('command', 'N/A')}`",
            f"- return code: `{workload.get('returncode', 'N/A')}`",
            f"- elapsed sec: `{round(workload.get('elapsed_sec', 0.0), 3)}`",
            "",
            "## Metrics Summary",
            f"- baseline: `{analysis.get('baseline', {})}`",
            f"- post_workload: `{analysis.get('post_workload', {})}`",
            "",
            "## Artifacts",
            "- metrics_baseline.json",
            "- system_info.json",
            "- workload_result.json",
            "- metrics_post_workload.json",
            "- analysis.json",
            "- run_log.json",
        ]
        report_path = ctx.run_dir / "report.md"
        write_text(report_path, "\n".join(lines))
        return {"artifact": str(report_path)}


def default_agents() -> list[Agent]:
    return [
        PlannerAgent(),
        MetricsCollectorAgent(),
        SystemInfoAgent(),
        WorkloadRunnerAgent(),
        AnalyzerAgent(),
        ReporterAgent(),
    ]
