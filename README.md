# Minimal Python CLI Todo

A tiny todo CLI built with `argparse` and JSON storage.

## Commands

- `python todo.py add "buy milk"`
- `python todo.py list`
- `python todo.py done 0`

Data is stored in `todos.json` in the current working directory.

## GPU Auto Profiling Infra

Package layout:
- `gpu_profiler/agents.py`
- `gpu_profiler/orchestrator.py`
- `gpu_profiler/store.py`
- `gpu_profiler/models.py`
- `gpu_profiler/cli.py`
- `gpu_autoprofile.py` (entrypoint)

Run a multi-agent profiling cycle:

```bash
python gpu_autoprofile.py run --workload "python -c \"print(123)\"" --samples 2 --interval 0.2 --retries 2 --retry-delay 0.1
```

Artifacts are written under `profiling_runs/<run_id>/`.

## Run tests

```bash
pytest -q
```
