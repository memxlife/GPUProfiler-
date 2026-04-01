# Project Structure

This repository is organized by purpose.

## Root

- `README.md`: user-facing overview and entry commands
- `AGENTS.md`: repo-specific agent instructions
- `gpu_autoprofile.py`: top-level CLI entrypoint

## Runtime Code

- `gpu_profiler/core/`: shared models and storage helpers
- `gpu_profiler/knowledge/`: markdown knowledge base and markdown parsing helpers
- `gpu_profiler/runtime/`: agents and orchestrator
- `gpu_profiler/workflow/`: planner, research, codegen, and analysis backend logic
- `gpu_profiler/cli.py`: package CLI

## Tests

- `tests/`: lightweight architecture-aligned test coverage

## Docs

- `docs/design_phi.md`: project philosophy
- `docs/BUILD_PROMPT.md`: implementation contract
- `docs/MARKDOWN_ARTIFACT_SPEC.md`: markdown artifact rules
- `docs/KNOWLEDGE_BASE_TEMPLATE.md`: canonical KB template
- `docs/KNOWLEDGE_BASE_EXAMPLE.md`: small KB example
- `docs/PROJECT_STRUCTURE.md`: this file

## Scripts

- `scripts/openai_ping.py`: minimal OpenAI connectivity check
- `scripts/openai_research_probe.py`: research-path latency probe
- `scripts/openai_agent_probe.py`: multi-agent OpenAI probe

## Generated Runtime Artifacts

These are intentionally not tracked:

- `profiling_runs/`
- `agent_probe_runs/`
- `research_probe_runs/`
- `openai_timeout_diagnostics/`
