# Repository Guidelines

## Purpose and Design Alignment
This repository should follow the philosophy in `design_phi.md`: build an autonomous, self-evolving research system, not a rigid scripted tool.

Contributor rule of thumb:
- preserve autonomy,
- preserve learning memory,
- preserve traceability,
- avoid hard-coding static assumptions unless required for safety.

## Project Structure & Ownership
Current files are minimal (`todo.py`, `test_todo.py`, `README.md`) and may evolve. As the project grows, organize by responsibility:
- `agent/`: loop control and orchestration
- `memory/`: persistent artifacts and history
- `experiments/`: executable experiment definitions
- `interfaces/`: observer and knowledge views
- `tests/`: behavior and regression tests

If introducing new directories, keep boundaries explicit and documented.

## Core System Invariants
All contributions must keep these invariants intact:
- Closed learning loop: world model -> question -> experiment -> observation -> claim -> update -> next question
- Persistent artifacts (no silent loss of evidence)
- Explicit provenance for claims
- Versioned history
- Reproducible execution
- Bounded safety constraints

## Development and Test Commands
- `python todo.py add "buy milk"`: current CLI smoke path
- `python todo.py list`
- `python todo.py done 0`
- `pytest -q`: run all tests

For new modules, add equivalent reproducible CLI or test entry points.

## Coding and Testing Standards
- Python, PEP 8, 4-space indentation, `snake_case`
- Prefer small, composable functions with explicit inputs/outputs
- Tests must validate behavior, not only implementation details
- Every claim-producing path should be testable and auditable

## Commit and PR Expectations
- Use imperative commit titles (e.g., `Add provenance metadata to claim records`)
- Keep PRs focused and include:
  - design intent (how it supports autonomy/learning)
  - risk and safety impact
  - reproducibility notes
  - `pytest -q` results
