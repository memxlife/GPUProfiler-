# Markdown Artifact Spec

This repository is moving to a markdown-first artifact model.

Rule of thumb:

- `*.md` is the canonical representation for persistent agent-written and runtime-written artifacts.
- `*.json` should not be used for persisted run artifacts in the autonomous profiling loop.

The orchestrator should prefer parsing markdown artifacts with deterministic section rules instead of asking agents to emit large JSON payloads.

## Canonical Artifact Roles

Per iteration, the preferred canonical research artifacts are:

- `research.md`
- `implementation.md`
- `analysis.md`

The canonical semantic source of truth is the markdown knowledge base under `knowledge_base/`.

## Section Rules

Artifacts should use stable headings and short bullet fields so Python can parse them deterministically.

Use:

- `##` for top-level sections within an artifact
- `###` for repeated entities such as benchmarks, findings, or claims
- `####` for per-entity lists such as evidence or dimensions
- `- key: value` bullets for scalar metadata
- plain `- item` bullets for list items

Avoid:

- arbitrary heading names for the same concept
- freeform structure with no stable anchors

## research.md

Required sections:

- `## Metadata`
- `## Current Question`
- `## Request Summary`
- `## Proposed Dimensions`
- `## Unanswered Questions`
- `## Findings`

Each finding should use a `### Finding N` subsection with concise metadata and summary fields.

## implementation.md

Required sections:

- `## Metadata`
- `## Benchmarks`

Optional sections:

- `## Feasibility Summary`
- `## Rejected Benchmarks`
- `## Negotiation`
- `## Contract Amendments`
- `## Generated Files`

Each benchmark should use a `### Benchmark N` subsection with fields needed for execution review.

## analysis.md

Required sections:

- `## Metadata`
- `## Covered Dimensions`
- `## Claims`

Optional sections:

- `## Required Observability`
- `## Contract Amendments`

Each claim should use a `### Claim N` subsection with:

- `claim`
- `claim type`
- `confidence`
- `status`
- `method summary`
- `#### Dimensions`
- `#### Evidence`

## Operational Guidance

- planning should rely on the KB plus the selected current question, not separate proposal/request memos
- markdown should become more detailed, not less
- parsers should read markdown first when possible
- operational run-state data may be embedded inside markdown artifacts when structured recovery is needed
