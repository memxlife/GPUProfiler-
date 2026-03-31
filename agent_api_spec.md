# Agent API Specification

Version: 0.1
Status: Draft Agent API Contract
Last Updated: 2026-03-31

## 1. Purpose

This document defines the invocation and artifact APIs for the major agents in the system:

1. planner
2. search
3. codegen
4. runner
5. analyzer
6. flow_controller

These APIs complement the role specifications by defining:

- invocation inputs
- produced artifacts
- result objects
- error objects
- downstream consumers
- invariants

## 2. Shared Conventions

### 2.1 Standard Result Object

Each agent should return a result object of the form:

```json
{
  "agent": "string",
  "iteration": "int|null",
  "status": "succeeded|failed|partial|stopped",
  "reason": "string",
  "artifacts": {},
  "summary": {},
  "error": null
}
```

### 2.2 Standard Error Object

Each agent should use an error object of the form:

```json
{
  "code": "string",
  "message": "string",
  "recoverable": "bool",
  "details": ["string"]
}
```

### 2.3 Artifact Rule

Artifact paths should point to persisted files. Agents should prefer artifact handoff over large in-memory opaque structures.

## 3. Planner API

### 3.1 Agent Name

- `planner`

### 3.2 Purpose

- read current intent and knowledge state
- maintain domain knowledge hierarchy
- produce next non-executable proposal
- emit `research_request.json` when external search is needed

### 3.3 Invocation Inputs

Required:

```json
{
  "intent": "string",
  "iteration": "int",
  "knowledge_base_artifact": "string|null"
}
```

Optional:

```json
{
  "prior_knowledge_model_artifact": "string|null",
  "prior_proposal_artifact": "string|null",
  "prior_analysis_artifact": "string|null",
  "research_artifact": "string|null",
  "system_info_artifact": "string|null",
  "constraints": {
    "max_proposals": "int|null",
    "curriculum_stage_limit": "string|null"
  }
}
```

### 3.4 Produced Artifacts

Required:

- `knowledge_model.json`
- `proposal.json`
- `proposal.md`

Optional:

- `research_request.json`

### 3.5 Result Object

```json
{
  "agent": "planner",
  "iteration": "int",
  "status": "succeeded|failed|partial",
  "reason": "string",
  "artifacts": {
    "knowledge_model": "string",
    "proposal_json": "string",
    "proposal_md": "string",
    "research_request": "string|null"
  },
  "summary": {
    "focus_nodes": ["string"],
    "proposal_count": "int",
    "research_requested": "bool"
  },
  "error": null
}
```

### 3.6 Downstream Consumers

- search consumes `research_request.json`
- codegen consumes `proposal.json`
- analyzer may read `knowledge_model.json` and `proposal.json`
- flow controller consumes planner result object

### 3.7 Invariants

- must not emit executable code
- must not emit run commands
- must produce `knowledge_model.json` and `proposal.json` on success
- `research_request.json` only when search is needed
- planner output content is domain-specific, but schema is generic

## 4. Search API

### 4.1 Agent Name

- `search`

### 4.2 Purpose

- execute planner-issued `research_request.json`
- gather external knowledge with provenance
- return planner-consumable research artifacts

### 4.3 Invocation Inputs

Required:

```json
{
  "iteration": "int",
  "research_request_artifact": "string"
}
```

Optional:

```json
{
  "prior_research_artifact": "string|null",
  "knowledge_model_artifact": "string|null",
  "system_info_artifact": "string|null",
  "constraints": {
    "max_sources": "int|null",
    "source_preferences_override": ["string"]
  }
}
```

### 4.4 Produced Artifacts

Required:

- `research.json`
- `research.md`

### 4.5 Result Object

```json
{
  "agent": "search",
  "iteration": "int",
  "status": "succeeded|failed|partial",
  "reason": "string",
  "artifacts": {
    "research_json": "string",
    "research_md": "string"
  },
  "summary": {
    "finding_count": "int",
    "unanswered_question_count": "int"
  },
  "error": null
}
```

### 4.6 Downstream Consumers

- planner consumes `research.json`
- flow controller consumes search result object

### 4.7 Invariants

- must not change research scope beyond minor search refinement
- must preserve provenance for findings
- must not generate proposals or code
- must produce `research.json` on success

## 5. Codegen API

### 5.1 Agent Name

- `codegen`

### 5.2 Purpose

- convert `proposal.json` into concrete implementation artifacts
- validate implementability with build and smoke checks
- assess implementation complexity
- emit feasibility feedback to planner

### 5.3 Invocation Inputs

Required:

```json
{
  "iteration": "int",
  "proposal_artifact": "string",
  "knowledge_model_artifact": "string"
}
```

Optional:

```json
{
  "system_info_artifact": "string|null",
  "research_artifact": "string|null",
  "constraints": {
    "max_implementations": "int|null",
    "validate_build": "bool|null",
    "validate_smoke_test": "bool|null"
  }
}
```

### 5.4 Produced Artifacts

Required:

- `implementation.json`
- `implementation.md`
- `feasibility_report.json`

Optional:

- generated source/build helper files such as `.cu`, `.json`, `.md`

### 5.5 Result Object

```json
{
  "agent": "codegen",
  "iteration": "int",
  "status": "succeeded|failed|partial",
  "reason": "string",
  "artifacts": {
    "implementation_json": "string",
    "implementation_md": "string",
    "feasibility_report": "string"
  },
  "summary": {
    "implementation_count": "int",
    "feasible_count": "int",
    "revision_needed_count": "int",
    "not_feasible_count": "int"
  },
  "error": null
}
```

### 5.6 Downstream Consumers

- runner consumes `implementation.json`
- planner may consume `feasibility_report.json`
- flow controller consumes codegen result object

### 5.7 Invariants

- must not rewrite planner intent
- must preserve mapping from proposal items to implementation items
- feasibility judgment must include implementation complexity
- validation is implementation validation only, not scientific benchmarking
- must produce `feasibility_report.json` on success or partial success

## 6. Runner API

### 6.1 Agent Name

- `runner`

### 6.2 Purpose

- execute implementation artifacts
- preserve raw execution evidence
- package structured execution results for analyzer

### 6.3 Invocation Inputs

Required:

```json
{
  "iteration": "int",
  "implementation_artifact": "string"
}
```

Optional:

```json
{
  "system_info_artifact": "string|null",
  "constraints": {
    "timeout_sec": "float|null",
    "retry_count": "int|null",
    "parallelism": "int|null"
  }
}
```

### 6.4 Produced Artifacts

Required:

- `execution_results.json`
- `execution.md`

Optional or preserved:

- build logs
- stdout/stderr logs
- profiler outputs
- trace files
- metric snapshots
- binaries
- per-item execution directories

### 6.5 Result Object

```json
{
  "agent": "runner",
  "iteration": "int",
  "status": "succeeded|failed|partial",
  "reason": "string",
  "artifacts": {
    "execution_results": "string",
    "execution_md": "string"
  },
  "summary": {
    "item_count": "int",
    "succeeded_count": "int",
    "failed_count": "int",
    "skipped_count": "int",
    "partial_count": "int"
  },
  "error": null
}
```

### 6.6 Downstream Consumers

- analyzer consumes `execution_results.json`
- flow controller consumes runner result object

### 6.7 Invariants

- must preserve raw evidence without silent loss
- must record commands actually executed
- must not generate scientific claims
- must not reinterpret raw evidence
- must produce `execution_results.json` on success or partial success

## 7. Analyzer API

### 7.1 Agent Name

- `analyzer`

### 7.2 Purpose

- interpret execution evidence
- produce claims with provenance
- update knowledge state
- provide planner-facing gap and revision feedback

### 7.3 Invocation Inputs

Required:

```json
{
  "iteration": "int",
  "execution_results_artifact": "string",
  "knowledge_model_artifact": "string",
  "proposal_artifact": "string"
}
```

Optional:

```json
{
  "research_artifact": "string|null",
  "knowledge_base_artifact": "string|null",
  "constraints": {
    "max_claims": "int|null"
  }
}
```

### 7.4 Produced Artifacts

Required:

- `analysis_update.json`
- `analysis.md`

Optional:

- updated knowledge-base artifact
- supporting analysis artifacts

### 7.5 Result Object

```json
{
  "agent": "analyzer",
  "iteration": "int",
  "status": "succeeded|failed|partial",
  "reason": "string",
  "artifacts": {
    "analysis_update": "string",
    "analysis_md": "string",
    "knowledge_base": "string|null"
  },
  "summary": {
    "claim_count": "int",
    "knowledge_update_count": "int",
    "open_gap_count": "int",
    "observability_gap_count": "int"
  },
  "error": null
}
```

### 7.6 Downstream Consumers

- planner consumes analyzer feedback
- flow controller consumes analyzer result object
- knowledge base is updated from analyzer outputs

### 7.7 Invariants

- every claim must reference evidence
- must not overclaim beyond preserved evidence
- must preserve provenance for claims and KB updates
- must distinguish supported vs unresolved vs contradictory knowledge
- must produce planner-facing feedback in `analysis_update.json`

## 8. Flow Controller API

### 8.1 Agent Name

- `flow_controller`

### 8.2 Purpose

- orchestrate the agent sequence
- manage iterations, retries, bounds, and stopping
- preserve workflow history and run-level state

### 8.3 Invocation Inputs

Required:

```json
{
  "intent": "string",
  "run_id": "string|null",
  "constraints": {
    "max_iterations": "int",
    "retry_count": "int",
    "timeout_sec": "float|null"
  }
}
```

Optional:

```json
{
  "knowledge_base_artifact": "string|null",
  "system_info_artifact": "string|null",
  "target_coverage": "float|null"
}
```

### 8.4 Produced Artifacts

Required:

- `run_log.json`

Optional:

- run-level history or state artifact
- final summary artifact such as `run_summary.md`

### 8.5 Result Object

```json
{
  "agent": "flow_controller",
  "status": "succeeded|failed|stopped|partial",
  "reason": "string",
  "artifacts": {
    "run_log": "string",
    "run_summary": "string|null",
    "knowledge_base": "string|null"
  },
  "summary": {
    "iterations_completed": "int",
    "planner_calls": "int",
    "search_calls": "int",
    "codegen_calls": "int",
    "runner_calls": "int",
    "analyzer_calls": "int"
  },
  "error": null
}
```

### 8.6 Downstream Consumers

- top-level CLI
- user-facing orchestration or reporting layer

### 8.7 Invariants

- must not take over domain reasoning
- must preserve workflow history
- must enforce iteration and retry bounds
- must invoke agents according to artifact dependencies
- must make stopping decisions only from explicit policies and agent outputs
