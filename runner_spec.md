# Runner / Executor Specification

Version: 0.1
Status: Draft Runner Contract
Last Updated: 2026-03-31

## 1. Purpose

The runner executes generated implementations reproducibly and preserves raw execution evidence for downstream analysis.

The runner is not an analysis agent. It is the execution-and-evidence preservation agent.

## 2. Design Principles

The runner must follow these principles:

- implementation direction is upstream-owned
- runner owns execution and evidence preservation
- raw evidence must be preserved without silent loss
- execution provenance must be explicit
- runner may collect raw metrics and profiler outputs, but must not interpret their scientific meaning
- output must be structured enough for analyzer and flow control to consume deterministically

## 3. Scope Boundary

### 3.1 Runner Responsibilities

The runner must:

1. Read implementation artifacts.
2. Prepare per-item execution workspaces.
3. Execute build commands where required.
4. Execute run commands where required.
5. Collect raw execution evidence.
6. Preserve logs, return codes, elapsed times, and output files.
7. Preserve raw profiler, trace, and metric artifacts when produced.
8. Record execution provenance.
9. Return structured execution results.

### 3.2 Runner Non-Responsibilities

The runner must not:

1. Own planning decisions.
2. Define research scope.
3. Generate implementation code.
4. Produce final performance analysis.
5. Generate performance claims.
6. Control iteration flow.

## 4. Interaction with Other Agents

The codegen agent owns:

- implementation generation
- implementation feasibility
- implementation complexity assessment

The runner owns:

- actual execution of validated implementations
- raw evidence capture
- execution result packaging

The analyzer owns:

- interpretation of preserved evidence
- claim generation
- knowledge-base update

## 5. MVP Functions

These functions define the minimum viable runner.

### 5.1 Read Implementation Artifacts

The runner must consume `implementation.json` and the generated files it references.

### 5.2 Prepare Execution Workspace

The runner must prepare per-item directories or workspaces that isolate execution artifacts.

### 5.3 Execute Build Steps

The runner must execute build or compile steps where the implementation requires them.

### 5.4 Execute Run Steps

The runner must execute run steps for the implementation.

### 5.5 Collect Raw Evidence

The runner must capture:

- commands executed
- return codes
- elapsed times
- stdout
- stderr
- declared output file references

### 5.6 Preserve Raw Profiler and Metric Artifacts

When implementations produce profiler, trace, or metric outputs, the runner must preserve them as raw evidence.

### 5.7 Return Structured Execution Results

The runner must emit structured execution results that downstream analyzer and flow controller can consume.

## 6. Advanced Functions

These functions improve runner quality after MVP is working.

1. Richer environment capture.
2. Better isolation and sandboxing per execution item.
3. Retry rules for transient execution failures.
4. Multi-stage execution pipelines per item.
5. Better artifact normalization across tools.

## 7. Optional Functions

These functions are useful later and must not block MVP.

1. Parallel execution policy.
2. Resource reservation or throttling.
3. Advanced timeout and cancellation handling.
4. Artifact compression or export policies.

## 8. Inputs

The runner should read:

1. `implementation.json`
2. generated implementation files
3. system facts summary
4. tool availability summary
5. execution constraints or safety constraints if present

## 9. Outputs

The runner must write:

1. `execution_results.json`
2. `execution.md`

It may also write or preserve raw artifacts such as:

- stdout logs
- stderr logs
- build logs
- generated binaries
- profiler outputs
- trace files
- raw metric snapshots
- per-item execution directories

## 10. Execution States

At minimum the runner must support:

- `succeeded`
- `failed`
- `skipped`
- `partial`

## 11. `execution_results.json` MVP

This artifact is runner-authored and should summarize the execution outcomes and preserved raw evidence.

Suggested MVP structure:

```json
{
  "implementation_artifact": "string",
  "items": [
    {
      "implementation_id": "string",
      "status": "succeeded|failed|skipped|partial",
      "workspace": "string",
      "build": {
        "attempted": "bool",
        "succeeded": "bool",
        "command": "string",
        "returncode": "int|null",
        "elapsed_sec": "float|null",
        "stdout_artifact": "string|null",
        "stderr_artifact": "string|null"
      },
      "run": {
        "attempted": "bool",
        "succeeded": "bool",
        "command": "string",
        "returncode": "int|null",
        "elapsed_sec": "float|null",
        "stdout_artifact": "string|null",
        "stderr_artifact": "string|null"
      },
      "raw_artifacts": [
        {
          "path": "string",
          "kind": "log|binary|trace|profile|metric|other"
        }
      ],
      "notes": "string"
    }
  ],
  "generated_at": "string"
}
```

## 12. `execution.md`

`execution.md` is the human-readable companion to `execution_results.json`.

It should summarize:

- which implementation items were executed
- which build steps succeeded or failed
- which run steps succeeded or failed
- what raw artifacts were preserved
- which items need analyzer attention due to failure or partial execution

## 13. Success Criteria

The runner is behaving correctly when:

1. It executes implementation artifacts reproducibly.
2. It preserves raw execution evidence without silent loss.
3. It records explicit execution provenance.
4. It does not convert raw evidence into scientific conclusions.
5. It returns structured results analyzer can consume directly.

## 14. MVP Implementation Target

The initial implementation should only aim to satisfy:

- Sections 1 through 5
- Section 8
- Section 9
- Section 10
- Section 11
- Section 12

Advanced and optional functions should be deferred until the MVP is stable.
