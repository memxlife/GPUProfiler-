# Codegen Agent Specification

Version: 0.1
Status: Draft Codegen Contract
Last Updated: 2026-03-31

## 1. Purpose

The codegen agent transforms planner proposals into concrete implementations, validates whether they are implementable in the current environment, assesses implementation complexity, and provides structured feasibility feedback to the planner.

The codegen agent is not only a code emitter. It is the implementation reality check for planner proposals.

## 2. Design Principles

The codegen agent must follow these principles:

- planner owns proposal direction
- codegen owns implementation generation and implementation feasibility
- codegen validation is implementation validation, not scientific benchmarking
- implementation complexity must be part of feasibility judgment
- generated artifacts must preserve provenance and failure evidence
- output must be structured enough for downstream agents and planner to consume deterministically

## 3. Scope Boundary

### 3.1 Codegen Agent Responsibilities

The codegen agent must:

1. Read planner-authored proposal artifacts.
2. Generate implementation artifacts from the proposal.
3. Generate source files and supporting files.
4. Compile or build generated implementations where applicable.
5. Run minimal smoke tests or debug passes where applicable.
6. Assess implementation feasibility.
7. Assess implementation complexity.
8. Provide structured feedback to planner.
9. Preserve evidence from compile, build, and smoke-test outcomes.

### 3.2 Codegen Agent Non-Responsibilities

The codegen agent must not:

1. Own planning decisions.
2. Define research scope.
3. Execute the full scientific benchmark campaign.
4. Produce final performance conclusions.
5. Control iteration flow.

## 4. Interaction with Planner

The planner owns:

- what should be learned
- what proposal items should be implemented
- why a benchmark is needed
- how the benchmark fits the curriculum

The codegen agent owns:

- whether the proposal can be implemented now
- whether the implementation is too complex for the current stage
- what implementation artifacts should be generated
- what proposal revisions are needed if implementation is blocked or too complex

## 5. Feasibility Model

Feasibility must consider at least these dimensions:

1. Implementability
- can code be generated for the proposal?

2. Buildability
- can the generated code compile or build in the current environment?

3. Minimal validation
- can the implementation pass a smoke test or minimal execution check?

4. Implementation complexity
- is the implementation complexity reasonable for the current proposal stage?

The codegen agent should not judge only possible versus impossible. It should judge whether the proposal is reasonable to implement now.

## 6. MVP Functions

These functions define the minimum viable codegen agent.

### 6.1 Read Planner Proposal

The codegen agent must consume:

- `proposal.json`
- `knowledge_model.json`
- system or tool availability facts

### 6.2 Generate Implementation

The codegen agent must produce implementation artifacts corresponding to proposal items.

### 6.3 Generate Source Files

The codegen agent must generate the benchmark source files and any supporting files needed for implementation validation.

### 6.4 Validate Buildability

The codegen agent must attempt compile or build validation where applicable.

### 6.5 Run Minimal Validation

The codegen agent must perform smoke tests or minimal debug runs where applicable, only to assess implementability.

### 6.6 Assess Feasibility

The codegen agent must assign a feasibility status to each proposal item.

### 6.7 Assess Implementation Complexity

The codegen agent must assess whether implementation complexity is appropriate for the current stage.

### 6.8 Emit Planner Feedback

The codegen agent must provide structured feedback that helps planner revise proposals when needed.

## 7. Advanced Functions

These functions improve codegen quality after MVP is working.

1. Automatic repair of straightforward compile failures.
2. Proposal splitting suggestions for overly complex items.
3. Better smoke-test generation.
4. Multi-file implementation planning.
5. Richer dependency and toolchain diagnostics.

## 8. Optional Functions

These functions are useful later and must not block MVP.

1. Multi-round implementation refinement.
2. Cost estimation for implementation effort.
3. Alternative implementation strategy generation.
4. Deeper automated debugging.

## 9. Inputs

The codegen agent should read:

1. `proposal.json`
2. `knowledge_model.json`
3. system facts summary
4. tool availability summary
5. optional contract constraints

## 10. Outputs

The codegen agent must write:

1. `implementation.json`
2. `implementation.md`
3. `feasibility_report.json`

It may also write generated implementation files such as:

- `.cu`
- `.json`
- `.md`
- optional build helpers

## 11. Feasibility States

At minimum the codegen agent must support:

- `feasible`
- `feasible_with_revision`
- `not_feasible`

## 12. Complexity Levels

At minimum the codegen agent must support:

- `low`
- `medium`
- `high`
- `excessive`

## 13. Typical Feedback Reasons

The codegen agent should be able to report reasons such as:

- ambiguous proposal
- missing prerequisite
- unsupported toolchain or library
- compile failure
- build failure
- smoke-test failure
- observability mismatch
- excessive implementation complexity
- proposal should be simplified
- proposal should be split into smaller steps

## 14. `implementation.json` MVP

This artifact is codegen-authored and should describe the produced implementation for the current proposal items.

Suggested MVP structure:

```json
{
  "proposal_artifact": "string",
  "knowledge_model_artifact": "string",
  "items": [
    {
      "proposal_id": "string",
      "implementation_id": "string",
      "status": "generated|validated|failed",
      "generated_files": [
        {
          "path": "string",
          "type": "cu|json|md|other"
        }
      ],
      "build": {
        "attempted": "bool",
        "succeeded": "bool",
        "command": "string",
        "artifact_refs": ["string"]
      },
      "smoke_test": {
        "attempted": "bool",
        "succeeded": "bool",
        "command": "string",
        "artifact_refs": ["string"]
      },
      "notes": "string"
    }
  ],
  "generated_at": "string"
}
```

## 15. `feasibility_report.json` MVP

This artifact is codegen-authored and should summarize the feasibility judgment for the current proposal items.

Suggested MVP structure:

```json
{
  "proposal_artifact": "string",
  "items": [
    {
      "proposal_id": "string",
      "feasibility_status": "feasible|feasible_with_revision|not_feasible",
      "implementation_complexity": "low|medium|high|excessive",
      "summary": "string",
      "blocking_issues": ["string"],
      "evidence_refs": ["string"],
      "planner_feedback": {
        "revise": "bool",
        "recommended_changes": ["string"]
      }
    }
  ],
  "generated_at": "string"
}
```

## 16. `implementation.md`

`implementation.md` is the human-readable companion to `implementation.json` and `feasibility_report.json`.

It should summarize:

- which proposal items were implemented
- what files were generated
- whether build and smoke validation succeeded
- which items are feasible now
- which items require planner revision

## 17. Success Criteria

The codegen agent is behaving correctly when:

1. It transforms non-executable proposal items into concrete implementations.
2. It validates implementation feasibility rather than assuming it.
3. It includes implementation complexity in its feasibility judgment.
4. It distinguishes implementation validation from full benchmark execution.
5. It returns structured feedback planner can use to revise proposals.

## 18. MVP Implementation Target

The initial implementation should only aim to satisfy:

- Sections 1 through 6
- Section 9
- Section 10
- Section 11
- Section 12
- Section 14
- Section 15
- Section 16

Advanced and optional functions should be deferred until the MVP is stable.
