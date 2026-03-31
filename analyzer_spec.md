# Analyzer Specification

Version: 0.1
Status: Draft Analyzer Contract
Last Updated: 2026-03-31

## 1. Purpose

The analyzer interprets preserved execution evidence, produces structured claims, updates the evolving knowledge base, and provides feedback to planner about remaining gaps, contradictions, and next-step needs.

The analyzer is not an execution agent and not a planner. It is the evidence-to-knowledge agent.

## 2. Design Principles

The analyzer must follow these principles:

- raw evidence is runner-owned; interpretation is analyzer-owned
- claims must be grounded in preserved artifacts
- every claim must preserve provenance
- external research may inform interpretation but does not replace local evidence
- analyzer must distinguish strong evidence from weak or inconclusive evidence
- output must be structured enough for planner and flow control to consume deterministically

## 3. Scope Boundary

### 3.1 Analyzer Responsibilities

The analyzer must:

1. Read execution artifacts and preserved evidence.
2. Interpret execution outcomes and raw results.
3. Produce structured claims tied to evidence.
4. Assess coverage, unresolved gaps, and contradictions.
5. Update the evolving knowledge base.
6. Provide planner-facing feedback about what remains unknown or weakly supported.
7. Preserve claim provenance and confidence.

### 3.2 Analyzer Non-Responsibilities

The analyzer must not:

1. Own planning decisions.
2. Define search scope.
3. Generate implementation code.
4. Execute benchmarks.
5. Control iteration flow.
6. Rewrite evidence to fit a preferred interpretation.

## 4. Interaction with Other Agents

The runner owns:

- execution
- raw artifact preservation
- execution provenance

The analyzer owns:

- interpretation of preserved evidence
- claim generation
- coverage and gap assessment
- knowledge-base update

The planner owns:

- deciding what should be learned next
- deciding whether new research is needed
- producing the next proposal

## 5. Evidence and Claim Model

The analyzer should reason in terms of:

1. Evidence
- raw logs
- profiler outputs
- traces
- metric files
- execution status

2. Claims
- statements supported by available evidence
- each claim must identify supporting artifacts

3. Knowledge state
- what is now supported
- what remains uncertain
- what is contradictory
- what needs follow-up

The analyzer must not overclaim beyond the preserved evidence.

## 6. MVP Functions

These functions define the minimum viable analyzer.

### 6.1 Read Execution Results

The analyzer must consume:

- `execution_results.json`
- relevant raw artifacts referenced by it
- current `knowledge_model.json` if present
- current `proposal.json` if present

### 6.2 Interpret Execution Outcomes

The analyzer must determine what the execution outcomes and raw artifacts imply, within the limits of the evidence.

### 6.3 Produce Claims

The analyzer must emit structured claims with:

- claim text
- related nodes or dimensions
- supporting evidence references
- confidence

### 6.4 Update Knowledge State

The analyzer must update what parts of the domain hierarchy are:

- supported
- partially supported
- still unknown
- contradictory

### 6.5 Identify Remaining Gaps

The analyzer must identify:

- missing evidence
- weakly supported areas
- contradictions
- observability gaps

### 6.6 Provide Planner Feedback

The analyzer must provide structured planner feedback about:

- what needs refinement
- what needs follow-up
- what evidence was insufficient
- what proposal directions are blocked or incomplete

## 7. Advanced Functions

These functions improve analyzer quality after MVP is working.

1. Contradiction detection across runs.
2. Distinguishing competing hypotheses.
3. Better confidence calibration.
4. Structural coverage scoring over the knowledge hierarchy.
5. Better observability-gap diagnostics.

## 8. Optional Functions

These functions are useful later and must not block MVP.

1. Cross-run trend synthesis.
2. Automatic anomaly clustering.
3. Counterfactual or alternative-interpretation reporting.
4. Richer uncertainty decomposition.

## 9. Inputs

The analyzer should read:

1. `execution_results.json`
2. raw artifacts referenced by execution results
3. `knowledge_model.json`
4. `proposal.json`
5. optional `research.json`
6. optional contract constraints

## 10. Outputs

The analyzer must write:

1. `analysis_update.json`
2. `analysis.md`
3. updated knowledge-base artifact

It may also write supporting analysis artifacts if needed.

## 11. Claim Confidence Levels

At minimum the analyzer must support:

- `low`
- `medium`
- `high`

## 12. Knowledge Status Values

At minimum the analyzer must be able to update or report:

- `unknown`
- `hypothesized`
- `partially_supported`
- `well_supported`
- `contradictory`

## 13. `analysis_update.json` MVP

This artifact is analyzer-authored and should summarize the new claims, updated knowledge state, and planner-relevant follow-up information.

Suggested MVP structure:

```json
{
  "execution_artifact": "string",
  "claims": [
    {
      "id": "string",
      "claim": "string",
      "related_nodes": ["string"],
      "confidence": "low|medium|high",
      "evidence_refs": ["string"],
      "notes": "string"
    }
  ],
  "knowledge_updates": [
    {
      "node_id": "string",
      "new_status": "unknown|hypothesized|partially_supported|well_supported|contradictory",
      "reason": "string",
      "evidence_refs": ["string"]
    }
  ],
  "open_gaps": ["string"],
  "observability_gaps": ["string"],
  "planner_feedback": {
    "next_focus_suggestions": ["string"],
    "proposal_revision_suggestions": ["string"],
    "research_needed": ["string"]
  },
  "generated_at": "string"
}
```

## 14. `analysis.md`

`analysis.md` is the human-readable companion to `analysis_update.json`.

It should summarize:

- what execution evidence was interpreted
- what claims were supported
- what remains weak or unresolved
- what contradictions were observed
- what the planner should pay attention to next

## 15. Knowledge-Base Update Rule

The analyzer should update the knowledge base by:

- appending new claims
- updating node statuses where justified
- preserving prior history rather than silently overwriting it
- recording evidence references for all meaningful updates

## 16. Success Criteria

The analyzer is behaving correctly when:

1. It produces claims grounded in preserved evidence.
2. It preserves provenance for claims and updates.
3. It does not overclaim beyond the available evidence.
4. It updates the knowledge state in a structured way.
5. It provides planner-usable feedback about remaining gaps and next needs.

## 17. MVP Implementation Target

The initial implementation should only aim to satisfy:

- Sections 1 through 6
- Section 9
- Section 10
- Section 11
- Section 12
- Section 13
- Section 14
- Section 15

Advanced and optional functions should be deferred until the MVP is stable.
