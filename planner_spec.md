# Planner Specification

Version: 0.3
Status: Finalized Planner Contract
Last Updated: 2026-03-31

## 1. Purpose

The planner is responsible for planning only.

Its job is to read the evolving local knowledge model for the user's intent, decide what must be learned next, issue research requests when external knowledge is needed, and produce the next non-executable proposal that improves that knowledge.

The planner does not control the iteration loop, execute experiments, generate benchmark code, or regenerate the knowledge model from scratch.

## 2. Design Principles

The planner must follow these principles:

- planner implementation is generic
- planner output is domain-specific
- the persistent knowledge model is local state built incrementally across iterations, not regenerated wholesale by the planner
- internet search execution is delegated, but research need is planner-decided
- proposals must follow a curriculum from basic to comprehensive
- proposals must contain no executable code
- every proposal must preserve rationale and provenance

## 3. Scope Boundary

### 3.1 Planner Responsibilities

The planner must:

1. Interpret user intent.
2. Read existing knowledge artifacts.
3. Read the current local knowledge model.
4. Identify immediate knowledge gaps.
5. Issue research requests when external knowledge is needed.
6. Produce the next-step proposal using the current knowledge model plus returned research.
7. Preserve rationale and provenance.

### 3.2 Planner Non-Responsibilities

The planner must not:

1. Control iteration flow.
2. Decide retries or scheduling.
3. Generate executable code.
4. Produce benchmark commands.
5. Execute experiments.
6. Collect profiler outputs.
7. Perform result analysis.
8. Update final performance claims directly.
9. Hardcode domain taxonomy in planner code.
10. Perform internet search itself.
11. Regenerate the full knowledge model from scratch every iteration.

## 4. Role of Research Input

Internet-searched knowledge must be produced by a separate search agent.

The planner first decides what should be researched by issuing a structured research request. After search completes, the planner consumes the returned research results alongside the existing local knowledge model and recent analysis feedback to produce a proposal.

Research can influence:

- knowledge-gap clarification
- hypothesis formation
- benchmark idea selection
- evidence requirements
- prioritization

Research must not be treated as proof of local hardware behavior. Only local observations can support final local claims.

The planner therefore owns research direction, but not research execution.

## 5. Knowledge Structure Model

The planner's central responsibility is to reason over the evolving knowledge structure of the target domain.

This structure must be:

- top-down: derived from the user's intent into a domain hierarchy
- bottom-up: updated from accumulated evidence, prior proposals, and analysis gaps

The knowledge model is persistent local state. It is updated incrementally by later analysis and knowledge-base update logic. The planner reads that model and uses it to decide what should be learned next.

The planner does not primarily manage a list of tasks. It reasons over an evolving knowledge model.

## 6. Curriculum Policy

The planner must follow curriculum learning.

For each feature or topic, planning should start from the simplest useful test and then expand only after basic evidence is obtained.

The expected progression is:

1. baseline
2. isolation
3. sensitivity sweep
4. interaction
5. stress or boundary
6. refinement or validation

The planner should therefore prefer:

- simple before complex
- isolated before entangled
- evidence-backed expansion before speculative expansion

## 7. MVP Functions

These functions define the minimum viable planner.

### 7.1 Interpret Intent

The planner must read the user's high-level objective and restate it as a planning goal.

### 7.2 Read Existing Knowledge

The planner must consume:

- current knowledge model if present
- prior proposals
- prior observation summaries
- prior analysis gaps
- system facts summary
- research artifacts

### 7.3 Read Current Knowledge State

The planner must consume the current local knowledge model as persistent state.

The knowledge model:

- is versioned local state
- is updated incrementally by later analysis and knowledge-base update logic
- must not be regenerated wholesale by the planner on each iteration

### 7.4 Identify Immediate Knowledge Gaps

The planner must identify the next important missing, weakly supported, or contradictory areas that deserve follow-up.

### 7.5 Issue Research Requests

When external knowledge is needed, the planner must emit a structured `research_request.json` for the search agent.

This request should specify:

- target nodes
- target questions
- search topics
- preferred source types
- expected outputs

### 7.6 Produce Next-Step Proposal

After search results are available, the planner must produce a non-executable proposal for the next iteration.

Proposal generation must use:

- the current knowledge model
- the returned research artifacts
- prior analysis gaps
- curriculum constraints

The proposal must be curriculum-first and biased toward the simplest useful next step.

### 7.7 Preserve Rationale and Provenance

The planner must record:

- why a proposal exists
- what prior knowledge it depends on
- what gap it addresses
- what evidence it expects

## 8. Advanced Functions

These functions improve quality after MVP is working.

1. Curriculum staging per feature.
2. Prerequisite and promotion logic.
3. Structural coverage tracking.
4. Proposal prioritization.
5. Competing hypothesis management.
6. Selective research-request generation based on current uncertainty.

These are important but not required for initial implementation.

## 9. Optional Functions

These functions are useful later and must not block MVP.

1. Contradiction detection.
2. Long-horizon roadmap generation.
3. Planner self-critique.
4. Multi-branch planning under uncertainty.
5. Contract negotiation support with other agents.

## 10. Inputs

The planner should read the following inputs:

- `user_intent`
- current `knowledge_model.json`
- prior proposals
- prior observations or result summaries
- research artifacts
- system facts summary
- analyzer-provided gaps or unresolved questions

## 11. Outputs

The planner must write these artifacts:

1. `research_request.json` when external research is needed
2. `proposal.json`
3. `proposal.md`

These artifacts define the planner boundary.

The planner must not emit executable commands, benchmark source code, profiler invocations, or a full replacement knowledge model.

## 12. `proposal.json` MVP

The MVP `proposal.json` must describe the next non-executable proposal derived from the current knowledge model plus relevant research.

```json
{
  "intent_summary": "string",
  "proposal_summary": "string",
  "target_nodes": ["string"],
  "proposals": [
    {
      "id": "string",
      "title": "string",
      "objective": "string",
      "target_node_ids": ["string"],
      "priority": "high|medium|low",
      "benchmark_role": "baseline|isolation|sweep|interaction|stress|refinement|validation",
      "description": "string",
      "hypothesis": "string",
      "required_evidence": ["string"],
      "rationale": "string",
      "prerequisites": ["string"],
      "next_if_success": ["string"],
      "next_if_failure": ["string"]
    }
  ],
  "planner_notes": "string",
  "generated_at": "string"
}
```

## 13. `research_request.json` MVP

```json
{
  "intent_summary": "string",
  "request_summary": "string",
  "target_nodes": ["string"],
  "target_questions": ["string"],
  "search_topics": ["string"],
  "source_preferences": ["vendor_doc", "official_tool_doc", "paper", "article"],
  "source_constraints": ["string"],
  "expected_outputs": ["string"],
  "notes": "string"
}
```

## 14. Success Criteria

The planner is behaving correctly when:

1. It reads and reasons over the current local knowledge model.
2. It identifies the next important knowledge gaps.
3. It emits a curriculum-first, non-executable proposal.
4. It issues research requests when external knowledge is needed.
5. It preserves rationale and provenance.
6. It does not generate executable benchmark content.
7. It does not regenerate the knowledge model wholesale.
