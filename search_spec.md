# Search Agent Specification

Version: 0.1
Status: Finalized Search Agent Contract
Last Updated: 2026-03-31

## 1. Purpose

The search agent executes planner-issued external research requests and returns structured, source-grounded findings for planner consumption.

The search agent does not decide what should be researched. It performs search and retrieval requested by the planner.

## 2. Design Principles

The search agent must follow these principles:

- research direction is planner-owned
- search execution is search-agent-owned
- external knowledge must preserve provenance
- findings must remain scoped to the planner request
- external findings are guidance, not proof of local hardware behavior
- output must be structured enough for the planner to consume deterministically

## 3. Scope Boundary

### 3.1 Search Agent Responsibilities

The search agent must:

1. Read planner-issued `research_request.json`.
2. Search the internet according to planner-defined scope.
3. Retrieve relevant external sources.
4. Filter and summarize useful findings.
5. Preserve provenance for findings.
6. Map findings back to planner-requested nodes and questions.
7. Return planner-consumable research artifacts.

### 3.2 Search Agent Non-Responsibilities

The search agent must not:

1. Define its own research agenda.
2. Decide what should be searched.
3. Own planning decisions.
4. Generate benchmark proposals independently.
5. Generate executable code.
6. Execute benchmarks.
7. Analyze local benchmark results.
8. Produce final local performance claims.

## 4. Interaction with Planner

The planner owns:

- what knowledge gaps require external research
- what target nodes need support
- what questions should be answered
- what kinds of sources are preferred
- what outputs are needed for planning

The search agent owns:

- executing the search
- retrieving sources
- summarizing findings
- preserving provenance
- returning planner-consumable research artifacts

## 5. MVP Functions

These functions define the minimum viable search agent.

### 5.1 Read Planner Request

The search agent must consume `research_request.json` as its main driver.

### 5.2 Execute Scoped Search

The search agent must search the internet according to the planner-defined topics, target questions, and source preferences.

### 5.3 Retrieve and Filter Sources

The search agent should prefer high-signal technical sources within the requested scope.

### 5.4 Summarize Requested Knowledge

The search agent must summarize findings in a planner-consumable way, staying within the requested scope.

### 5.5 Preserve Provenance

Every important finding must remain traceable to its source.

### 5.6 Return Structured Research Results

The search agent must produce structured artifacts the planner can read directly.

## 6. Advanced Functions

These functions improve research quality after MVP is working.

1. Source ranking.
2. Conflict identification across sources.
3. Repeated-search avoidance.
4. Scoped search refinement.
5. Clustering findings by target node and question.

## 7. Optional Functions

These functions are useful later and must not block MVP.

1. Multi-round retrieval.
2. Contradiction reports.
3. Terminology normalization.
4. Evidence strength scoring.

## 8. Inputs

The search agent should read:

1. `research_request.json`
2. optionally prior `research.json`
3. optionally current `knowledge_model.json`
4. optionally system facts summary

The main required input is `research_request.json`.

## 9. Outputs

The search agent must write:

1. `research.json`
2. `research.md`

These artifacts define the search-agent boundary.

## 10. `research_request.json` MVP

This artifact is planner-authored and is the primary input to the search agent.

Suggested MVP structure:

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

## 11. `research.json` MVP

This artifact is search-agent-authored and should summarize the results of the requested search.

Suggested MVP structure:

```json
{
  "intent_summary": "string",
  "request_summary": "string",
  "findings": [
    {
      "id": "string",
      "title": "string",
      "source_url": "string",
      "source_type": "vendor_doc|official_tool_doc|paper|article|forum|other",
      "summary": "string",
      "relevance": "string",
      "related_nodes": ["string"],
      "answers_questions": ["string"],
      "extracted_knowledge": ["string"],
      "confidence": "low|medium|high"
    }
  ],
  "unanswered_questions": ["string"],
  "notes": "string",
  "generated_at": "string"
}
```

## 12. `research.md`

`research.md` is the human-readable companion to `research.json`.

It should summarize:

- the planner request being answered
- the main findings
- the most relevant sources
- which questions were answered
- which questions remain unresolved

## 13. Success Criteria

The search agent is behaving correctly when:

1. It stays within planner-requested scope.
2. It preserves provenance for findings.
3. It returns structured outputs planner can consume directly.
4. It does not substitute its own planning agenda for the planner's request.
5. It treats external knowledge as guidance rather than proof of local behavior.

## 14. MVP Implementation Target

The initial implementation should only aim to satisfy:

- Sections 1 through 5
- Section 8
- Section 9
- Section 10
- Section 11
- Section 12

Advanced and optional functions should be deferred until the MVP is stable.
