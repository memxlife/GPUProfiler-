# GPU Architecture Research Agent: Build Prompt

Build a local-first autonomous research system for single-GPU architecture characterization and performance investigation. The system must run on one machine, preserve all research artifacts, and generate reproducible, evidence-backed claims about GPU behavior. Its outputs must be useful both to humans and to downstream coding or planning agents.

The system is not a one-shot benchmark runner. It is a bounded research loop:

1. Maintain a world model / knowledge base.
2. Identify the next uncertainty or research question.
3. Plan a bounded experiment.
4. Execute the experiment reproducibly.
5. Record raw observations and derived metrics.
6. Produce explicit claims with provenance and confidence.
7. Update the knowledge base.
8. Decide the next question.

## Primary Goal

Produce reproducible, traceable claims about GPU architecture and behavior, and accumulate them into a structured knowledge base.

## Secondary Goal

Use those claims to support better kernel design, scheduling decisions, operator decomposition, and optimization strategy.

## Tertiary Goal

Improve the research process itself, but only in controlled, reviewable ways. The system may propose workflow or benchmark improvements, but it must not silently mutate core safety, provenance, or execution rules.

## Scope

The initial system targets one GPU on one machine. It should support research into:

- Memory hierarchy
- Cache and shared memory behavior
- Warp scheduling and latency hiding
- Occupancy and register pressure
- Tensor core behavior
- Global memory bandwidth and latency
- Instruction throughput and issue behavior
- Observable single-card topology / interconnect characteristics
- Numerics-related behavior, including low-precision modes, when justified by prior evidence
- High-level operator behavior only when tied back to measurable low-level mechanisms

The first milestone should focus on microbenchmarks and architecture characterization, not full end-to-end kernel autotuning.

## Required Invariants

The system must preserve these invariants:

- Closed research loop: model -> question -> experiment -> observation -> claim -> update -> next question
- Persistent artifacts: no silent loss of data, logs, code, prompts, or outputs
- Explicit provenance: every claim must cite its evidence
- Versioned history: changes to the knowledge base and conclusions must be traceable over time
- Reproducible execution: experiments must be rerunnable from stored specs
- Bounded autonomy: all execution must stay within explicit safety constraints
- Human interruptibility: a human must be able to inspect, pause, veto, or redirect the system at any time

## Artifact Requirements

The system must persist at least these artifact types:

- Research question
- Experiment plan
- Benchmark or probe implementation
- Execution configuration
- Raw execution results
- Derived metrics
- Analysis summary
- Claims
- Knowledge base snapshot
- Run log / event history

Each claim must include:

- Claim text
- Claim type: observation, inference, or hypothesis
- Confidence
- Supporting evidence artifact paths or IDs
- Method summary
- Relevant hardware / software context
- Timestamp
- Status: active, superseded, or invalidated

The knowledge base must distinguish:

- Raw observations
- Derived measurements
- Hypotheses
- Validated claims
- Open questions
- Contradictions / invalidations

## Execution Model

The orchestration layer should be generic and deterministic where possible. Domain content such as benchmark design, analysis framing, and question generation may be agent-assisted, but must always be materialized into explicit artifacts before execution.

The system should prefer:

- Local tools first
- Deterministic parsing where possible
- Thin machine-readable sidecars for orchestration
- Human-readable memos for agent handoff and auditability

## Safety Constraints

The system must not:

- Silently overwrite or discard previous evidence
- Make unsupported claims without linking evidence
- Treat placeholder or skipped runs as successful research evidence
- Modify core orchestration logic autonomously without explicit human approval
- Expand beyond the single-machine single-GPU scope unless explicitly directed

## Acceptance Criteria

A successful first version should be able to:

- Run an end-to-end research iteration on one GPU
- Produce at least one reproducible microbenchmark result
- Generate at least one evidence-backed claim with clear provenance
- Update a persistent knowledge base
- Preserve a complete audit trail for the iteration
- Continue to a reasonable next question based on current uncertainty
