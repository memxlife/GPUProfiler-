# State Summary

## Current Branch State

This repo is now at a documented checkpoint for the multi-agent autonomous profiling workflow.

Recent commits of interest:

1. `781dc00` `Split planner into research and proposal phases`
2. `f1994de` `Use memo-first planner and search handoffs`

These sit on top of earlier spec and workflow reconciliation commits already on `main`.

## Current Agent Flow

The current autonomous flow is:

1. `collect_system_info`
2. `llm_schema_contract`
3. `llm_plan_research`
4. `llm_research`
5. `llm_plan_proposal`
6. `llm_generate_implementation`
7. `execute_implementation`
8. `llm_analyze_update`
9. `autonomous_report`

Key architectural decisions now implemented:

- planner is split into research-request planning and proposal planning
- persistent `knowledge_model.json` lives at the run root
- analyzer updates the knowledge model incrementally
- planner/search/codegen handoffs are moving to memo-first `.md` artifacts
- thin JSON sidecars remain for orchestration

## Current Artifact Model

Run root:

- `performance_model.json`
- `knowledge_model.json`
- `run_log.json`
- `autonomous_report.md`

Per iteration:

- `schema_contract.json`
- `schema_contract.md`
- `research_request.md`
- `research_request.json`
- `research.json`
- `research.md`
- `proposal.json`
- `proposal.md`
- `implementation.json`
- `implementation.md`
- `feasibility_report.json`
- `execution_results.json`
- `execution.md`
- `analysis_update.json`
- `analysis.md`
- `knowledge_model.json` (iteration snapshot)

## What Is Working

1. Structural orchestration works end to end in heuristic mode.
2. Planner split is live and validated.
3. Schema contract now matches split planner outputs:
   - `research_request_output`
   - `proposal_output`
4. Persistent root `knowledge_model.json` exists and is updated through analysis.
5. Planner/search/codegen memo artifacts are emitted:
   - `research_request.md`
   - `proposal.md`
6. Tests currently pass:
   - `pytest -q` -> `12 passed`

## What Is Still Not Working Well

1. OpenAI path is still unreliable on this machine/network.
   - direct minimal `responses.create(...)` probes hit `APIConnectionError('Connection error.')`
   - orchestrated OpenAI runs are partially successful but still often time out in:
     - `llm_plan_research`
     - `llm_research`
     - `llm_plan_proposal`
     - `llm_generate_implementation`

2. Real benchmark generation is still not reliable.
   - fallback implementation often produces placeholder `SKIP` workloads
   - that means runs can complete structurally without producing real profiling evidence

3. Content-first agent communication is only partially implemented.
   - planner/search/codegen memo handoff is started
   - prompts are still too JSON-heavy internally for some OpenAI stages

## Important Current Diagnosis

There are two separate failure classes:

1. direct OpenAI transport instability on this machine
2. prompt/task latency in heavier orchestrated stages when OpenAI requests do get through

So testing on another machine with a different network is useful and likely informative.

## Known Good Validation Commands

Basic tests:

```bash
python -m py_compile gpu_profiler/*.py test_gpu_autoprofile.py
pytest -q
```

Heuristic autonomous smoke run:

```bash
python gpu_autoprofile.py autonomous \
  --intent "Develop a performance model for the local GPU" \
  --out profiling_runs \
  --samples 1 \
  --interval 0 \
  --max-iterations 1 \
  --max-benchmarks 1 \
  --target-coverage 1.0 \
  --planner-backend heuristic \
  --retries 1 \
  --retry-delay 0
```

OpenAI autonomous smoke run:

```bash
source ./set_key.sh
python gpu_autoprofile.py autonomous \
  --intent "Develop a performance model for the local GPU" \
  --planner-backend openai \
  --out profiling_runs \
  --samples 1 \
  --interval 0 \
  --max-iterations 1 \
  --max-benchmarks 1 \
  --target-coverage 1.0 \
  --retries 1 \
  --retry-delay 0
```

## What To Inspect On Another Machine

If you test elsewhere, inspect these first:

1. `run_log.json`
   - whether `llm_plan_research` succeeds with OpenAI
   - whether `llm_research` succeeds with OpenAI
   - whether `llm_plan_proposal` succeeds with OpenAI
   - whether `llm_generate_implementation` succeeds with OpenAI

2. `research_request.md`
3. `proposal.md`
4. `implementation.md`
5. `execution_results.json`
6. `analysis_update.json`

Most important outcome:

- whether the OpenAI path produces a non-placeholder implementation and a non-skipped execution

## Recommended Next Engineering Steps

If the other machine still shows the same issues, the next changes should be:

1. make planner/search/codegen prompts question-first and memo-first, not schema-first
2. parse memo outputs into thin sidecar JSON deterministically in Python
3. tighten acceptance so placeholder implementations cannot pass as feasible
4. add a small standalone OpenAI diagnostics script for transport debugging

