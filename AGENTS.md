# Repository Guidelines

## Purpose and Design Alignment
This repository follows the philosophy in `docs/design_phi.md`: build an autonomous, self-evolving research system rather than a rigid scripted tool.

Contributor rule of thumb:
- preserve autonomy
- preserve learning memory
- preserve traceability
- avoid hard-coding static assumptions unless required for safety

## Agent Quality Bar
The multi-agent framework should behave like a rigorous GPU architect and researcher.
This quality bar applies to every agent and callable capability in the system, including `Book Builder`, `Navigator`, `Researcher`, `Codegen`, and `Analyzer`.

All agents should aim to be:
- knowledgeable in GPU architecture and performance reasoning
- professional in tone and technical judgment
- scientific in method
- rigorous about evidence and uncertainty
- hands-on and concrete about experiments, artifacts, and implementation details
- critical in evaluating assumptions, explanations, and claimed conclusions

This means agents should:
- prefer mechanism-level explanations over superficial summaries
- distinguish clearly between evidence, inference, and speculation
- challenge weak assumptions rather than silently accepting them
- look for confounders, missing controls, and alternative explanations
- produce explanations and benchmarks that are technically actionable
- preserve auditable links between questions, experiments, evidence, and conclusions

## Canonical Shared Artifact: The Book
The central artifact of the system is the knowledge book. All major agent coordination should happen through this book.

The book is:
- the accumulated knowledge base
- the current map of unresolved questions
- the main coordination medium between agents

The book should stay as simple as possible.

### Book Structure
The canonical structure is:
- chapters
- sections within chapters
- questions at the end of each section

Each section should use this minimal shape:

```md
## Chapter N. Title

### N.M Section Title

Summary
Short explanation of the concept or phenomenon.

Mechanism
The causal model or process that currently explains the section.

Evidence
- local artifact or run evidence
- external reference when needed

Current Understanding
The current synthesized conclusion under the known conditions and evidence.

Uncertainty
What remains unclear, weakly supported, or unresolved.

Questions
- Question: Precise unresolved question.
  Why It Matters: Why this question needs to be answered.
  Context: Optional background gathered before execution.
  Answer: Current best answer from local evidence, or "Not answered yet."
  Evidence: Causal evidence and artifact references supporting the answer.
  Resolved: Yes | No
```

### Book Principles
- The book is the canonical representation of understanding.
- The book relies on semantic structure and disciplined agent reasoning rather than rigid formatting rules.
- Knowledge should live in the main section body, not in detached logs or side lists.
- The book must accumulate causally structured knowledge, not a bag of disconnected statements.
- Questions must be section-local, not stored in a separate global todo list as the source of truth.
- Each question must state both the problem itself and why answering it matters.
- `Context` holds supporting background for a question.
- `Answer` holds the current best answer derived from analysis and must explain the causal chain, not merely state an outcome.
- `Evidence` under a question holds the causal evidence and artifact references supporting the current answer.
- `Resolved` indicates whether the question is sufficiently answered for the system to move on.
- When a question is resolved, `Why It Matters`, `Context`, `Answer`, and question-level `Evidence` should be folded back into the main section body. In particular, `Why It Matters` should inform `Summary` and `Current Understanding`, `Context` should inform `Mechanism`, `Evidence`, and `Uncertainty`, and the resolved question should then be removed.
- `Mechanism` should explain the causal model.
- `Current Understanding` should summarize what the system currently concludes under the known conditions.

## Agent Definitions
Agent responsibilities must be sharply separated. Each agent should have a narrow write scope.

### 1. Book Builder
Purpose:
- own the structure and editorial coherence of the book

Responsibilities:
- create the initial book from the user intent
- define initial chapters and sections
- seed initial questions under sections
- run at the end of each iteration to consolidate new knowledge into the main section body
- read answered questions and fold `Why It Matters`, `Context`, causal `Answer`, and question-level `Evidence` back into the section body
- remove resolved questions after consolidation, including their `Why It Matters`, `Context`, `Answer`, question `Evidence`, and `Resolved` fields
- propose and insert new questions when justified by answered questions and the overall book state
- keep the book concise, non-redundant, and structurally coherent

Book Builder writes:
- chapter structure
- section structure
- section text
- question list membership
- any canonical book updates derived from researcher output used during restructuring

Book Builder does not:
- choose the next question to execute
- search online directly; it must use `Researcher` when external knowledge is needed
- generate code
- interpret raw benchmark output directly

Book Builder trigger:
- once at initialization
- once at the end of every iteration

### 2. Navigator
Purpose:
- decide what unresolved question the system should pursue next

Responsibilities:
- read the current book
- identify unresolved section-local questions
- choose the next question to pursue
- decide whether external context is needed before code generation
- hand the selected question to researcher or codegen as appropriate

Navigator writes:
- iteration control state
- selected question reference for the current iteration
- selected question `Context` when navigator uses researcher output for question enrichment

Navigator does not:
- create or restructure the book
- rewrite section knowledge
- answer questions
- generate benchmark code

Navigator selection rule:
- prefer questions that are foundational, clear, and tractable

### 3. Researcher
Purpose:
- provide standalone external research capability when an agent needs outside context

Responsibilities:
- gather external context for a selected question, section, or restructuring need
- find prior methods, terminology, measurement advice, and relevant references
- return research findings to the calling agent

Researcher writes:
- no canonical book fields directly
- only research artifacts or return values consumed by the calling agent

Researcher does not:
- choose the next question
- edit section structure directly
- write `Answer`
- mark a question resolved
- generate benchmark code

Researcher rule:
- researcher is a callable capability, not the owner of the frontier or the book
- researcher never updates the canonical book directly
- the calling agent is responsible for interpreting researcher output and applying any book updates

### 4. Codegen
Purpose:
- turn the selected question into an executable benchmark or probe

Responsibilities:
- read the selected section and selected question
- use the question and its context to generate a concrete implementation
- produce bounded, auditable execution artifacts

Codegen writes:
- implementation artifacts
- benchmark source files
- run commands
- execution configuration artifacts

Codegen does not:
- edit the book
- choose questions
- answer questions
- resolve uncertainty in prose

Codegen rule:
- codegen targets one selected question at a time

### 5. Analyzer
Purpose:
- interpret execution evidence with respect to the selected question

Responsibilities:
- read benchmark outputs and execution artifacts
- determine what the evidence says about the selected question
- write the question's `Answer` as a causal explanation rather than a bare statement
- write the question's causal `Evidence`
- decide whether the question is fully answered and set `Resolved` to `Yes` or `No`
- propose follow-up questions when results expose new unknowns

Analyzer writes:
- selected question `Answer`
- selected question `Evidence`
- selected question `Resolved`
- proposed follow-up questions

Analyzer does not:
- choose the next question
- restructure the book
- generate code
- perform external research

Analyzer rule:
- analyzer answers questions from evidence; Book Builder performs final consolidation and may originate new questions during restructuring

## Agent Ownership Summary
- Book Builder owns structure and consolidation
- Navigator owns next-question selection
- Researcher provides callable external-research support
- Codegen owns executable experiment generation
- Analyzer owns evidence interpretation and provisional answers

## Write Permissions
Write permissions should stay narrow and explicit.

Book Builder may edit:
- chapters
- sections
- all section fields
- question list membership

Navigator may edit:
- iteration control state only
- selected question `Context`

Researcher may edit:
- research artifacts only

Codegen may edit:
- implementation and execution artifacts only

Analyzer may edit:
- selected question `Answer`
- selected question `Evidence`
- selected question `Resolved`
- proposed follow-up questions

## Iteration Flow
The intended loop is:

1. Book Builder initializes the book from the user intent.
2. Navigator reads the book and selects one unresolved question.
3. Navigator may optionally call Researcher to gather context for that question, then interpret the returned research and update the question's `Context` itself.
4. Codegen generates the benchmark or probe for that question.
5. The system executes the generated implementation and preserves artifacts.
6. Analyzer writes `Answer` as a causal explanation, writes question-level causal `Evidence`, decides whether the question is fully answered, sets `Resolved`, and may provide justified follow-up question proposals.
7. Book Builder reads answered questions, consolidates `Why It Matters`, `Context`, causal `Answer`, and question-level `Evidence` back into the section body, especially `Summary`, `Mechanism`, `Current Understanding`, `Evidence`, and `Uncertainty`, removes resolved questions together with their context, answer, evidence, and resolved state, and may propose and add new justified questions based on answered questions, analyzer proposals, or its own restructuring pass over the book. Book Builder may also call Researcher during restructuring and then interpret the returned research to update the book itself.
8. Navigator reads the updated book and selects the next unresolved question.

## Core System Invariants
All contributions must preserve these invariants:
- Closed learning loop: world model -> question -> experiment -> observation -> claim -> update -> next question
- Persistent artifacts with no silent loss of evidence
- Explicit provenance for claims and section updates
- Versioned history
- Reproducible execution
- Bounded safety constraints

## Project Structure Guidance
Organize the project by responsibility:
- `gpu_profiler/runtime/`: orchestration and agent execution
- `gpu_profiler/knowledge/`: knowledge book structure and parsing
- `gpu_profiler/workflow/`: backend logic for book builder, navigator, researcher, codegen, and analyzer
- `tests/`: behavior and regression tests
- `docs/`: design references and supporting specifications

If introducing new directories, keep their boundaries explicit and documented.

## Development and Test Commands
- `python gpu_autoprofile.py run --workload "python -c \"print(123)\""`: fixed profiling smoke path
- `python gpu_autoprofile.py autonomous --intent "Develop a performance model for the local GPU"`: autonomous loop entrypoint
- `pytest -q`: run all tests

For new modules, add equivalent reproducible CLI or test entry points.

## Coding and Testing Standards
- Python, PEP 8, 4-space indentation, `snake_case`
- Prefer small, composable functions with explicit inputs and outputs
- Tests must validate behavior, not only implementation details
- Every claim-producing path should be testable and auditable
- Preserve human-readable artifacts; do not hide key reasoning in opaque state only

## Commit and PR Expectations
- Use imperative commit titles, for example `Add book-builder consolidation flow`
- Keep PRs focused and include:
  - design intent
  - risk and safety impact
  - reproducibility notes
  - `pytest -q` results
