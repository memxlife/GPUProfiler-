---
name: book-builder
description: Use this skill when initializing or restructuring the canonical GPU knowledge book, especially when a run needs the first textbook skeleton, resolved questions must be consolidated into section knowledge, or section-local follow-up questions need to be added while keeping the book causally structured and profiling-driven.
---

# Book Builder

## Overview

This skill authors and maintains the canonical GPU knowledge book as one markdown textbook. It is the editorial authority over book structure, section prose, and question lifecycle.

## When To Use

Use this skill when the task is to:

- create the first `knowledge_book.md` from the user intent
- reorganize the book around better microarchitecture mechanisms or dependencies
- fold resolved questions back into section `Summary`, `Mechanism`, `Evidence`, `Current Understanding`, and `Uncertainty`
- remove resolved questions after their content has been absorbed
- add new section-local questions that arise from answered questions or restructuring
- keep the book focused on profiling-driven, experimentally answerable GPU microarchitecture questions

Do not use this skill for:

- selecting the next question to run
- searching online directly when another agent owns research
- generating benchmark code
- deciding empirical answers from raw execution results without analyzer output

## Book Contract

The book is:

- one canonical markdown textbook
- causally structured knowledge, not a bag of claims
- organized as chapters, then sections, then section-local questions
- semantic and human-readable first

Each section should contain:

- `Summary`
- `Mechanism`
- `Evidence`
- `Current Understanding`
- `Uncertainty`
- `Questions`

Each question should clearly carry:

- `Question`
- `Why It Matters`
- `Context`
- `Answer`
- `Evidence`
- `Resolved`

Exact markdown syntax may vary. Preserve the semantic role of each field.

## Section Design Rules

Each section should be built around one mechanism, not one topic label.

Good section centers:

- instruction issue throughput for a particular class of work
- warp scheduling and latency hiding under specific readiness conditions
- global-memory throughput under controlled access patterns
- cache or shared-memory locality transitions
- occupancy limits induced by a specific resource
- synchronization costs under specific contention patterns

Weak section centers:

- "compute"
- "memory"
- "optimization"
- "performance tips"

If a proposed section contains multiple distinct mechanisms, split it.
If a proposed section is too thin to support multiple meaningful questions, merge it into a neighboring mechanism.

## Working Modes

### Initialize

Use initialize mode to create the first book from the user intent.

Workflow:

1. Read the normalized user intent.
2. First perform a structure pass.
3. In the structure pass, use the skill and the user intent to author the initial book chapters and sections that cover the important NVIDIA GPU microarchitecture-performance mechanisms.
4. In the structure pass, populate each section with substantive `Summary`, `Mechanism`, `Evidence`, `Current Understanding`, and `Uncertainty` prose.
5. Do not finalize section-local question lists during the structure pass.
6. After the structure pass is complete, perform a question pass.
7. In the question pass, read the finished initial book section by section.
8. For each section, generate and attach a strong section-local question list under that section's `Questions`.
9. Prefer mechanisms that are foundational and experimentally answerable on the local GPU.

Initialization rules:

- initialize mode should be handled as two separate editorial passes: structure first, questions second
- the structure pass should focus on building the right textbook decomposition for NVIDIA GPU microarchitecture performance
- the structure pass should cover the important mechanism families, dependencies, and explanatory scope before frontier generation begins
- after the structure pass, the book should already cover the necessary GPU microarchitecture-performance topics; major mechanism families should not be missing
- if an important mechanism family is absent after the structure pass, initialization is not good enough yet
- the initial book should contain substantive section text from the start; do not leave placeholder prose such as "No summary recorded yet"
- the structure pass should not confuse semantic field names with document hierarchy
- `Summary`, `Mechanism`, `Evidence`, `Current Understanding`, and `Uncertainty` are section fields, not section headings in the chapter hierarchy
- the question pass should read each completed section in context and then generate the section-local frontier for that section
- question generation should happen after the section exists, not while the section is still being invented
- prioritize execution pipelines, warp scheduling, latency hiding, memory hierarchy, locality, occupancy, and synchronization
- avoid broad textbook filler
- try to produce an exhaustive initial question list for each section, covering the important unresolved questions that the mechanism naturally raises
- make the initial questions detailed and technically specific enough that they imply concrete benchmark or profiling designs
- make the initial frontier comprehensive enough that a serious microarchitecture researcher could immediately see the main unknowns, discriminative measurements, and likely experiment families for that section
- do not force one question per section
- richer mechanisms should usually have multiple questions
- be exhaustive without padding; include all high-value questions you can justify, but do not add weak or generic filler questions

Initialization quality bar:

- a strong initial book should make a knowledgeable reader feel that the main microarchitectural unknowns are already mapped
- the first frontier should look like a real research agenda, not a generic outline
- most sections should contain several questions, not one
- each question list should expose both measurement targets and explanation gaps
- the book structure should still make sense even before the question pass is added
- the question pass should improve an already coherent textbook, not rescue a bad structure
- the structure pass should already give a knowledgeable reader confidence that the main microarchitecture topic coverage is present, even before any questions are added

Minimum structure-pass coverage expectation:

- execution pipelines and instruction throughput
- dependency latency and instruction-level parallelism
- warp scheduling and latency hiding
- occupancy and resource constraints
- global-memory throughput and latency
- coalescing and transaction formation
- cache hierarchy behavior
- shared-memory behavior and bank conflicts
- synchronization and communication costs
- profiling methodology and observability

### Consolidate

Use consolidate mode at the end of an iteration after analyzer output exists.

Workflow:

1. Read the current book and the latest answered questions.
2. Promote `Why It Matters`, `Context`, `Answer`, and question-level `Evidence` into the main section body.
3. Update `Summary`, `Mechanism`, `Evidence`, `Current Understanding`, and `Uncertainty` accordingly.
4. Remove resolved questions once their content has been absorbed.
5. Add new section-local follow-up questions when needed.
6. Keep the book concise, coherent, and causally organized.

Consolidation rules:

- resolved questions should disappear after their content is integrated
- unanswered questions remain section-local
- new questions should be attached to the most relevant section
- if a section has become overloaded or mis-scoped, restructure it

## Question Writing Rules

Section-local questions should be:

- causally meaningful
- profiling-driven
- locally benchmarkable
- foundational enough to support later sections
- clear about why they matter
- detailed enough to constrain what kind of experiment, profiler evidence, or benchmark intervention would answer them
- specific about the variable, regime, transition, or competing explanation being investigated
- comprehensive enough that the section's main unresolved causal subproblems are visible up front
- phrased at the level of measurable microarchitectural behavior, not vague performance intuition

Each question should ideally make clear at least one of:

- the manipulated variable
- the measured quantity
- the expected transition or threshold
- the competing mechanisms to separate
- the profiler signal or benchmark evidence needed for discrimination

Avoid questions like:

- "How does this mechanism affect performance?"
- "What is the role of memory here?"
- "How important is occupancy?"

Prefer questions like:

- which variable is being varied
- which ceiling, regime boundary, or transition is being estimated
- which competing mechanisms must be separated
- which profiler signals or microbenchmark controls are needed to disambiguate the explanation

Prefer wording such as:

- "How does achieved throughput change as X varies under Y control condition?"
- "At what point does the regime shift from A-limited to B-limited behavior?"
- "Which counters or traces distinguish explanation A from explanation B?"
- "Under what conditions does mechanism A stop being the dominant limiter?"
- "What experimental control is required so measurement Z is reproducible and interpretable?"

When a mechanism is rich, questions should cover different needs such as:

- baseline quantitative characterization
- regime boundaries or transition points
- confounders or competing explanations
- interaction with another mechanism
- measurement methodology needed to resolve uncertainty
- profiler observables that discriminate between mechanisms
- sensitivity to launch configuration, access pattern, dependency structure, or resource pressure
- conditions under which the mechanism stops being the dominant limiter

In initialize mode, aim for broad frontier coverage within each section:

- enumerate the major unresolved questions that would matter to a rigorous microarchitecture study of that mechanism
- include multiple questions when different causal subproblems must be separated to understand the section properly
- do not stop after the first plausible question if the section clearly contains other foundational unknowns
- prefer questions that separate baseline characterization, regime boundaries, confounders, discriminative measurements, and mechanism interactions
- make sure the section's question list covers both "what is the quantitative behavior?" and "what evidence would distinguish why that behavior occurs?"

Minimum initialization checklist for a strong section:

- one question for baseline quantitative characterization
- one question for a regime boundary, transition, or threshold
- one question for discriminating between competing explanations or confounders
- one question for the most important interaction with another mechanism, if relevant
- one question for methodology or observability, if the mechanism is difficult to measure cleanly

Not every section needs every category, but the skill should check them explicitly before finalizing the initial book.

Question-pass rule:

- during initialize mode, generate questions only after the structure pass has produced a coherent chapter/section book
- when generating questions for a section, read that section's prose first and make the questions match its mechanism and uncertainty
- do not generate a flat global question bag and then force it into sections afterward

## Good Question Patterns

Use patterns like these when initializing the book:

- quantitative ceiling questions
- latency-versus-throughput regime questions
- threshold or breakpoint questions
- counterfactual or competing-mechanism questions
- observability and measurement-design questions
- interaction questions between two mechanisms

Avoid:

- vague restatements of the section title
- questions that can only be answered by generic prose
- questions whose answer would not change later planning
- near-duplicate questions that differ only in wording

## Editorial Rules

- The book is the source of truth.
- Keep one coherent book, not multiple parallel summaries.
- Explanations must be causal and evidence-aware.
- Preserve clear dependency structure across chapters and sections.
- Prefer microarchitecture-level mechanisms over generic GPU-performance language.
- Maintain a professional, rigorous, scientific tone.
- Do not turn the book into a to-do list or note dump.
- In initialize mode, optimize for a frontier map that is detailed enough to guide many iterations before the section needs to be rethought.
