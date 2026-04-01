# Knowledge Base Template

This file defines the preferred template for the canonical textbook-style GPU knowledge base.

## Design Intent

The knowledge base should read like a living technical textbook:

- hierarchical
- precise
- concise
- evidence-aware
- quantitatively oriented
- explicit about frontier questions

The primary representation is markdown. Any graph or index should be derived from the markdown structure, not replace it.

## Top-Level Structure

Use this hierarchy:

- `Part`
- `Chapter`
- `Section`

Recommended heading levels:

- `## Part X. ...`
- `### Chapter X. ...`
- `#### X.Y Section Name`

## Required Section Template

Each `####` section should use the following fields in order.

```md
#### X.Y Section Name

Summary  
One short paragraph stating what this section is about.

Mechanism  
One short paragraph describing the causal or structural mechanism.

Quantitative Understanding  
State what is known numerically, with units and conditions when available. If unknown, say so directly.

Local Findings  
Concise narrative of local evidence gained so far for this section. If there are no accepted local findings yet, say so directly.

Evidence  
- artifact or source reference
- artifact or source reference

Open Questions  
- unresolved question
- unresolved question

Cross References  
- [X.Y Related Section]
- [X.Y Related Section]

Status  
Known | Frontier | Unknown | Contradicted

Prerequisites  
- lower-level section required before this section can advance confidently

Frontier Criteria  
- evidence condition required to move this section upward
```

## Writing Rules

Summary:
- concise and technically precise
- no filler or motivational language

Mechanism:
- explain how performance-relevant causality works
- prefer direct statements over vague descriptions

Quantitative Understanding:
- always include units when numerical values are stated
- include conditions or benchmark context when relevant
- if no trusted local estimate exists, say that explicitly

Local Findings:
- keep this focused on local run evidence, not generic background knowledge
- write in short narrative prose, not raw logs
- analyzer should update this field when new local evidence materially changes the section

Evidence:
- prefer local artifact references first
- external sources may appear, but should be clearly identifiable

Open Questions:
- questions should be researchable
- questions should be attached to a specific section, not left globally vague

Cross References:
- use them to connect causal or compositional dependencies
- this is the main graph-like structure in the canonical markdown representation

Status:
- `Known` means evidence-backed and locally trusted
- `Frontier` means partially understood or awaiting decisive local evidence
- `Unknown` means recognized but not yet characterized
- `Contradicted` means prior understanding has been challenged by evidence

Prerequisites:
- list the lower-level sections that should already be sufficiently known
- planner ranking should prefer frontier questions whose prerequisites are already satisfied

Frontier Criteria:
- state what evidence would justify moving the section from frontier toward known
- analyzer should use these criteria when deciding whether the frontier advanced
- matching evidence may update `Quantitative Understanding` and `Local Findings` without advancing `Status` if the criteria are only partially satisfied

## Frontier Summary

At the end of the KB, include a compact frontier section:

```md
## Frontier Summary

### Active Frontier Questions

1. Question text
Related Sections  
- [X.Y Related Section]
- [X.Y Related Section]
```

This summary is a navigation aid. It should not replace the section-local `Open Questions` lists.

## Intended Agent Use

Planner:
- reads section-local open questions and frontier summary
- selects the highest-value next question

Researcher:
- expands sections with literature, prior methods, and new cross-references
- sharpens or adds open questions

Analyzer:
- updates quantitative understanding, local findings, and status from execution evidence
- only moves a section from frontier toward known when its stated frontier criteria are met

Codegen and runner:
- exist to resolve KB questions through bounded experiments
