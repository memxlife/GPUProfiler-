# Autonomous GPU Research Agent — Design Philosophy

Version: 0.5  
Status: Foundational Manifesto  
Last Updated: 2026-03-28  

---

## 1. The Nature of the System

This system is not a tool.

It is not a pipeline.  
It is not a benchmark framework.  
It is not a collection of agents.

It is a **self-evolving research organism**.

Its purpose is not to execute predefined logic, but to:
- explore an unknown space (GPU architecture),
- construct knowledge,
- refine its own reasoning,
- and continuously improve how it learns.

This system must be understood as a **process that becomes correct over time**, not something that is correct at initialization.

---

## 2. Human as God: Observe, Do Not Intervene

The role of the human is deliberately minimal and deliberately powerful.

The human defines:
- the existence of the system,
- the boundary conditions of its environment,
- and the laws under which it operates.

But once the system begins operating, the human does not intervene in its cognition.

The correct mental model is:

> The human is a god who creates a universe,  
> then observes its evolution without interfering in every event.

If the human:
- designs experiments,
- fixes reasoning,
- manually edits knowledge,
- or overrides decisions,

then the system is no longer autonomous.

Instead, the system must:
- act,
- fail,
- reflect,
- and improve on its own.

---

## 3. The Role of the Large Language Model

The large language model is not a component.  
It is the **cognitive substrate of the system**.

It is responsible for:
- deciding what to study,
- designing experiments,
- constructing representations,
- interpreting results,
- building knowledge,
- and proposing how the system itself should evolve.

This includes:
- schema design,
- schema validation,
- knowledge representation,
- and interface generation.

All intelligence belongs to the model.

---

## 4. Learning Through Mistakes

The system must be allowed to make mistakes.

> Mistakes are not failures of the system.  
> They are the primary driver of its learning.

Mistakes must be:
- observable,
- traceable,
- and recoverable.

The system must never:
- lose its history,
- lose its evidence,
- or silently overwrite its past.

Because without memory, mistakes cannot become knowledge.

---

## 5. Dynamic by Default, Stability as Emergence

Everything in the system should be considered dynamic:

- research direction  
- experiment design  
- schema  
- knowledge representation  
- interface format  
- execution strategy  

There is no predefined ontology.  
There is no fixed abstraction.

> Stability is not imposed.  
> Stability emerges from repeated successful structure.

> Static is relative.  
> Dynamic is absolute.

---

## 6. Minimal Laws, Maximum Freedom

Only minimal invariants exist:

- a closed learning loop  
- persistent artifacts  
- explicit provenance  
- versioned history  
- reproducible execution  
- bounded safety constraints  

Everything else evolves.

---

## 7. The Closed Learning Loop

world model  
→ question  
→ experiment  
→ observation  
→ claim  
→ update  
→ next question  

This loop is the only stable structure.

---

## 8. Two Interfaces: The Human–System Bridge

Because human cognitive bandwidth is limited, the system must provide efficient observation.

### 8.1 Observer Interface (Process)

Answers:

> What is the system doing right now?

It shows:
- current question  
- recent experiment  
- key result  
- claim  
- confidence  
- next question  
- system health  

---

### 8.2 Knowledge Interface (Learning)

Answers:

> What has the system learned?

It shows:
- concepts  
- mechanisms  
- evidence  
- uncertainty  
- implications  

---

## 9. Interface Principle

The interface is:

- LLM-generated  
- LLM-evolving  
- human-guided  

> The interface is a cognitive compression engine.

It compresses:
- complex execution  
- evolving schemas  
- multiple artifacts  

into:
- fast human understanding  

---

## 10. Observability

Every claim must be traceable:

claim → observation → execution → code → logs  

The system must expose:
- loop history  
- schema evolution  
- world model changes  
- failure points  

---

## 11. Final Principle

This system is not a tool.

It is:

> A continuously learning, self-modifying research organism  
> that produces both knowledge and understanding.
