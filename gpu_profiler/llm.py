import json
import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class PlanDecision:
    reason: str
    knowledge_model: dict[str, Any]
    proposal: dict[str, Any]
    research_request: dict[str, Any] | None
    planner: str


@dataclass
class ImplementationDecision:
    reason: str
    benchmarks: list[dict[str, Any]]
    negotiation: dict[str, Any]
    contract_amendments: list[dict[str, Any]]
    planner: str


@dataclass
class AnalysisDecision:
    summary: str
    claims: list[dict[str, Any]]
    covered_dimensions: list[str]
    stop: bool
    reason: str
    veto_next_plan: bool
    veto_reason: str
    required_observability: list[str]
    contract_amendments: list[dict[str, Any]]
    planner: str


@dataclass
class ContractDecision:
    schema_contract: dict[str, Any]
    reason: str
    planner: str


@dataclass
class ResearchDecision:
    reason: str
    request_summary: str
    unanswered_questions: list[str]
    findings: list[dict[str, Any]]
    proposed_dimensions: list[str]
    planner: str


class LLMWorkflowBackend:
    def negotiate_schema(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
    ) -> ContractDecision:
        raise NotImplementedError

    def research_context(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        research_request: dict[str, Any] | None = None,
        max_sources: int = 8,
    ) -> ResearchDecision:
        raise NotImplementedError

    def propose_plan(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
        max_benchmarks: int,
    ) -> PlanDecision:
        raise NotImplementedError

    def generate_implementation(
        self,
        intent: str,
        kb: dict[str, Any],
        plan: dict[str, Any],
        iteration: int,
        max_benchmarks: int,
    ) -> ImplementationDecision:
        raise NotImplementedError

    def analyze_results(
        self,
        intent: str,
        kb: dict[str, Any],
        plan: dict[str, Any],
        execution_results: list[dict[str, Any]],
        iteration: int,
        max_iterations: int,
    ) -> AnalysisDecision:
        raise NotImplementedError


class HeuristicWorkflowBackend(LLMWorkflowBackend):
    """
    Generic fallback backend.
    It intentionally avoids domain-specific benchmark logic; it only keeps loop progress alive.
    """

    name = "heuristic-workflow-v2"

    def negotiate_schema(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
    ) -> ContractDecision:
        _ = (intent, kb, iteration, max_iterations)
        return ContractDecision(
            schema_contract={
                "version": "1.0",
                "negotiation_policy": {
                    "goals": {
                        "planner": "maximize_coverage",
                        "codegen": "maximize_implementability",
                        "analyzer": "maximize_observability",
                    },
                    "score_fields": [
                        "coverage_gain_score",
                        "implementability_score",
                        "observability_score",
                    ],
                    "thresholds": {
                        "coverage_gain_min": 0.45,
                        "implementability_min": 0.55,
                        "observability_min": 0.5,
                        "utility_min": 0.55,
                    },
                    "weights": {
                        "coverage_gain_score": 0.4,
                        "implementability_score": 0.3,
                        "observability_score": 0.3,
                    },
                    "max_amendment_rounds": 2,
                    "amendment_policy": "Reject below-threshold proposals, keep rationale, and request revised implementation.",
                },
                "planner_output": {
                    "required_keys": ["reason", "knowledge_model", "proposal"],
                },
                "implementation_output": {
                    "required_keys": ["reason", "benchmarks"],
                    "benchmark_required_keys": [
                        "id",
                        "command",
                        "hypothesis",
                        "dimensions",
                        "scores",
                    ],
                },
                "analysis_output": {
                    "required_keys": [
                        "summary",
                        "claims",
                        "covered_dimensions",
                        "stop",
                        "reason",
                        "veto_next_plan",
                        "veto_reason",
                    ],
                },
            },
            reason="Fallback schema contract with generic required keys.",
            planner=self.name,
        )

    def research_context(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        research_request: dict[str, Any] | None = None,
        max_sources: int = 8,
    ) -> ResearchDecision:
        _ = (intent, kb, iteration, max_sources)
        return ResearchDecision(
            reason="Heuristic backend does not perform online research.",
            request_summary=str((research_request or {}).get("request_summary", "")).strip(),
            unanswered_questions=[
                str(x).strip() for x in (research_request or {}).get("target_questions", []) if str(x).strip()
            ],
            findings=[],
            proposed_dimensions=[],
            planner=self.name,
        )

    def propose_plan(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
        max_benchmarks: int,
    ) -> PlanDecision:
        target = kb.get("target_dimensions", [])
        covered = set(kb.get("covered_dimensions", []))
        uncovered = [d for d in target if d not in covered]

        if iteration >= max_iterations:
            return PlanDecision(
                reason="Reached iteration limit.",
                knowledge_model=_default_knowledge_model(intent=intent, focus_dimensions=[]),
                proposal=_default_proposal(intent=intent, focus_dimensions=[], iteration=iteration),
                research_request=None,
                planner=self.name,
            )

        if not target:
            focus = [f"dimension_{iteration + 1}"]
        else:
            focus = uncovered[: max(1, max_benchmarks)] or target[:1]

        _ = intent
        knowledge_model = _default_knowledge_model(intent=intent, focus_dimensions=focus)
        proposal = _default_proposal(intent=intent, focus_dimensions=focus, iteration=iteration)
        research_request = _default_research_request(intent=intent, focus_dimensions=focus)
        return PlanDecision(
            reason="Fallback planner produced generic plan items.",
            knowledge_model=knowledge_model,
            proposal=proposal,
            research_request=research_request,
            planner=self.name,
        )

    def generate_implementation(
        self,
        intent: str,
        kb: dict[str, Any],
        plan: dict[str, Any],
        iteration: int,
        max_benchmarks: int,
    ) -> ImplementationDecision:
        _ = (intent, kb)
        focus = _proposal_focus_nodes(plan.get("proposal", {}))[: max(1, max_benchmarks)]
        benchmarks = []
        for idx, dim in enumerate(focus or [f"dimension_{iteration + 1}"]):
            benchmarks.append(
                {
                    "id": f"generic_benchmark_{iteration}_{idx}",
                    "command": "python -c \"print('SKIP: heuristic backend requires LLM-generated benchmark content')\"",
                    "hypothesis": f"Placeholder benchmark for {dim}.",
                    "dimensions": [dim],
                    "analysis_method": {
                        "summary": "Use workload return code and artifact inspection to determine if benchmark executed.",
                        "metrics": ["returncode", "elapsed_sec"],
                    },
                    "scores": {
                        "coverage_gain_score": 0.55,
                        "implementability_score": 0.95,
                        "observability_score": 0.65,
                        "rationale": "Fallback placeholder benchmark is executable with basic observability.",
                    },
                    "files": [
                        {
                            "path": f"generated/iter_{iteration:02d}/benchmark_{idx:02d}.cu",
                            "type": "cu",
                            "content": "// Placeholder .cu benchmark generated by heuristic fallback\nint main(){return 0;}\n",
                        },
                        {
                            "path": f"generated/iter_{iteration:02d}/analysis_method_{idx:02d}.md",
                            "type": "md",
                            "content": "# Analysis Method\n\nEvaluate run status and produced artifacts.\n",
                        },
                    ],
                    "provenance": {"planner": self.name},
                }
            )
        return ImplementationDecision(
            reason="Fallback code generator emitted generic placeholders.",
            benchmarks=benchmarks,
            negotiation={"accepted": [], "rejected": [], "policy": _default_negotiation_policy()},
            contract_amendments=[],
            planner=self.name,
        )

    def analyze_results(
        self,
        intent: str,
        kb: dict[str, Any],
        plan: dict[str, Any],
        execution_results: list[dict[str, Any]],
        iteration: int,
        max_iterations: int,
    ) -> AnalysisDecision:
        _ = (intent, plan)
        covered = set(kb.get("covered_dimensions", []))
        claims: list[dict[str, Any]] = []
        for item in execution_results:
            workload = item.get("workload", {})
            if workload.get("returncode") == 0 and not workload.get("skipped", False):
                dims = [str(d) for d in item.get("dimensions", []) if str(d)]
                covered.update(dims)
                claims.append(
                    {
                        "claim": f"Benchmark `{item.get('benchmark_id')}` executed successfully.",
                        "dimensions": dims,
                        "confidence": "low",
                        "evidence": {
                            "analysis_artifact": item.get("analysis_artifact"),
                            "workload_artifact": item.get("workload_artifact"),
                        },
                    }
                )

        target = kb.get("target_dimensions", [])
        coverage = 0.0
        if target:
            coverage = len(set(target).intersection(covered)) / len(target)

        stop = coverage >= float(kb.get("target_coverage", 0.9)) or iteration + 1 >= max_iterations
        failed_or_skipped = [
            item
            for item in execution_results
            if item.get("workload", {}).get("returncode") != 0 or item.get("workload", {}).get("skipped", False)
        ]
        veto_next_plan = len(failed_or_skipped) == len(execution_results) and len(execution_results) > 0
        return AnalysisDecision(
            summary="Fallback analysis updated claims from executable evidence.",
            claims=claims,
            covered_dimensions=sorted(covered),
            stop=stop,
            reason="Coverage or iteration limit reached." if stop else "Continue iterations.",
            veto_next_plan=veto_next_plan,
            veto_reason="All suite benchmarks failed or were skipped." if veto_next_plan else "",
            required_observability=[],
            contract_amendments=[],
            planner=self.name,
        )


class OpenAIWorkflowBackend(LLMWorkflowBackend):
    def __init__(self, model: str = "gpt-5.4") -> None:
        self.model = model
        self.name = f"openai:{self.model}"

    def negotiate_schema(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
    ) -> ContractDecision:
        payload = {
            "intent": intent,
            "knowledge_base": kb,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "task": (
                "Planner agent, code-generation agent, and analysis agent collaboratively define "
                "a shared JSON contract for this iteration."
            ),
            "output_schema": {
                "reason": "str",
                "schema_contract": {
                    "version": "str",
                    "negotiation_policy": {
                        "score_fields": ["str"],
                        "thresholds": {
                            "coverage_gain_min": "float(0..1)",
                            "implementability_min": "float(0..1)",
                            "observability_min": "float(0..1)",
                            "utility_min": "float(0..1)",
                        },
                        "weights": {
                            "coverage_gain_score": "float(0..1)",
                            "implementability_score": "float(0..1)",
                            "observability_score": "float(0..1)",
                        },
                        "max_amendment_rounds": "int>=0",
                        "amendment_policy": "str",
                    },
                    "planner_output": {"required_keys": ["str"]},
                    "implementation_output": {"required_keys": ["str"], "benchmark_required_keys": ["str"]},
                    "analysis_output": {"required_keys": ["str"]},
                },
            },
        }
        out = self._json_completion(
            system=(
                "You are three collaborating agents: planner, code-generator, analyzer. "
                "Negotiate and return one strict JSON contract all agents must follow. "
                "Return strict JSON only."
            ),
            user=payload,
        )
        contract = out.get("schema_contract", {})
        if not isinstance(contract, dict):
            contract = {}
        contract = _merge_schema_contract(contract)
        return ContractDecision(
            schema_contract=contract,
            reason=str(out.get("reason", "")),
            planner=self.name,
        )

    def research_context(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        research_request: dict[str, Any] | None = None,
        max_sources: int = 8,
    ) -> ResearchDecision:
        request = research_request or {}
        payload = {
            "intent": intent,
            "knowledge_base": kb,
            "iteration": iteration,
            "research_request": request,
            "max_sources": max_sources,
            "task": (
                "Search online only within the scope requested by the planner-authored research request. "
                "Prioritize vendor docs, profiler docs, papers, and reproducible benchmark methodologies."
            ),
            "output_schema": {
                "reason": "str",
                "request_summary": "str",
                "findings": [
                    {
                        "title": "str",
                        "summary": "str",
                        "relevance": "str",
                        "source_url": "str",
                    }
                ],
                "unanswered_questions": ["str"],
                "proposed_dimensions": ["str"],
            },
        }
        out = self._json_completion(
            system=(
                "You are a planner-directed search agent. Use web search to answer the planner's research request. "
                "Stay within scope unless minor search refinement is necessary. Return strict JSON only."
            ),
            user=payload,
            tools=[{"type": "web_search_preview"}],
        )
        findings = out.get("findings", [])
        if not isinstance(findings, list):
            findings = []
        clean_findings: list[dict[str, Any]] = []
        for item in findings[: max(1, max_sources)]:
            if not isinstance(item, dict):
                continue
            source_url = str(item.get("source_url", "")).strip()
            if not source_url.startswith("http"):
                continue
            clean_findings.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "summary": str(item.get("summary", "")).strip(),
                    "relevance": str(item.get("relevance", "")).strip(),
                    "source_url": source_url,
                }
            )
        dims = _sanitize_dimensions(out.get("proposed_dimensions", []), [], 24)
        return ResearchDecision(
            reason=str(out.get("reason", "")),
            request_summary=str(out.get("request_summary", request.get("request_summary", ""))).strip(),
            unanswered_questions=[str(x).strip() for x in out.get("unanswered_questions", []) if str(x).strip()],
            findings=clean_findings,
            proposed_dimensions=dims,
            planner=self.name,
        )

    def propose_plan(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
        max_benchmarks: int,
    ) -> PlanDecision:
        payload = {
            "intent": intent,
            "knowledge_base": kb,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "max_benchmarks": max_benchmarks,
            "output_schema": {
                "reason": "str",
                "knowledge_model": {
                    "intent": {"summary": "str"},
                    "domain_hierarchy": [
                        {
                            "id": "str",
                            "name": "str",
                            "description": "str",
                            "parent_id": "str|null",
                            "node_type": "domain|subdomain|feature|question",
                            "status": "unknown|hypothesized|partially_supported|well_supported|contradictory",
                            "rationale": "str",
                            "evidence_refs": ["str"],
                            "open_gaps": ["str"],
                        }
                    ],
                    "focus_nodes": ["str"],
                    "generated_at": "str",
                    "planner_notes": "str",
                },
                "proposal": {
                    "intent_summary": "str",
                    "proposal_summary": "str",
                    "target_nodes": ["str"],
                    "proposals": [
                        {
                            "id": "str",
                            "title": "str",
                            "objective": "str",
                            "target_node_ids": ["str"],
                            "priority": "high|medium|low",
                            "benchmark_role": "baseline|isolation|sweep|interaction|stress|refinement|validation",
                            "description": "str",
                            "hypothesis": "str",
                            "required_evidence": ["str"],
                            "rationale": "str",
                            "prerequisites": ["str"],
                            "next_if_success": ["str"],
                            "next_if_failure": ["str"],
                        }
                    ],
                    "planner_notes": "str",
                    "generated_at": "str",
                },
                "research_request": {
                    "intent_summary": "str",
                    "request_summary": "str",
                    "target_nodes": ["str"],
                    "target_questions": ["str"],
                    "search_topics": ["str"],
                    "source_preferences": ["str"],
                    "source_constraints": ["str"],
                    "expected_outputs": ["str"],
                    "notes": "str",
                },
            },
        }
        out = self._json_completion(
            system=(
                "You are a planner agent for iterative performance-modeling workflows. Return strict JSON only. "
                "Generate a domain-specific knowledge hierarchy and a non-executable research proposal from user intent. "
                "Do not emit executable code, commands, or profiler invocations."
            ),
            user=payload,
        )
        focus = _sanitize_dimensions(_proposal_focus_nodes(out.get("proposal", {})), [], max_benchmarks)
        if not focus:
            focus = [f"dimension_{iteration + 1}"]
        knowledge_model = _sanitize_knowledge_model(out.get("knowledge_model", {}), intent=intent, focus_nodes=focus)
        proposal = _sanitize_proposal(out.get("proposal", {}), intent=intent, focus_nodes=focus, iteration=iteration)
        research_request = _sanitize_research_request(
            out.get("research_request", {}),
            intent=intent,
            proposal=proposal,
        )
        return PlanDecision(
            reason=str(out.get("reason", "")),
            knowledge_model=knowledge_model,
            proposal=proposal,
            research_request=research_request,
            planner=self.name,
        )

    def generate_implementation(
        self,
        intent: str,
        kb: dict[str, Any],
        plan: dict[str, Any],
        iteration: int,
        max_benchmarks: int,
    ) -> ImplementationDecision:
        focus = _proposal_focus_nodes(plan.get("proposal", {}))
        if not focus:
            focus = [f"dimension_{iteration + 1}"]
        focus = focus[: max(1, max_benchmarks)]
        benchmarks_raw: list[dict[str, Any]] = []
        reasons: list[str] = []
        amendments: list[dict[str, Any]] = []
        for dim in focus:
            payload = {
                "intent": intent,
                "knowledge_base": kb,
                "plan": plan,
                "iteration": iteration,
                "dimension": dim,
                "output_schema": {
                    "reason": "str",
                    "benchmark": {
                        "id": "str",
                        "command": "str",
                        "hypothesis": "str",
                        "dimensions": ["str"],
                        "analysis_method": {
                            "summary": "str",
                            "metrics": ["str"],
                            "decision_logic": "str",
                        },
                        "scores": {
                            "coverage_gain_score": "float(0..1)",
                            "implementability_score": "float(0..1)",
                            "observability_score": "float(0..1)",
                            "rationale": "str",
                        },
                        "files": [
                            {"path": "str(.md|.json|.cu)", "type": "md|json|cu", "content": "str"}
                        ],
                    },
                    "contract_amendments": [
                        {"path": "str", "change": "str", "rationale": "str", "priority": "low|medium|high"}
                    ],
                },
                "constraints": {
                    "bounded_runtime": "Each command should complete in <= 45 seconds when possible.",
                    "safety": "No destructive commands, no system configuration mutation.",
                    "no_inventory_only": "Do not return inventory/topology-only probes as benchmarks.",
                },
            }
            out = self._json_completion(
                system=(
                    "You are a code-generation agent. Return strict JSON only. "
                    "Generate exactly one concrete benchmark for the requested dimension. "
                    "The benchmark implementation must be CUDA C++ source (.cu), and the command must compile and/or run it "
                    "(for example with nvcc and executable invocation). Do not use Python benchmark scripts."
                ),
                user=payload,
            )
            reasons.append(str(out.get("reason", "")))
            bench = out.get("benchmark", {})
            if isinstance(bench, dict):
                benchmarks_raw.append(bench)
            amendments.extend(_sanitize_amendments(out.get("contract_amendments", [])))
        benches = _sanitize_benchmarks(
            benchmarks_raw,
            target_dimensions=kb.get("target_dimensions", []),
            max_benchmarks=max_benchmarks,
            planner_name=self.name,
            focus_dimensions=_proposal_focus_nodes(plan.get("proposal", {})),
            contract=kb.get("schema_contract", {}),
        )
        return ImplementationDecision(
            reason="; ".join([r for r in reasons if r]).strip(),
            benchmarks=benches,
            negotiation={"accepted": [], "rejected": [], "policy": _policy_from_contract(kb.get("schema_contract", {}))},
            contract_amendments=amendments,
            planner=self.name,
        )

    def analyze_results(
        self,
        intent: str,
        kb: dict[str, Any],
        plan: dict[str, Any],
        execution_results: list[dict[str, Any]],
        iteration: int,
        max_iterations: int,
    ) -> AnalysisDecision:
        payload = {
            "intent": intent,
            "knowledge_base": kb,
            "plan": plan,
            "execution_results": execution_results,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "output_schema": {
                "summary": "str",
                "claims": [
                    {
                        "claim": "str",
                        "dimensions": ["str"],
                        "confidence": "low|medium|high",
                        "evidence": {"analysis_artifact": "str", "workload_artifact": "str"},
                    }
                ],
                "covered_dimensions": ["str"],
                "stop": "bool",
                "reason": "str",
                "veto_next_plan": "bool",
                "veto_reason": "str",
                "required_observability": ["str"],
                "contract_amendments": [
                    {"path": "str", "change": "str", "rationale": "str", "priority": "low|medium|high"}
                ],
            },
        }
        out = self._json_completion(
            system=(
                "You are an analysis agent that updates a knowledge base from benchmark evidence. "
                "Return strict JSON only. Only claim coverage for successful non-skipped runs."
            ),
            user=payload,
        )
        claims = out.get("claims", [])
        if not isinstance(claims, list):
            claims = []
        covered = _sanitize_dimensions(out.get("covered_dimensions", []), kb.get("target_dimensions", []), 999)
        if not covered:
            covered = _dims_from_claims(claims)
        return AnalysisDecision(
            summary=str(out.get("summary", "")),
            claims=claims,
            covered_dimensions=covered,
            stop=bool(out.get("stop", False)) or iteration + 1 >= max_iterations,
            reason=str(out.get("reason", "")),
            veto_next_plan=bool(out.get("veto_next_plan", False)),
            veto_reason=str(out.get("veto_reason", "")),
            required_observability=[str(x) for x in out.get("required_observability", []) if str(x).strip()],
            contract_amendments=_sanitize_amendments(out.get("contract_amendments", [])),
            planner=self.name,
        )

    def _json_completion(self, system: str, user: dict[str, Any], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("OpenAI workflow backend requires `openai` package.") from exc

        api_key = _normalized_api_key()
        client = OpenAI(api_key=api_key or None, timeout=45.0)
        request: dict[str, Any] = {
            "model": self.model,
            "max_output_tokens": 2600,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
            ],
        }
        if tools:
            request["tools"] = tools

        parse_error: Exception | None = None
        raw_text = ""
        for _ in range(2):
            resp = client.responses.create(**request)
            raw_text = (resp.output_text or "").strip()
            try:
                return _parse_json_object(raw_text)
            except Exception as exc:  # noqa: BLE001
                parse_error = exc
                raw_text = self._repair_json(client, raw_text, user)
                try:
                    return _parse_json_object(raw_text)
                except Exception as exc2:  # noqa: BLE001
                    parse_error = exc2
        raise ValueError(f"Failed to parse JSON response: {parse_error}; text_prefix={raw_text[:180]}")

    def _repair_json(self, client: Any, malformed: str, schema_hint: dict[str, Any]) -> str:
        repair = client.responses.create(
            model=self.model,
            max_output_tokens=1400,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Repair malformed JSON. Return only a valid JSON object. "
                        "No markdown, no explanation, no extra text."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "malformed_output": malformed,
                            "required_shape_hint": schema_hint.get("output_schema", {}),
                        }
                    ),
                },
            ],
        )
        return (repair.output_text or "").strip()


class ResilientWorkflowBackend(LLMWorkflowBackend):
    def __init__(self, primary: LLMWorkflowBackend, fallback: LLMWorkflowBackend) -> None:
        self.primary = primary
        self.fallback = fallback

    def negotiate_schema(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
    ) -> ContractDecision:
        try:
            return self.primary.negotiate_schema(intent, kb, iteration, max_iterations)
        except Exception as exc:  # noqa: BLE001
            alt = self.fallback.negotiate_schema(intent, kb, iteration, max_iterations)
            alt.reason = f"Primary schema negotiation failed ({exc}); fallback used. {alt.reason}"
            alt.planner = f"{getattr(self.primary, 'name', 'primary')}->fallback:{alt.planner}"
            return alt

    def propose_plan(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
        max_benchmarks: int,
    ) -> PlanDecision:
        try:
            return self.primary.propose_plan(intent, kb, iteration, max_iterations, max_benchmarks)
        except Exception as exc:  # noqa: BLE001
            alt = self.fallback.propose_plan(intent, kb, iteration, max_iterations, max_benchmarks)
            alt.reason = f"Primary planning failed ({exc}); fallback used. {alt.reason}"
            alt.planner = f"{getattr(self.primary, 'name', 'primary')}->fallback:{alt.planner}"
            return alt

    def research_context(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        research_request: dict[str, Any] | None = None,
        max_sources: int = 8,
    ) -> ResearchDecision:
        try:
            return self.primary.research_context(intent, kb, iteration, research_request, max_sources)
        except Exception as exc:  # noqa: BLE001
            alt = self.fallback.research_context(intent, kb, iteration, research_request, max_sources)
            alt.reason = f"Primary research failed ({exc}); fallback used. {alt.reason}"
            alt.planner = f"{getattr(self.primary, 'name', 'primary')}->fallback:{alt.planner}"
            return alt

    def generate_implementation(
        self,
        intent: str,
        kb: dict[str, Any],
        plan: dict[str, Any],
        iteration: int,
        max_benchmarks: int,
    ) -> ImplementationDecision:
        try:
            return self.primary.generate_implementation(intent, kb, plan, iteration, max_benchmarks)
        except Exception as exc:  # noqa: BLE001
            alt = self.fallback.generate_implementation(intent, kb, plan, iteration, max_benchmarks)
            alt.reason = f"Primary implementation generation failed ({exc}); fallback used. {alt.reason}"
            alt.planner = f"{getattr(self.primary, 'name', 'primary')}->fallback:{alt.planner}"
            return alt

    def analyze_results(
        self,
        intent: str,
        kb: dict[str, Any],
        plan: dict[str, Any],
        execution_results: list[dict[str, Any]],
        iteration: int,
        max_iterations: int,
    ) -> AnalysisDecision:
        try:
            return self.primary.analyze_results(intent, kb, plan, execution_results, iteration, max_iterations)
        except Exception as exc:  # noqa: BLE001
            alt = self.fallback.analyze_results(intent, kb, plan, execution_results, iteration, max_iterations)
            alt.reason = f"Primary analysis failed ({exc}); fallback used. {alt.reason}"
            alt.planner = f"{getattr(self.primary, 'name', 'primary')}->fallback:{alt.planner}"
            return alt


def _normalized_api_key() -> str:
    return "".join(os.getenv("OPENAI_API_KEY", "").split())


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("LLM output did not contain JSON object")
    return json.loads(match.group(0))


def _sanitize_dimensions(raw: list[Any], allowed: list[str], max_items: int) -> list[str]:
    out: list[str] = []
    allow = set(allowed or [])
    for item in raw:
        dim = str(item).strip()
        if not dim:
            continue
        if allow and dim not in allow:
            continue
        if dim not in out:
            out.append(dim)
        if len(out) >= max_items:
            break
    return out


def _default_knowledge_model(intent: str, focus_dimensions: list[str]) -> dict[str, Any]:
    hierarchy: list[dict[str, Any]] = []
    for idx, dim in enumerate(focus_dimensions):
        hierarchy.append(
            {
                "id": f"feature_{idx}",
                "name": dim,
                "description": f"Domain feature derived from planning intent: {dim}",
                "parent_id": None,
                "node_type": "feature",
                "status": "unknown",
                "rationale": "No verified local evidence yet.",
                "evidence_refs": [],
                "open_gaps": [f"Need a baseline benchmark proposal for {dim}."],
            }
        )
    return {
        "intent": {"summary": intent},
        "domain_hierarchy": hierarchy,
        "focus_nodes": [item["id"] for item in hierarchy],
        "generated_at": "",
        "planner_notes": "Fallback knowledge model.",
    }


def _default_proposal(intent: str, focus_dimensions: list[str], iteration: int) -> dict[str, Any]:
    proposals: list[dict[str, Any]] = []
    target_nodes: list[str] = []
    for idx, dim in enumerate(focus_dimensions):
        node_id = f"feature_{idx}"
        target_nodes.append(node_id)
        proposals.append(
            {
                "id": f"proposal_{iteration}_{idx}",
                "title": f"Baseline benchmark for {dim}",
                "objective": f"Establish the first measurable characterization for {dim}.",
                "target_node_ids": [node_id],
                "priority": "high",
                "benchmark_role": "baseline",
                "description": f"Start with a simple benchmark that isolates {dim}.",
                "hypothesis": f"A minimal benchmark can produce first evidence for {dim}.",
                "required_evidence": ["Successful execution", "Auditable measurement artifacts"],
                "rationale": "Curriculum-first proposal from fallback planner.",
                "prerequisites": [],
                "next_if_success": [f"Expand {dim} into a parameter sweep."],
                "next_if_failure": [f"Simplify or repair the benchmark design for {dim}."],
            }
        )
    return {
        "intent_summary": intent,
        "proposal_summary": "Fallback non-executable proposal.",
        "target_nodes": target_nodes,
        "proposals": proposals,
        "planner_notes": "Fallback proposal generated without domain-specific executable content.",
        "generated_at": "",
    }


def _default_research_request(intent: str, focus_dimensions: list[str]) -> dict[str, Any]:
    if not focus_dimensions:
        return {
            "intent_summary": intent,
            "request_summary": "",
            "target_nodes": [],
            "target_questions": [],
            "search_topics": [],
            "source_preferences": [],
            "source_constraints": [],
            "expected_outputs": [],
            "notes": "",
        }
    return {
        "intent_summary": intent,
        "request_summary": "Gather external background knowledge for the current planner focus.",
        "target_nodes": focus_dimensions[:3],
        "target_questions": [f"What established methods and prior knowledge exist for {item}?" for item in focus_dimensions[:3]],
        "search_topics": focus_dimensions[:5],
        "source_preferences": ["vendor_doc", "official_tool_doc", "paper", "article"],
        "source_constraints": [],
        "expected_outputs": ["benchmark methodology", "architecture background", "observability guidance"],
        "notes": "Fallback planner-generated research request.",
    }


def _sanitize_knowledge_model(raw: dict[str, Any], intent: str, focus_nodes: list[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    hierarchy = raw.get("domain_hierarchy", [])
    clean_hierarchy: list[dict[str, Any]] = []
    if isinstance(hierarchy, list):
        for item in hierarchy[:200]:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("id", "")).strip()
            name = str(item.get("name", "")).strip()
            if not node_id or not name:
                continue
            clean_hierarchy.append(
                {
                    "id": node_id,
                    "name": name,
                    "description": str(item.get("description", "")).strip(),
                    "parent_id": item.get("parent_id"),
                    "node_type": str(item.get("node_type", "feature")).strip() or "feature",
                    "status": str(item.get("status", "unknown")).strip() or "unknown",
                    "rationale": str(item.get("rationale", "")).strip(),
                    "evidence_refs": [str(x).strip() for x in item.get("evidence_refs", []) if str(x).strip()],
                    "open_gaps": [str(x).strip() for x in item.get("open_gaps", []) if str(x).strip()],
                }
            )
    if not clean_hierarchy:
        return _default_knowledge_model(intent=intent, focus_dimensions=focus_nodes)
    valid_node_ids = {item["id"] for item in clean_hierarchy}
    sanitized_focus = [node for node in focus_nodes if node in valid_node_ids]
    if not sanitized_focus:
        sanitized_focus = [item["id"] for item in clean_hierarchy[: min(3, len(clean_hierarchy))]]
    return {
        "intent": {"summary": str(raw.get("intent", {}).get("summary", intent)).strip() or intent},
        "domain_hierarchy": clean_hierarchy,
        "focus_nodes": sanitized_focus,
        "generated_at": str(raw.get("generated_at", "")).strip(),
        "planner_notes": str(raw.get("planner_notes", "")).strip(),
    }


def _sanitize_proposal(raw: dict[str, Any], intent: str, focus_nodes: list[str], iteration: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    proposals_raw = raw.get("proposals", [])
    clean_proposals: list[dict[str, Any]] = []
    if isinstance(proposals_raw, list):
        for idx, item in enumerate(proposals_raw[:50]):
            if not isinstance(item, dict):
                continue
            proposal_id = str(item.get("id", "")).strip() or f"proposal_{iteration}_{idx}"
            title = str(item.get("title", "")).strip()
            objective = str(item.get("objective", "")).strip()
            if not title or not objective:
                continue
            clean_proposals.append(
                {
                    "id": proposal_id,
                    "title": title,
                    "objective": objective,
                    "target_node_ids": [str(x).strip() for x in item.get("target_node_ids", []) if str(x).strip()],
                    "priority": _sanitize_priority(item.get("priority", "medium")),
                    "benchmark_role": _sanitize_benchmark_role(item.get("benchmark_role", "baseline")),
                    "description": str(item.get("description", "")).strip(),
                    "hypothesis": str(item.get("hypothesis", "")).strip(),
                    "required_evidence": [str(x).strip() for x in item.get("required_evidence", []) if str(x).strip()],
                    "rationale": str(item.get("rationale", "")).strip(),
                    "prerequisites": [str(x).strip() for x in item.get("prerequisites", []) if str(x).strip()],
                    "next_if_success": [str(x).strip() for x in item.get("next_if_success", []) if str(x).strip()],
                    "next_if_failure": [str(x).strip() for x in item.get("next_if_failure", []) if str(x).strip()],
                }
            )
    if not clean_proposals:
        return _default_proposal(intent=intent, focus_dimensions=focus_nodes, iteration=iteration)
    target_nodes = [str(x).strip() for x in raw.get("target_nodes", []) if str(x).strip()]
    if not target_nodes:
        target_nodes = _proposal_focus_nodes({"proposals": clean_proposals})
    return {
        "intent_summary": str(raw.get("intent_summary", intent)).strip() or intent,
        "proposal_summary": str(raw.get("proposal_summary", "")).strip(),
        "target_nodes": target_nodes,
        "proposals": clean_proposals,
        "planner_notes": str(raw.get("planner_notes", "")).strip(),
        "generated_at": str(raw.get("generated_at", "")).strip(),
    }


def _sanitize_research_request(raw: dict[str, Any], intent: str, proposal: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        raw = {}
    target_nodes = [str(x).strip() for x in raw.get("target_nodes", []) if str(x).strip()]
    target_questions = [str(x).strip() for x in raw.get("target_questions", []) if str(x).strip()]
    search_topics = [str(x).strip() for x in raw.get("search_topics", []) if str(x).strip()]
    if not any([target_nodes, target_questions, search_topics]):
        fallback_focus = [str(x).strip() for x in proposal.get("target_nodes", []) if str(x).strip()]
        if not fallback_focus:
            return None
        return _default_research_request(intent=intent, focus_dimensions=fallback_focus)
    return {
        "intent_summary": str(raw.get("intent_summary", intent)).strip() or intent,
        "request_summary": str(raw.get("request_summary", "")).strip(),
        "target_nodes": target_nodes,
        "target_questions": target_questions,
        "search_topics": search_topics,
        "source_preferences": [str(x).strip() for x in raw.get("source_preferences", []) if str(x).strip()],
        "source_constraints": [str(x).strip() for x in raw.get("source_constraints", []) if str(x).strip()],
        "expected_outputs": [str(x).strip() for x in raw.get("expected_outputs", []) if str(x).strip()],
        "notes": str(raw.get("notes", "")).strip(),
    }


def _proposal_focus_nodes(proposal: dict[str, Any]) -> list[str]:
    if not isinstance(proposal, dict):
        return []
    out: list[str] = []
    for item in proposal.get("target_nodes", []):
        node = str(item).strip()
        if node and node not in out:
            out.append(node)
    for item in proposal.get("proposals", []):
        if not isinstance(item, dict):
            continue
        for node_item in item.get("target_node_ids", []):
            node = str(node_item).strip()
            if node and node not in out:
                out.append(node)
    return out


def _sanitize_priority(value: Any) -> str:
    priority = str(value).strip().lower()
    return priority if priority in {"high", "medium", "low"} else "medium"


def _sanitize_benchmark_role(value: Any) -> str:
    role = str(value).strip().lower()
    allowed = {"baseline", "isolation", "sweep", "interaction", "stress", "refinement", "validation"}
    return role if role in allowed else "baseline"


def _dims_from_claims(claims: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for claim in claims:
        for d in claim.get("dimensions", []):
            dim = str(d).strip()
            if dim and dim not in out:
                out.append(dim)
    return out


def _sanitize_benchmarks(
    raw_benchmarks: list[dict[str, Any]],
    target_dimensions: list[str],
    max_benchmarks: int,
    planner_name: str,
    focus_dimensions: list[str] | None = None,
    contract: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    focus_dimensions = focus_dimensions or []
    policy = _policy_from_contract(contract or {})
    clean: list[dict[str, Any]] = []
    for i, item in enumerate(raw_benchmarks[: max(0, max_benchmarks)]):
        if not isinstance(item, dict):
            continue
        command = str(item.get("command", "")).strip()
        if not command or _is_inventory_only_command(command):
            continue

        dims = _sanitize_dimensions(item.get("dimensions", []), target_dimensions, 2)
        if not dims:
            dims = _sanitize_dimensions(item.get("dimensions", []), [], 2)
        if not dims:
            dims = _sanitize_dimensions(focus_dimensions, target_dimensions, 1)
        if not dims:
            dims = _sanitize_dimensions(focus_dimensions, [], 1)
        if not dims:
            dims = [f"dimension_{i+1}"]

        files = _sanitize_files(item.get("files", []))
        if not _has_cuda_source(files):
            continue
        if not _is_cuda_build_or_run_command(command):
            continue
        analysis_method = item.get("analysis_method", {}) if isinstance(item.get("analysis_method", {}), dict) else {}
        scores = _sanitize_scores(item.get("scores", {}))
        utility = _compute_utility(scores, policy.get("weights", {}))

        clean.append(
            {
                "id": str(item.get("id", f"benchmark_{i}")) or f"benchmark_{i}",
                "command": command,
                "hypothesis": str(item.get("hypothesis", "LLM-proposed benchmark")),
                "dimensions": dims,
                "analysis_method": analysis_method,
                "scores": scores,
                "utility_score": utility,
                "files": files,
                "provenance": {"planner": planner_name},
            }
        )
    return clean


def _sanitize_files(raw_files: list[dict[str, Any]]) -> list[dict[str, str]]:
    allowed_suffix = (".cu", ".md", ".json")
    out: list[dict[str, str]] = []
    if not isinstance(raw_files, list):
        return out
    for entry in raw_files:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path", "")).strip()
        content = str(entry.get("content", ""))
        file_type = str(entry.get("type", "")).strip().lower()
        if not path or not content:
            continue
        if ".." in path or path.startswith("/"):
            continue
        if not path.endswith(allowed_suffix):
            continue
        if file_type and file_type not in {"cu", "md", "json"}:
            continue
        out.append({"path": path, "type": file_type or path.rsplit(".", 1)[-1], "content": content})
    return out


def _has_cuda_source(files: list[dict[str, Any]]) -> bool:
    for entry in files:
        path = str(entry.get("path", "")).strip().lower()
        if path.endswith(".cu"):
            return True
    return False


def _is_cuda_build_or_run_command(command: str) -> bool:
    normalized = command.lower()
    cuda_signals = ["nvcc", ".cu", "./", "ncu ", "nsys "]
    return any(sig in normalized for sig in cuda_signals)


def _coerce_score(value: Any, default: float) -> float:
    try:
        num = float(value)
    except Exception:
        num = default
    return max(0.0, min(1.0, num))


def _sanitize_scores(raw: dict[str, Any]) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "coverage_gain_score": _coerce_score(raw.get("coverage_gain_score"), 0.5),
        "implementability_score": _coerce_score(raw.get("implementability_score"), 0.5),
        "observability_score": _coerce_score(raw.get("observability_score"), 0.5),
        "rationale": str(raw.get("rationale", "")).strip(),
    }


def _sanitize_amendments(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        change = str(item.get("change", "")).strip()
        rationale = str(item.get("rationale", "")).strip()
        priority = str(item.get("priority", "medium")).strip().lower()
        if not path or not change:
            continue
        if priority not in {"low", "medium", "high"}:
            priority = "medium"
        out.append({"path": path, "change": change, "rationale": rationale, "priority": priority})
    return out[:20]


def _compute_utility(scores: dict[str, Any], weights: dict[str, Any]) -> float:
    cov = _coerce_score(scores.get("coverage_gain_score"), 0.0)
    imp = _coerce_score(scores.get("implementability_score"), 0.0)
    obs = _coerce_score(scores.get("observability_score"), 0.0)
    w_cov = _coerce_score(weights.get("coverage_gain_score"), 0.4)
    w_imp = _coerce_score(weights.get("implementability_score"), 0.3)
    w_obs = _coerce_score(weights.get("observability_score"), 0.3)
    total = w_cov + w_imp + w_obs
    if total <= 0:
        return round((cov + imp + obs) / 3.0, 4)
    return round((cov * w_cov + imp * w_imp + obs * w_obs) / total, 4)


def _default_negotiation_policy() -> dict[str, Any]:
    return {
        "goals": {
            "planner": "maximize_coverage",
            "codegen": "maximize_implementability",
            "analyzer": "maximize_observability",
        },
        "score_fields": ["coverage_gain_score", "implementability_score", "observability_score"],
        "thresholds": {
            "coverage_gain_min": 0.45,
            "implementability_min": 0.55,
            "observability_min": 0.5,
            "utility_min": 0.55,
        },
        "weights": {
            "coverage_gain_score": 0.4,
            "implementability_score": 0.3,
            "observability_score": 0.3,
        },
        "max_amendment_rounds": 2,
        "amendment_policy": "Reject below-threshold proposals and request a revised implementation with explicit rationale.",
    }


def _policy_from_contract(contract: dict[str, Any]) -> dict[str, Any]:
    base = _default_negotiation_policy()
    policy = contract.get("negotiation_policy", {}) if isinstance(contract, dict) else {}
    if not isinstance(policy, dict):
        return base
    thresholds = policy.get("thresholds", {})
    weights = policy.get("weights", {})
    merged = {
        **base,
        **{k: v for k, v in policy.items() if k not in {"thresholds", "weights"}},
        "thresholds": {**base["thresholds"], **(thresholds if isinstance(thresholds, dict) else {})},
        "weights": {**base["weights"], **(weights if isinstance(weights, dict) else {})},
    }
    merged["thresholds"] = {
        k: _coerce_score(v, base["thresholds"].get(k, 0.5)) for k, v in merged["thresholds"].items()
    }
    merged["weights"] = {k: _coerce_score(v, base["weights"].get(k, 0.0)) for k, v in merged["weights"].items()}
    try:
        merged["max_amendment_rounds"] = max(0, int(merged.get("max_amendment_rounds", 2)))
    except Exception:
        merged["max_amendment_rounds"] = 2
    return merged


def _merge_schema_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, dict):
        contract = {}
    merged = {
        "version": str(contract.get("version", "1.0")),
        "negotiation_policy": _policy_from_contract(contract),
        "planner_output": contract.get("planner_output", {}),
        "implementation_output": contract.get("implementation_output", {}),
        "analysis_output": contract.get("analysis_output", {}),
    }
    for key in ["planner_output", "implementation_output", "analysis_output"]:
        if not isinstance(merged[key], dict):
            merged[key] = {}
    return merged


def _is_inventory_only_command(command: str) -> bool:
    normalized = command.lower()
    inventory_signals = ["nvidia-smi -l", "nvidia-smi --query-gpu", "nvidia-smi topo"]
    workload_signals = ["python -c", "ncu ", "nsys ", "nvprof", "cuda", "torch", "./", "make "]
    if any(sig in normalized for sig in workload_signals):
        return False
    return "nvidia-smi" in normalized and any(sig in normalized for sig in inventory_signals)
