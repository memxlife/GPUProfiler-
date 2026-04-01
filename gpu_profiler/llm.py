import json
import multiprocessing
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

PLANNER_INPUT_TARGET_CHARS = 6000
PLANNER_INPUT_HARD_CAP_CHARS = 10000
PLANNER_RESEARCH_INPUT_TARGET_CHARS = 4000
PLANNER_RESEARCH_INPUT_HARD_CAP_CHARS = 6000
PLANNER_REQUEST_TIMEOUT_SEC = 15.0
CODEGEN_INPUT_TARGET_CHARS = 8000
CODEGEN_INPUT_HARD_CAP_CHARS = 12000
CODEGEN_TIMEOUT_SEC = 35.0
CODEGEN_REPAIR_ATTEMPTS = 1
RESEARCH_TIMEOUT_SEC = 35.0
ANALYSIS_TIMEOUT_SEC = 20.0
OPENAI_RESPONSE_ROBUST_THRESHOLD_MULTIPLIER = 2.0
OPENAI_RESPONSE_DEFAULT_THRESHOLD_SEC = 45.0
OPENAI_RESPONSE_RESEARCH_THRESHOLD_SEC = 90.0
OPENAI_RESPONSE_ANALYSIS_THRESHOLD_SEC = 75.0
OPENAI_RESPONSE_CODEGEN_THRESHOLD_SEC = 120.0
CODEGEN_SYSTEM_PROMPT = (
    "You are a CUDA benchmark code-generation agent. Answer briefly and directly in plain Markdown. "
    "Do not output JSON. Produce exactly one concrete benchmark implementation memo. "
    "The implementation must use CUDA C++ source (.cu) and a compile/run shell command. "
    "Do not use Python benchmark scripts."
)


@dataclass
class ResearchRequestPlanDecision:
    reason: str
    research_request: dict[str, Any] | None
    current_question: str
    planner: str
    raw_response: str = ""


@dataclass
class ProposalPlanDecision:
    reason: str
    proposal: dict[str, Any]
    planner: str
    current_question: str = ""
    raw_response: str = ""


@dataclass
class PlanDecision:
    reason: str
    knowledge_model: dict[str, Any]
    proposal: dict[str, Any]
    research_request: dict[str, Any] | None
    current_question: str
    planner: str


@dataclass
class ImplementationDecision:
    reason: str
    benchmarks: list[dict[str, Any]]
    negotiation: dict[str, Any]
    contract_amendments: list[dict[str, Any]]
    planner: str
    raw_response: str = ""


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
    raw_response: str = ""


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
        research_request_memo: str = "",
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

    def plan_research_request(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
        max_benchmarks: int,
    ) -> ResearchRequestPlanDecision:
        raise NotImplementedError

    def plan_proposal(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
        max_benchmarks: int,
        question_memo: str = "",
        research_memo: str = "",
    ) -> ProposalPlanDecision:
        raise NotImplementedError

    def generate_implementation(
        self,
        intent: str,
        kb: dict[str, Any],
        plan: dict[str, Any],
        iteration: int,
        max_benchmarks: int,
        proposal_memo: str = "",
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
                "research_request_output": {
                    "required_keys": ["reason", "research_request"],
                },
                "proposal_output": {
                    "required_keys": ["reason", "proposal"],
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
        research_request_memo: str = "",
        max_sources: int = 8,
    ) -> ResearchDecision:
        _ = (intent, kb, iteration, max_sources, research_request_memo)
        return ResearchDecision(
            reason="Heuristic backend does not perform online research.",
            request_summary=str((research_request or {}).get("request_summary", "")).strip(),
            unanswered_questions=[
                str(x).strip() for x in (research_request or {}).get("target_questions", []) if str(x).strip()
            ],
            findings=[],
            proposed_dimensions=[],
            planner=self.name,
            raw_response="",
        )

    def propose_plan(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
        max_benchmarks: int,
    ) -> PlanDecision:
        research = self.plan_research_request(intent, kb, iteration, max_iterations, max_benchmarks)
        proposal = self.plan_proposal(intent, kb, iteration, max_iterations, max_benchmarks)
        return PlanDecision(
            reason=proposal.reason,
            knowledge_model=_current_or_default_knowledge_model(kb=kb, intent=intent),
            proposal=proposal.proposal,
            research_request=research.research_request,
            current_question=research.current_question,
            planner=self.name,
        )

    def plan_research_request(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
        max_benchmarks: int,
    ) -> ResearchRequestPlanDecision:
        _ = max_iterations
        focus = _planner_focus_dimensions_from_kb(kb=kb, iteration=iteration, max_benchmarks=max_benchmarks)
        if not focus:
            return ResearchRequestPlanDecision(
                reason="No external research needed for the current planner state.",
                research_request=None,
                current_question=_next_frontier_question(kb=kb, focus_dimensions=focus or ["gpu_performance"]),
                planner=self.name,
            )
        return ResearchRequestPlanDecision(
            reason="Fallback planner produced a generic research request.",
            research_request=_default_research_request(intent=intent, focus_dimensions=focus),
            current_question=_next_frontier_question(kb=kb, focus_dimensions=focus),
            planner=self.name,
            raw_response="",
        )

    def plan_proposal(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
        max_benchmarks: int,
        question_memo: str = "",
        research_memo: str = "",
    ) -> ProposalPlanDecision:
        _ = (max_iterations, research_memo)
        focus = _planner_focus_dimensions_from_kb(kb=kb, iteration=iteration, max_benchmarks=max_benchmarks)
        return ProposalPlanDecision(
            reason="Fallback planner produced generic plan items.",
            proposal=_default_proposal(intent=intent, focus_dimensions=focus, iteration=iteration),
            current_question=_question_text_from_memo(question_memo) or _next_frontier_question(kb=kb, focus_dimensions=focus),
            planner=self.name,
            raw_response="",
        )

    def generate_implementation(
        self,
        intent: str,
        kb: dict[str, Any],
        plan: dict[str, Any],
        iteration: int,
        max_benchmarks: int,
        proposal_memo: str = "",
    ) -> ImplementationDecision:
        _ = (intent, kb, proposal_memo)
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
            raw_response="",
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
            veto_reason="All implementation runs failed or were skipped." if veto_next_plan else "",
            required_observability=[],
            contract_amendments=[],
            planner=self.name,
        )


class OpenAIWorkflowBackend(LLMWorkflowBackend):
    def __init__(self, model: str = "gpt-5.4", request_timeout_sec: float = PLANNER_REQUEST_TIMEOUT_SEC) -> None:
        self.model = model
        self.request_timeout_sec = max(1.0, float(request_timeout_sec))
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
                    "research_request_output": {"required_keys": ["str"]},
                    "proposal_output": {"required_keys": ["str"]},
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
            context="schema-contract",
            timeout_sec=PLANNER_REQUEST_TIMEOUT_SEC,
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
        research_request_memo: str = "",
        max_sources: int = 8,
    ) -> ResearchDecision:
        request = research_request or {}
        compact_kb = _compact_search_kb(kb)
        payload = {
            "intent": intent,
            "knowledge_base": compact_kb,
            "iteration": iteration,
            "research_request": _compact_research_request(request),
            "research_request_memo": _trim_text(research_request_memo, 3000),
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
            context="research-context",
            timeout_sec=RESEARCH_TIMEOUT_SEC,
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
            raw_response=json.dumps(out, indent=2),
        )

    def propose_plan(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
        max_benchmarks: int,
    ) -> PlanDecision:
        research = self.plan_research_request(intent, kb, iteration, max_iterations, max_benchmarks)
        proposal = self.plan_proposal(intent, kb, iteration, max_iterations, max_benchmarks)
        return PlanDecision(
            reason=proposal.reason,
            knowledge_model=_current_or_default_knowledge_model(kb=kb, intent=intent),
            proposal=proposal.proposal,
            research_request=research.research_request,
            current_question=research.current_question,
            planner=self.name,
        )

    def plan_research_request(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
        max_benchmarks: int,
    ) -> ResearchRequestPlanDecision:
        compact_kb = _compact_planner_kb(
            kb=kb,
            max_nodes=max(4, max_benchmarks * 3),
            max_history_items=1,
            max_research_items=2,
        )
        payload = {
            "intent": intent,
            "knowledge_base": compact_kb,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "max_benchmarks": max_benchmarks,
            "output_schema": {
                "reason": "str",
                "current_question": "str",
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
        payload = _enforce_payload_budget(
            payload,
            target_chars=PLANNER_RESEARCH_INPUT_TARGET_CHARS,
            hard_cap_chars=PLANNER_RESEARCH_INPUT_HARD_CAP_CHARS,
            trimmers=[_trim_planner_payload],
        )
        out = self._json_completion(
            system=(
                "You are a planner agent deciding only whether external research is needed next. "
                "Return strict JSON only. Do not generate proposals, executable code, commands, or profiler invocations."
            ),
            user=payload,
            context="plan-research-request",
            timeout_sec=PLANNER_REQUEST_TIMEOUT_SEC,
        )
        research_request = _sanitize_research_request(
            out.get("research_request", {}),
            intent=intent,
            proposal={"target_nodes": _planner_focus_dimensions_from_kb(kb=kb, iteration=iteration, max_benchmarks=max_benchmarks)},
        )
        return ResearchRequestPlanDecision(
            reason=str(out.get("reason", "")).strip(),
            research_request=research_request,
            current_question=str(out.get("current_question", "")).strip()
            or _next_frontier_question(
                kb=kb,
                focus_dimensions=_planner_focus_dimensions_from_kb(kb=kb, iteration=iteration, max_benchmarks=max_benchmarks),
            ),
            planner=self.name,
            raw_response=json.dumps(out, indent=2),
        )

    def plan_proposal(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
        max_benchmarks: int,
        question_memo: str = "",
        research_memo: str = "",
    ) -> ProposalPlanDecision:
        compact_kb = _compact_planner_kb(
            kb=kb,
            max_nodes=max(6, max_benchmarks * 4),
            max_history_items=2,
            max_research_items=3,
        )
        payload = {
            "intent": intent,
            "knowledge_base": compact_kb,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "max_benchmarks": max_benchmarks,
            "question_memo": _trim_text(question_memo, 1200),
            "research_memo": _trim_text(research_memo, 3000),
        }
        payload = _enforce_payload_budget(
            payload,
            target_chars=PLANNER_INPUT_TARGET_CHARS,
            hard_cap_chars=PLANNER_INPUT_HARD_CAP_CHARS,
            trimmers=[_trim_planner_payload],
        )
        user_prompt = _render_planner_proposal_prompt(payload)
        memo = self._text_completion(
            system=(
                "You are a GPU performance-model planner. Answer briefly and directly in plain Markdown. "
                "Do not output JSON. Do not output executable code, commands, or profiler invocations."
            ),
            user=user_prompt,
            context="planner-proposal",
            timeout_sec=PLANNER_REQUEST_TIMEOUT_SEC,
        )
        proposal = _proposal_from_memo(
            memo,
            intent=intent,
            focus_nodes=_planner_focus_dimensions_from_kb(kb=kb, iteration=iteration, max_benchmarks=max_benchmarks),
            iteration=iteration,
        )
        focus = _sanitize_dimensions(_proposal_focus_nodes(proposal), [], max_benchmarks)
        if not focus:
            focus = _planner_focus_dimensions_from_kb(kb=kb, iteration=iteration, max_benchmarks=max_benchmarks)
        proposal = _sanitize_proposal(proposal, intent=intent, focus_nodes=focus, iteration=iteration)
        return ProposalPlanDecision(
            reason=_proposal_reason_from_memo(memo),
            proposal=proposal,
            current_question=_question_text_from_memo(question_memo) or _next_frontier_question(kb=kb, focus_dimensions=focus),
            planner=self.name,
            raw_response=memo,
        )

    def generate_implementation(
        self,
        intent: str,
        kb: dict[str, Any],
        plan: dict[str, Any],
        iteration: int,
        max_benchmarks: int,
        proposal_memo: str = "",
    ) -> ImplementationDecision:
        focus = _proposal_focus_nodes(plan.get("proposal", {}))
        if not focus:
            focus = [f"dimension_{iteration + 1}"]
        focus = focus[: max(1, max_benchmarks)]
        benchmarks_raw: list[dict[str, Any]] = []
        reasons: list[str] = []
        amendments: list[dict[str, Any]] = []
        raw_memos: list[str] = []
        for dim in focus:
            compact_plan = _compact_codegen_plan(plan=plan, dimension=dim)
            compact_kb = _compact_codegen_kb(kb=kb, dimension=dim)
            payload = {
                "intent": intent,
                "knowledge_base": compact_kb,
                "plan": compact_plan,
                "iteration": iteration,
                "dimension": dim,
                "proposal_memo": _trim_text(proposal_memo, 3000),
                "constraints": {
                    "bounded_runtime": "Each command should complete in <= 45 seconds when possible.",
                    "safety": "No destructive commands, no system configuration mutation.",
                    "no_inventory_only": "Do not return inventory/topology-only probes as benchmarks.",
                },
            }
            payload = _enforce_payload_budget(
                payload,
                target_chars=CODEGEN_INPUT_TARGET_CHARS,
                hard_cap_chars=CODEGEN_INPUT_HARD_CAP_CHARS,
                trimmers=[_trim_codegen_payload],
            )
            user_prompt = _render_codegen_prompt(payload)
            memo, bench, bench_amendments, repair_used = self._generate_benchmark_with_repair(
                intent=intent,
                dimension=dim,
                iteration=iteration,
                benchmark_index=len(benchmarks_raw),
                proposal=compact_plan.get("proposal", {}),
                user_prompt=user_prompt,
            )
            raw_memos.append(memo)
            reason = _codegen_reason_from_memo(memo)
            if repair_used:
                reason = f"{reason} [repair_attempt_used]"
            reasons.append(reason)
            benchmarks_raw.append(bench)
            amendments.extend(bench_amendments)
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
            raw_response="\n\n---\n\n".join([m for m in raw_memos if str(m).strip()]),
        )

    def _generate_benchmark_with_repair(
        self,
        *,
        intent: str,
        dimension: str,
        iteration: int,
        benchmark_index: int,
        proposal: dict[str, Any],
        user_prompt: str,
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]], bool]:
        memo = self._text_completion(
            system=CODEGEN_SYSTEM_PROMPT,
            user=user_prompt,
            max_output_tokens=None,
            timeout_sec=CODEGEN_TIMEOUT_SEC,
            context="codegen-memo",
        )
        bench, amendments = _benchmark_from_memo(
            memo,
            intent=intent,
            dimension=dimension,
            iteration=iteration,
            benchmark_index=benchmark_index,
            proposal=proposal,
        )
        repair_used = False
        for _ in range(CODEGEN_REPAIR_ATTEMPTS):
            if _benchmark_is_codegen_ready(bench):
                break
            repair_used = True
            repaired_memo = self._text_completion(
                system=CODEGEN_SYSTEM_PROMPT,
                user=_render_codegen_repair_prompt(
                    original_prompt=user_prompt,
                    previous_memo=memo,
                    amendments=amendments,
                ),
                max_output_tokens=None,
                timeout_sec=CODEGEN_TIMEOUT_SEC,
                context="codegen-memo",
            )
            memo = (
                f"{memo.strip()}\n\n--- CODEGEN REPAIR ATTEMPT ---\n\n{repaired_memo.strip()}"
                if str(memo).strip()
                else repaired_memo
            )
            bench, amendments = _benchmark_from_memo(
                repaired_memo,
                intent=intent,
                dimension=dimension,
                iteration=iteration,
                benchmark_index=benchmark_index,
                proposal=proposal,
            )
        return memo, bench, amendments, repair_used

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
            context="analysis-json",
            timeout_sec=ANALYSIS_TIMEOUT_SEC,
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

    def _json_completion(
        self,
        system: str,
        user: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        context: str = "completion",
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("OpenAI workflow backend requires `openai` package.") from exc

        api_key = _normalized_api_key()
        effective_timeout = max(self.request_timeout_sec, float(timeout_sec or self.request_timeout_sec))
        robust_timeout = _robust_response_threshold(context, effective_timeout)
        client = OpenAI(api_key=api_key or None, timeout=robust_timeout)
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
            resp = self._responses_create(client, request, context=context, timeout_sec=robust_timeout)
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

    def _text_completion(
        self,
        system: str,
        user: str,
        tools: list[dict[str, Any]] | None = None,
        max_output_tokens: int | None = 900,
        timeout_sec: float | None = None,
        context: str = "completion",
    ) -> str:
        try:
            from openai import OpenAI
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("OpenAI workflow backend requires `openai` package.") from exc

        api_key = _normalized_api_key()
        effective_timeout = max(self.request_timeout_sec, float(timeout_sec or self.request_timeout_sec))
        robust_timeout = _robust_response_threshold(context, effective_timeout)
        client = OpenAI(api_key=api_key or None, timeout=robust_timeout)
        request: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if max_output_tokens is not None:
            request["max_output_tokens"] = max(64, int(max_output_tokens))
        if tools:
            request["tools"] = tools
        resp = self._responses_create(client, request, context=context, timeout_sec=robust_timeout)
        return (resp.output_text or "").strip()

    def _repair_json(self, client: Any, malformed: str, schema_hint: dict[str, Any]) -> str:
        repair = self._responses_create(
            client,
            {
                "model": self.model,
                "max_output_tokens": 1400,
                "input": [
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
            },
            context="json-repair",
        )
        return (repair.output_text or "").strip()

    def _responses_create(self, client: Any, request: dict[str, Any], context: str, timeout_sec: float | None = None) -> Any:
        effective_timeout = timeout_sec if timeout_sec is not None else self.request_timeout_sec + 5.0
        try:
            return self._run_with_timeout(
                lambda: client.responses.create(**request),
                timeout_sec=effective_timeout,
                context=context,
            )
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, TimeoutError):
                raise
            if _is_retryable_openai_exception(exc):
                raise TimeoutError(
                    f"OpenAI {context} failed during a single long-running request "
                    f"(timeout budget {effective_timeout:.1f}s): {exc}"
                ) from exc
            raise

    def _run_with_timeout(self, fn: Any, timeout_sec: float, context: str) -> Any:
        timeout_sec = max(1.0, float(timeout_sec))
        if not hasattr(signal, "SIGALRM"):
            return fn()

        def _handle_timeout(_signum: int, _frame: Any) -> None:
            raise TimeoutError(f"OpenAI {context} timed out after {timeout_sec:.1f}s")

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handle_timeout)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_sec)
        try:
            return fn()
        finally:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
            signal.signal(signal.SIGALRM, previous_handler)


class ResilientWorkflowBackend(LLMWorkflowBackend):
    def __init__(self, primary: LLMWorkflowBackend, fallback: LLMWorkflowBackend) -> None:
        self.primary = primary
        self.fallback = fallback
        if isinstance(primary, OpenAIWorkflowBackend):
            self.research_timeout_retries = 0
            self.analysis_timeout_retries = 0
        else:
            self.research_timeout_retries = 1
            self.analysis_timeout_retries = 1
        self.timeout_diagnostics_enabled = os.getenv("GPU_PROFILER_TIMEOUT_DIAGNOSTICS", "1") != "0"

    def negotiate_schema(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
    ) -> ContractDecision:
        try:
            return self._call_primary("negotiate_schema", intent, kb, iteration, max_iterations)
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
            return self._call_primary("propose_plan", intent, kb, iteration, max_iterations, max_benchmarks)
        except Exception as exc:  # noqa: BLE001
            alt = self.fallback.propose_plan(intent, kb, iteration, max_iterations, max_benchmarks)
            alt.reason = f"Primary planning failed ({exc}); fallback used. {alt.reason}"
            alt.planner = f"{getattr(self.primary, 'name', 'primary')}->fallback:{alt.planner}"
            return alt

    def plan_research_request(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
        max_benchmarks: int,
    ) -> ResearchRequestPlanDecision:
        try:
            return self._call_primary("plan_research_request", intent, kb, iteration, max_iterations, max_benchmarks)
        except Exception as exc:  # noqa: BLE001
            alt = self.fallback.plan_research_request(intent, kb, iteration, max_iterations, max_benchmarks)
            alt.reason = f"Primary research-request planning failed ({exc}); fallback used. {alt.reason}"
            alt.planner = f"{getattr(self.primary, 'name', 'primary')}->fallback:{alt.planner}"
            return alt

    def plan_proposal(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        max_iterations: int,
        max_benchmarks: int,
        question_memo: str = "",
        research_memo: str = "",
    ) -> ProposalPlanDecision:
        try:
            return self._call_primary("plan_proposal", intent, kb, iteration, max_iterations, max_benchmarks, question_memo, research_memo)
        except Exception as exc:  # noqa: BLE001
            alt = self.fallback.plan_proposal(intent, kb, iteration, max_iterations, max_benchmarks, question_memo, research_memo)
            alt.reason = f"Primary proposal planning failed ({exc}); fallback used. {alt.reason}"
            alt.planner = f"{getattr(self.primary, 'name', 'primary')}->fallback:{alt.planner}"
            return alt

    def research_context(
        self,
        intent: str,
        kb: dict[str, Any],
        iteration: int,
        research_request: dict[str, Any] | None = None,
        research_request_memo: str = "",
        max_sources: int = 8,
    ) -> ResearchDecision:
        try:
            result, retry_count = self._call_primary_with_retries(
                "research_context",
                intent,
                kb,
                iteration,
                research_request,
                research_request_memo,
                max_sources,
                retries=self.research_timeout_retries,
            )
            if retry_count:
                result.reason = (
                    f"Primary research succeeded after {retry_count} timeout retry attempt(s). {result.reason}"
                )
            return result
        except Exception as exc:  # noqa: BLE001
            alt = self.fallback.research_context(intent, kb, iteration, research_request, research_request_memo, max_sources)
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
        proposal_memo: str = "",
    ) -> ImplementationDecision:
        try:
            return self._call_primary("generate_implementation", intent, kb, plan, iteration, max_benchmarks, proposal_memo)
        except Exception as exc:  # noqa: BLE001
            alt = self.fallback.generate_implementation(intent, kb, plan, iteration, max_benchmarks, proposal_memo)
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
            result, retry_count = self._call_primary_with_retries(
                "analyze_results",
                intent,
                kb,
                plan,
                execution_results,
                iteration,
                max_iterations,
                retries=self.analysis_timeout_retries,
            )
            if retry_count:
                result.reason = (
                    f"Primary analysis succeeded after {retry_count} timeout retry attempt(s). {result.reason}"
                )
            return result
        except Exception as exc:  # noqa: BLE001
            alt = self.fallback.analyze_results(intent, kb, plan, execution_results, iteration, max_iterations)
            alt.reason = f"Primary analysis failed ({exc}); fallback used. {alt.reason}"
            alt.planner = f"{getattr(self.primary, 'name', 'primary')}->fallback:{alt.planner}"
            return alt

    def _call_primary(self, method_name: str, *args: Any) -> Any:
        timeout_sec = self._primary_timeout_sec(method_name)
        if timeout_sec <= 0:
            return getattr(self.primary, method_name)(*args)

        ctx = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else multiprocessing.get_context()
        queue: Any = ctx.Queue()
        proc = ctx.Process(target=_invoke_backend_method, args=(self.primary, method_name, args, queue))
        proc.start()
        proc.join(timeout_sec)
        if proc.is_alive():
            proc.terminate()
            proc.join(5.0)
            raise TimeoutError(
                f"Primary backend {getattr(self.primary, 'name', 'primary')}:{method_name} timed out after {timeout_sec:.1f}s"
            )
        if not queue.empty():
            status, payload = queue.get()
            if status == "ok":
                return payload
            raise RuntimeError(str(payload))
        if proc.exitcode not in {0, None}:
            raise RuntimeError(
                f"Primary backend {getattr(self.primary, 'name', 'primary')}:{method_name} exited with code {proc.exitcode}"
            )
        raise RuntimeError(f"Primary backend {getattr(self.primary, 'name', 'primary')}:{method_name} returned no result")

    def _call_primary_with_retries(
        self,
        method_name: str,
        *args: Any,
        retries: int = 0,
    ) -> tuple[Any, int]:
        effective_retries = max(0, int(retries))
        retry_count = 0
        last_exc: Exception | None = None
        while True:
            try:
                return self._call_primary(method_name, *args), retry_count
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if retry_count >= effective_retries or not self._is_retryable_timeout(exc):
                    self._emit_timeout_progress(method_name, retry_count, effective_retries, exc, final=True)
                    self._launch_timeout_diagnostic(method_name, args, exc)
                    break
                self._emit_timeout_progress(method_name, retry_count, effective_retries, exc, final=False)
                retry_count += 1
        if retry_count and last_exc is not None:
            raise RuntimeError(
                f"{last_exc}; timeout retry exhausted after {retry_count} additional attempt(s)"
            ) from last_exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            f"Primary backend {getattr(self.primary, 'name', 'primary')}:{method_name} failed without a captured error"
        )

    def _emit_timeout_progress(
        self,
        method_name: str,
        retry_count: int,
        retries: int,
        exc: Exception,
        *,
        final: bool,
    ) -> None:
        sender = _method_timeout_sender(method_name)
        if not sender:
            return
        total_attempts = retries + 1
        if final:
            message = (
                f"OpenAI {method_name} timed out after attempt {retry_count + 1}/{total_attempts}. "
                f"Falling back. Detail: {exc}"
            )
        else:
            message = (
                f"OpenAI {method_name} timed out after attempt {retry_count + 1}/{total_attempts}. "
                f"Retrying now. Detail: {exc}"
            )
        _emit_live_agent_status(sender=sender, recipient="orchestrator", message=message)

    def _is_retryable_timeout(self, exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        message = f"{exc.__class__.__name__}: {exc}".lower()
        return "timed out" in message or "timeout" in message

    def _launch_timeout_diagnostic(self, method_name: str, args: tuple[Any, ...], exc: Exception) -> None:
        if not self.timeout_diagnostics_enabled or not self._is_retryable_timeout(exc):
            return
        try:
            _launch_openai_timeout_diagnostic(
                primary=self.primary,
                method_name=method_name,
                args=args,
                exc=exc,
            )
        except Exception:
            return

    def _primary_timeout_sec(self, method_name: str) -> float:
        raw = getattr(self.primary, "request_timeout_sec", 0.0)
        try:
            timeout_sec = float(raw)
        except Exception:
            return 0.0
        if timeout_sec <= 0:
            return 0.0
        if not isinstance(self.primary, OpenAIWorkflowBackend):
            if method_name == "generate_implementation":
                return max(CODEGEN_TIMEOUT_SEC + 15.0, timeout_sec + 62.0)
            return max(2.0, timeout_sec + 2.0)
        if method_name == "generate_implementation":
            attempt_timeout = max(CODEGEN_TIMEOUT_SEC + 5.0, timeout_sec + 5.0)
            return _robust_response_threshold("codegen-memo", attempt_timeout) + 5.0
        if method_name == "propose_plan":
            attempt_timeout = max(5.0, timeout_sec + 5.0)
            return _robust_response_threshold("planner-proposal", attempt_timeout) + 5.0
        if method_name == "research_context":
            attempt_timeout = max(RESEARCH_TIMEOUT_SEC + 5.0, timeout_sec + 5.0)
            return _robust_response_threshold("research-context", attempt_timeout) + 5.0
        if method_name == "analyze_results":
            attempt_timeout = max(ANALYSIS_TIMEOUT_SEC + 5.0, timeout_sec + 5.0)
            return _robust_response_threshold("analysis-json", attempt_timeout) + 5.0
        attempt_timeout = max(5.0, timeout_sec + 5.0)
        return _robust_response_threshold(_method_timeout_context(method_name), attempt_timeout) + 5.0


def _invoke_backend_method(backend: Any, method_name: str, args: tuple[Any, ...], queue: Any) -> None:
    try:
        result = getattr(backend, method_name)(*args)
        queue.put(("ok", result))
    except Exception as exc:  # noqa: BLE001
        queue.put(("err", f"{exc.__class__.__name__}: {exc}"))


def _normalized_api_key() -> str:
    return "".join(os.getenv("OPENAI_API_KEY", "").split())


def _launch_openai_timeout_diagnostic(
    primary: Any,
    method_name: str,
    args: tuple[Any, ...],
    exc: Exception,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    diag_root = repo_root / "openai_timeout_diagnostics"
    run_dir = diag_root / f"{time.strftime('%Y%m%d-%H%M%S')}-{method_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = _timeout_diagnostic_payload(method_name, args)
    event = {
        "method_name": method_name,
        "model": str(getattr(primary, "model", "")),
        "primary_name": str(getattr(primary, "name", "primary")),
        "request_timeout_sec": float(getattr(primary, "request_timeout_sec", 0.0) or 0.0),
        "exception_type": exc.__class__.__name__,
        "exception": str(exc),
        "payload": payload,
        "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "event.json").write_text(json.dumps(event, indent=2), encoding="utf-8")

    ping_script = repo_root / "openai_ping.py"
    if ping_script.exists():
        ping_cmd = [
            sys.executable,
            str(ping_script),
            "--model",
            str(getattr(primary, "model", "gpt-5.4")),
            "--timeout",
            str(max(5.0, float(getattr(primary, "request_timeout_sec", 0.0) or 0.0))),
            "--retries",
            "1",
            "--message",
            f"Reply in one sentence. This is a timeout diagnostic for {method_name}.",
        ]
        _spawn_background_command(
            ping_cmd,
            cwd=repo_root,
            stdout_path=run_dir / "openai_ping.stdout.txt",
            stderr_path=run_dir / "openai_ping.stderr.txt",
            env=_diagnostic_environment(primary),
        )

    probe_script = repo_root / "openai_agent_probe.py"
    query = str(payload.get("query", "")).strip()
    if probe_script.exists() and query and method_name in {
        "plan_research_request",
        "research_context",
        "plan_proposal",
        "generate_implementation",
        "analyze_results",
    }:
        base_timeout = float(getattr(primary, "request_timeout_sec", 0.0) or 0.0)
        if method_name == "research_context":
            base_timeout = max(RESEARCH_TIMEOUT_SEC, base_timeout)
        elif method_name == "analyze_results":
            base_timeout = max(ANALYSIS_TIMEOUT_SEC, base_timeout)
        else:
            base_timeout = max(15.0, base_timeout)
        probe_cmd = [
            sys.executable,
            str(probe_script),
            "--model",
            str(getattr(primary, "model", "gpt-5.4")),
            "--agent",
            method_name,
            "--timeout-value",
            str(base_timeout),
            "--timeout-value",
            str(base_timeout + 15.0),
            "--query",
            query,
            "--control-query",
            "How should I characterize latency and result quality for this OpenAI-backed GPU profiling agent call?",
            "--out",
            str(run_dir / "agent_probe_runs"),
        ]
        _spawn_background_command(
            probe_cmd,
            cwd=repo_root,
            stdout_path=run_dir / "agent_probe.stdout.txt",
            stderr_path=run_dir / "agent_probe.stderr.txt",
            env=_diagnostic_environment(primary),
        )


def _spawn_background_command(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    env: dict[str, str] | None = None,
) -> None:
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    try:
        subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
            env=env,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()


def _diagnostic_environment(primary: Any) -> dict[str, str]:
    env = {str(k): str(v) for k, v in os.environ.items()}
    api_key = _normalized_api_key()
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    model = str(getattr(primary, "model", "")).strip()
    if model:
        env.setdefault("GPU_PROFILER_OPENAI_MODEL", model)
    return env


def _timeout_diagnostic_payload(method_name: str, args: tuple[Any, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {"method_name": method_name}
    if method_name == "research_context":
        request = args[3] if len(args) > 3 and isinstance(args[3], dict) else {}
        memo = str(args[4]).strip() if len(args) > 4 else ""
        payload["intent"] = str(args[0]).strip() if args else ""
        payload["iteration"] = int(args[2]) if len(args) > 2 else 0
        payload["query"] = (
            memo
            or str(request.get("request_summary", "")).strip()
            or next((str(x).strip() for x in request.get("target_questions", []) if str(x).strip()), "")
        )
        payload["target_questions"] = [str(x).strip() for x in request.get("target_questions", []) if str(x).strip()]
        payload["max_sources"] = int(args[5]) if len(args) > 5 else 0
        return payload
    if method_name == "plan_research_request":
        payload["intent"] = str(args[0]).strip() if args else ""
        payload["query"] = payload["intent"]
        payload["iteration"] = int(args[2]) if len(args) > 2 else 0
        return payload
    if method_name == "plan_proposal":
        payload["intent"] = str(args[0]).strip() if args else ""
        payload["iteration"] = int(args[2]) if len(args) > 2 else 0
        payload["query"] = str(args[5]).strip() if len(args) > 5 and str(args[5]).strip() else payload["intent"]
        return payload
    if method_name == "generate_implementation":
        payload["intent"] = str(args[0]).strip() if args else ""
        plan = args[2] if len(args) > 2 and isinstance(args[2], dict) else {}
        proposal = plan.get("proposal", {}) if isinstance(plan.get("proposal", {}), dict) else {}
        first_proposal = proposal.get("proposals", [None])[0] if isinstance(proposal.get("proposals", []), list) else None
        payload["query"] = str(args[5]).strip() if len(args) > 5 and str(args[5]).strip() else ""
        if not payload["query"] and isinstance(first_proposal, dict):
            payload["query"] = str(first_proposal.get("title", "")).strip() or str(first_proposal.get("objective", "")).strip()
        if not payload["query"]:
            payload["query"] = payload["intent"]
        payload["iteration"] = int(args[3]) if len(args) > 3 else 0
        return payload
    if method_name == "analyze_results":
        payload["intent"] = str(args[0]).strip() if args else ""
        payload["query"] = payload["intent"]
        payload["iteration"] = int(args[4]) if len(args) > 4 else 0
        return payload
    payload["arg_types"] = [type(item).__name__ for item in args]
    payload["payload_bytes"] = len(json.dumps([_json_safe(item) for item in args]))
    return payload


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _method_timeout_context(method_name: str) -> str:
    mapping = {
        "negotiate_schema": "schema-contract",
        "plan_research_request": "plan-research-request",
        "plan_proposal": "planner-proposal",
        "research_context": "research-context",
        "generate_implementation": "codegen-memo",
        "analyze_results": "analysis-json",
    }
    return mapping.get(method_name, "completion")


def _method_timeout_sender(method_name: str) -> str:
    mapping = {
        "plan_research_request": "llm-planner",
        "plan_proposal": "llm-planner",
        "research_context": "llm-research",
        "generate_implementation": "llm-codegen",
        "analyze_results": "llm-analysis",
    }
    return mapping.get(method_name, "")


def _emit_live_agent_status(sender: str, recipient: str, message: str) -> None:
    if not sender or not recipient or not str(message).strip():
        return
    stream = sys.stdout if getattr(sys.stdout, "isatty", lambda: False)() else sys.stderr
    print(f"{sender} -> {recipient}: message: {message}", file=stream, flush=True)


def _robust_response_threshold(context: str, attempt_timeout_sec: float) -> float:
    floor = OPENAI_RESPONSE_DEFAULT_THRESHOLD_SEC
    if context == "research-context":
        floor = OPENAI_RESPONSE_RESEARCH_THRESHOLD_SEC
    elif context == "analysis-json":
        floor = OPENAI_RESPONSE_ANALYSIS_THRESHOLD_SEC
    elif context == "codegen-memo":
        floor = OPENAI_RESPONSE_CODEGEN_THRESHOLD_SEC
    return max(float(floor), float(attempt_timeout_sec) * OPENAI_RESPONSE_ROBUST_THRESHOLD_MULTIPLIER)


def _is_retryable_openai_exception(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    if "timeout" in message or "timed out" in message:
        return True
    retryable_names = {"apiconnectionerror", "internalservererror", "ratelimiterror"}
    if name in retryable_names:
        return True
    retryable_signals = [
        "connection reset",
        "temporarily unavailable",
        "server error",
        "rate limit",
        "429",
        "502",
        "503",
        "504",
    ]
    return any(signal_text in message for signal_text in retryable_signals)


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("LLM output did not contain JSON object")
    return json.loads(match.group(0))


def _current_or_default_knowledge_model(kb: dict[str, Any], intent: str) -> dict[str, Any]:
    model = kb.get("current_knowledge_model", {}) if isinstance(kb.get("current_knowledge_model", {}), dict) else {}
    if isinstance(model.get("domain_hierarchy"), list):
        return model
    return _default_knowledge_model(intent=intent, focus_dimensions=[])


def _planner_focus_dimensions_from_kb(kb: dict[str, Any], iteration: int, max_benchmarks: int) -> list[str]:
    model = kb.get("current_knowledge_model", {}) if isinstance(kb.get("current_knowledge_model", {}), dict) else {}
    hierarchy = model.get("domain_hierarchy", []) if isinstance(model.get("domain_hierarchy", []), list) else []
    focus: list[str] = []
    for item in hierarchy:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name and name not in focus:
            focus.append(name)
        if len(focus) >= max(1, max_benchmarks):
            return focus
    target = kb.get("target_dimensions", []) if isinstance(kb.get("target_dimensions", []), list) else []
    covered = set(str(x).strip() for x in kb.get("covered_dimensions", []) if str(x).strip())
    uncovered = [str(x).strip() for x in target if str(x).strip() and str(x).strip() not in covered]
    focus.extend([dim for dim in uncovered if dim not in focus][: max(1, max_benchmarks) - len(focus)])
    if focus:
        return focus[: max(1, max_benchmarks)]
    return [f"dimension_{iteration + 1}"]


def _serialized_size_chars(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True, default=str))


def _trim_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _enforce_payload_budget(
    payload: dict[str, Any],
    target_chars: int,
    hard_cap_chars: int,
    trimmers: list[Any] | None = None,
) -> dict[str, Any]:
    current = payload
    if _serialized_size_chars(current) <= target_chars:
        return current
    for trimmer in trimmers or []:
        current = trimmer(current, target_chars)
        if _serialized_size_chars(current) <= hard_cap_chars:
            break
    size = _serialized_size_chars(current)
    if size > hard_cap_chars:
        raise ValueError(f"LLM payload exceeded hard cap: {size} > {hard_cap_chars}")
    return current


def _compact_planner_kb(
    kb: dict[str, Any],
    max_nodes: int = 8,
    max_history_items: int = 2,
    max_research_items: int = 3,
) -> dict[str, Any]:
    target_dimensions = _sanitize_dimensions(kb.get("target_dimensions", []), [], max_nodes)
    covered_dimensions = _sanitize_dimensions(kb.get("covered_dimensions", []), [], max_nodes)
    available_tools = {
        str(name): bool(enabled)
        for name, enabled in (kb.get("available_tools", {}) if isinstance(kb.get("available_tools", {}), dict) else {}).items()
        if enabled
    }
    current_model = kb.get("current_knowledge_model", {}) if isinstance(kb.get("current_knowledge_model", {}), dict) else {}
    compact_model = {
        "focus_nodes": [str(x).strip() for x in current_model.get("focus_nodes", []) if str(x).strip()][:max_nodes],
        "domain_hierarchy": _compact_domain_hierarchy(current_model.get("domain_hierarchy", []), max_nodes=max_nodes),
    }
    current_proposal = kb.get("current_proposal", {}) if isinstance(kb.get("current_proposal", {}), dict) else {}
    compact_proposal = {
        "target_nodes": [str(x).strip() for x in current_proposal.get("target_nodes", []) if str(x).strip()][:max_nodes],
        "proposals": _compact_proposals(current_proposal.get("proposals", []), max_items=max_nodes),
    }
    history = kb.get("history", []) if isinstance(kb.get("history", []), list) else []
    compact_history = []
    for item in history[-max_history_items:]:
        if not isinstance(item, dict):
            continue
        compact_history.append(
            {
                "iteration": item.get("iteration"),
                "summary": _trim_text(item.get("summary", ""), 160),
                "coverage_score": item.get("coverage_score"),
                "claims_added": item.get("claims_added"),
                "open_gaps": [str(x).strip() for x in item.get("required_observability", []) if str(x).strip()][:4],
            }
        )
    research_history = kb.get("research_history", []) if isinstance(kb.get("research_history", []), list) else []
    compact_research = []
    for item in research_history[-max_history_items:]:
        if not isinstance(item, dict):
            continue
        findings = item.get("findings", []) if isinstance(item.get("findings", []), list) else []
        compact_research.append(
            {
                "iteration": item.get("iteration"),
                "request_summary": _trim_text(item.get("request_summary", ""), 160),
                "proposed_dimensions": _sanitize_dimensions(item.get("proposed_dimensions", []), [], max_nodes),
                "findings": [
                    {
                        "title": _trim_text(finding.get("title", ""), 80),
                        "summary": _trim_text(finding.get("summary", ""), 160),
                        "source_url": _trim_text(finding.get("source_url", ""), 120),
                    }
                    for finding in findings[:max_research_items]
                    if isinstance(finding, dict)
                ],
            }
        )
    pending_amendments = kb.get("pending_contract_amendments", []) if isinstance(kb.get("pending_contract_amendments", []), list) else []
    compact_amendments = [
        {
            "path": _trim_text(item.get("path", ""), 80),
            "change": _trim_text(item.get("change", ""), 160),
            "priority": _trim_text(item.get("priority", ""), 16),
        }
        for item in pending_amendments[:4]
        if isinstance(item, dict)
    ]
    knowledge_book_excerpt = _trim_text(kb.get("knowledge_base_book_memo", ""), 1600)
    frontier_excerpt = _trim_text(kb.get("knowledge_base_frontier_memo", ""), 1200)
    frontier_questions = [
        _trim_text(item, 220)
        for item in (kb.get("knowledge_base_frontier_questions", []) if isinstance(kb.get("knowledge_base_frontier_questions", []), list) else [])
        if str(item).strip()
    ][:8]
    frontier_candidates = []
    for item in (kb.get("knowledge_base_frontier_candidates", []) if isinstance(kb.get("knowledge_base_frontier_candidates", []), list) else [])[:8]:
        if not isinstance(item, dict):
            continue
        frontier_candidates.append(
            {
                "question": _trim_text(item.get("question", ""), 220),
                "source": _trim_text(item.get("source", ""), 24),
                "section_refs": [_trim_text(ref, 80) for ref in item.get("section_refs", []) if str(ref).strip()][:3],
                "unmet_frontier_criteria": [
                    _trim_text(ref, 160)
                    for ref in item.get("unmet_frontier_criteria", [])
                    if str(ref).strip()
                ][:3],
                "unsatisfied_prerequisites": [
                    _trim_text(ref, 80)
                    for ref in item.get("unsatisfied_prerequisites", [])
                    if str(ref).strip()
                ][:3],
            }
        )
    return {
        "intent": _trim_text(kb.get("intent", ""), 160),
        "target_dimensions": target_dimensions,
        "covered_dimensions": covered_dimensions,
        "coverage_score": kb.get("coverage_score", 0.0),
        "target_coverage": kb.get("target_coverage", 0.0),
        "available_tools": available_tools,
        "current_knowledge_model": compact_model,
        "current_proposal": compact_proposal,
        "history": compact_history,
        "research_history": compact_research,
        "pending_contract_amendments": compact_amendments,
        "knowledge_base_book_excerpt": knowledge_book_excerpt,
        "knowledge_base_frontier_excerpt": frontier_excerpt,
        "knowledge_base_frontier_questions": frontier_questions,
        "knowledge_base_frontier_candidates": frontier_candidates,
    }


def _compact_search_kb(kb: dict[str, Any]) -> dict[str, Any]:
    tools = {
        str(name): bool(enabled)
        for name, enabled in (kb.get("available_tools", {}) if isinstance(kb.get("available_tools", {}), dict) else {}).items()
        if enabled
    }
    return {
        "intent": _trim_text(kb.get("intent", ""), 160),
        "available_tools": tools,
        "target_dimensions": _sanitize_dimensions(kb.get("target_dimensions", []), [], 8),
        "current_knowledge_model": {
            "focus_nodes": [
                str(x).strip()
                for x in (
                    kb.get("current_knowledge_model", {})
                    if isinstance(kb.get("current_knowledge_model", {}), dict)
                    else {}
                ).get("focus_nodes", [])
                if str(x).strip()
            ][:6]
        },
    }


def _compact_research_request(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict):
        return {}
    return {
        "intent_summary": _trim_text(request.get("intent_summary", ""), 160),
        "request_summary": _trim_text(request.get("request_summary", ""), 220),
        "target_nodes": [str(x).strip() for x in request.get("target_nodes", []) if str(x).strip()][:8],
        "target_questions": [str(x).strip() for x in request.get("target_questions", []) if str(x).strip()][:6],
        "search_topics": [str(x).strip() for x in request.get("search_topics", []) if str(x).strip()][:8],
        "source_preferences": [str(x).strip() for x in request.get("source_preferences", []) if str(x).strip()][:6],
        "expected_outputs": [str(x).strip() for x in request.get("expected_outputs", []) if str(x).strip()][:6],
    }


def _next_frontier_question(kb: dict[str, Any], focus_dimensions: list[str]) -> str:
    explicit_candidates = kb.get("knowledge_base_frontier_candidates", [])
    if isinstance(explicit_candidates, list):
        for item in explicit_candidates:
            if not isinstance(item, dict):
                continue
            text = _frontier_candidate_question(item)
            if text:
                return text
    explicit_questions = kb.get("knowledge_base_frontier_questions", [])
    if isinstance(explicit_questions, list):
        for item in explicit_questions:
            text = str(item).strip()
            if text:
                return text
    frontier = str(kb.get("knowledge_base_frontier_memo", "") or kb.get("knowledge_base_frontier_excerpt", "")).strip()
    for raw_line in frontier.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^\d+\.\s+", line):
            return re.sub(r"^\d+\.\s+", "", line).strip()
    for item in focus_dimensions:
        text = str(item).strip()
        if text:
            return f"What is the next bottom-up question needed to characterize {text} on this GPU?"
    return "What is the next bottom-up question needed to extend the current known frontier of this GPU knowledge base?"


def _frontier_candidate_question(item: dict[str, Any]) -> str:
    section_refs = [str(ref).strip() for ref in item.get("section_refs", []) if str(ref).strip()]
    section = section_refs[0] if section_refs else "this section"
    unmet_prereqs = [str(ref).strip() for ref in item.get("unsatisfied_prerequisites", []) if str(ref).strip()]
    if unmet_prereqs:
        return f"What prerequisite evidence is still needed before {section} can advance, starting with {unmet_prereqs[0]}?"
    unmet_criteria = [str(ref).strip() for ref in item.get("unmet_frontier_criteria", []) if str(ref).strip()]
    if unmet_criteria:
        return f"For {section}, what evidence is still needed to satisfy this frontier criterion: {unmet_criteria[0]}?"
    return str(item.get("question", "")).strip()


def _question_text_from_memo(question_memo: str) -> str:
    for raw_line in str(question_memo or "").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and not line.lower().startswith("summary") and not line.lower().startswith("why this question matters") and not line.lower().startswith("frontier source"):
            return line
    return ""


def _render_planner_proposal_prompt(payload: dict[str, Any]) -> str:
    kb = payload.get("knowledge_base", {}) if isinstance(payload.get("knowledge_base", {}), dict) else {}
    model = kb.get("current_knowledge_model", {}) if isinstance(kb.get("current_knowledge_model", {}), dict) else {}
    tools = ", ".join(sorted(str(k) for k, v in (kb.get("available_tools", {}) or {}).items() if v)) or "none"
    focus_nodes = ", ".join(str(x) for x in model.get("focus_nodes", []) if str(x).strip()) or "none yet"
    covered = ", ".join(str(x) for x in kb.get("covered_dimensions", []) if str(x).strip()) or "none yet"
    question_memo = str(payload.get("question_memo", "")).strip() or "No explicit planner question recorded."
    research_memo = str(payload.get("research_memo", "")).strip() or "No external findings yet."
    frontier_memo = str(kb.get("knowledge_base_frontier_excerpt", "")).strip() or "No frontier memo recorded."
    knowledge_book_excerpt = str(kb.get("knowledge_base_book_excerpt", "")).strip() or "No knowledge-base excerpt recorded."
    frontier_questions = "\n".join(f"- {item}" for item in kb.get("knowledge_base_frontier_questions", []) if str(item).strip()) or "- none recorded"
    frontier_candidates = "\n".join(
        f"- {item.get('question', '')} | source={item.get('source', '')} | refs={item.get('section_refs', [])} | unmet_criteria={item.get('unmet_frontier_criteria', [])} | unsatisfied_prerequisites={item.get('unsatisfied_prerequisites', [])}"
        for item in kb.get("knowledge_base_frontier_candidates", [])
        if isinstance(item, dict) and str(item.get("question", "")).strip()
    ) or "- none recorded"
    return (
        f"Intent: {payload.get('intent', '')}\n"
        f"Iteration: {payload.get('iteration', 0)} of {payload.get('max_iterations', 1)}\n"
        f"Available tools: {tools}\n"
        f"Current focus nodes: {focus_nodes}\n"
        f"Covered dimensions so far: {covered}\n\n"
        f"Knowledge-base excerpt:\n{knowledge_book_excerpt}\n\n"
        f"Frontier memo:\n{frontier_memo}\n\n"
        f"Frontier questions:\n{frontier_questions}\n\n"
        f"Ranked frontier candidates:\n{frontier_candidates}\n\n"
        f"Selected question:\n{question_memo}\n\n"
        f"Research memo:\n{research_memo}\n\n"
        "Task: Propose the single best next benchmark proposal memo that directly answers the selected question.\n"
        "Requirements:\n"
        "- non-executable\n"
        "- one benchmark only\n"
        "- domain-specific\n"
        "- precise\n"
        "- concise\n"
        "- curriculum-first\n"
        "Output sections exactly:\n"
        "1. Target Dimension\n"
        "2. Why This First\n"
        "3. Benchmark Idea\n"
        "4. Required Evidence\n"
        "5. What Success Unlocks\n"
    )


def _render_codegen_prompt(payload: dict[str, Any]) -> str:
    kb = payload.get("knowledge_base", {}) if isinstance(payload.get("knowledge_base", {}), dict) else {}
    plan = payload.get("plan", {}) if isinstance(payload.get("plan", {}), dict) else {}
    proposal = plan.get("proposal", {}) if isinstance(plan.get("proposal", {}), dict) else {}
    proposal_items = proposal.get("proposals", []) if isinstance(proposal.get("proposals", []), list) else []
    item = proposal_items[0] if proposal_items else {}
    current_question = str(plan.get("current_question", "")).strip() or "No current question recorded."
    tools = ", ".join(sorted(str(k) for k, v in (kb.get("available_tools", {}) or {}).items() if v)) or "none"
    relevant_nodes = kb.get("relevant_knowledge_nodes", []) if isinstance(kb.get("relevant_knowledge_nodes", []), list) else []
    node_summary = "\n".join(
        f"- {str(node.get('name', '')).strip()}: {str(node.get('description', '')).strip()}"
        for node in relevant_nodes
        if isinstance(node, dict) and str(node.get("name", "")).strip()
    ) or "- none"
    return (
        f"Intent: {payload.get('intent', '')}\n"
        f"Iteration: {payload.get('iteration', 0)}\n"
        f"Target dimension: {payload.get('dimension', '')}\n"
        f"Available tools: {tools}\n\n"
        "Current frontier question:\n"
        f"{current_question}\n\n"
        "Relevant knowledge:\n"
        f"{node_summary}\n\n"
        "Planning context:\n"
        f"{str(payload.get('proposal_memo', '')).strip() or 'No additional planning context provided.'}\n\n"
        "Relevant internal benchmark-plan item:\n"
        f"- title: {str(item.get('title', '')).strip()}\n"
        f"- objective: {str(item.get('objective', '')).strip()}\n"
        f"- benchmark role: {str(item.get('benchmark_role', '')).strip()}\n"
        f"- required evidence: {', '.join(str(x) for x in item.get('required_evidence', []) if str(x).strip()) or 'none specified'}\n"
        f"- rationale: {str(item.get('rationale', '')).strip()}\n\n"
        "Task: Write a concise implementation memo for exactly one CUDA benchmark.\n"
        "Requirements:\n"
        "- domain-specific\n"
        "- precise\n"
        "- concise\n"
        "- one CUDA source file only\n"
        "- one shell command only\n"
        "- bounded runtime when possible\n"
        "- no Python benchmark scripts\n"
        "Output sections exactly:\n"
        "1. Implementation Summary\n"
        "2. CUDA Source File\n"
        "3. Build and Run Command\n"
        "4. Validation Checks\n"
        "5. Feasibility and Risks\n"
        "\n"
        "In section 2, include:\n"
        "Path: <relative path ending in .cu>\n"
        "then one fenced ```cuda code block.\n"
        "In section 3, include one fenced ```bash code block with the command.\n"
        "In section 5, include lines starting with:\n"
        "- Feasibility: feasible|feasible_with_revision|not_feasible\n"
        "- Complexity: low|medium|high|excessive\n"
    )


def _compact_codegen_kb(kb: dict[str, Any], dimension: str) -> dict[str, Any]:
    available_tools = {
        str(name): bool(enabled)
        for name, enabled in (kb.get("available_tools", {}) if isinstance(kb.get("available_tools", {}), dict) else {}).items()
        if enabled
    }
    current_model = kb.get("current_knowledge_model", {}) if isinstance(kb.get("current_knowledge_model", {}), dict) else {}
    relevant_nodes = _filter_domain_nodes(current_model.get("domain_hierarchy", []), needle=dimension, max_nodes=4)
    schema_contract = kb.get("schema_contract", {}) if isinstance(kb.get("schema_contract", {}), dict) else {}
    negotiation = schema_contract.get("negotiation_policy", {}) if isinstance(schema_contract.get("negotiation_policy", {}), dict) else {}
    return {
        "intent": _trim_text(kb.get("intent", ""), 160),
        "target_dimensions": _sanitize_dimensions(kb.get("target_dimensions", []), [], 8),
        "available_tools": available_tools,
        "relevant_knowledge_nodes": relevant_nodes,
        "schema_contract": {
            "negotiation_policy": {
                "thresholds": negotiation.get("thresholds", {}),
                "weights": negotiation.get("weights", {}),
                "max_amendment_rounds": negotiation.get("max_amendment_rounds"),
            }
        },
    }


def _compact_codegen_plan(plan: dict[str, Any], dimension: str) -> dict[str, Any]:
    proposal = plan.get("proposal", {}) if isinstance(plan.get("proposal", {}), dict) else {}
    relevant_items = []
    for item in proposal.get("proposals", []):
        if not isinstance(item, dict):
            continue
        targets = [str(x).strip() for x in item.get("target_node_ids", []) if str(x).strip()]
        haystack = " ".join(
            [str(item.get("title", "")), str(item.get("objective", "")), str(item.get("description", "")), " ".join(targets)]
        ).lower()
        if dimension.lower() in haystack or not relevant_items:
            relevant_items.append(
                {
                    "id": str(item.get("id", "")).strip(),
                    "title": _trim_text(item.get("title", ""), 120),
                    "objective": _trim_text(item.get("objective", ""), 180),
                    "target_node_ids": targets[:4],
                    "benchmark_role": _trim_text(item.get("benchmark_role", ""), 24),
                    "hypothesis": _trim_text(item.get("hypothesis", ""), 180),
                    "required_evidence": [str(x).strip() for x in item.get("required_evidence", []) if str(x).strip()][:4],
                    "rationale": _trim_text(item.get("rationale", ""), 180),
                }
            )
        if len(relevant_items) >= 2:
            break
    knowledge_model = plan.get("knowledge_model", {}) if isinstance(plan.get("knowledge_model", {}), dict) else {}
    return {
        "iteration": plan.get("iteration"),
        "planner": _trim_text(plan.get("planner", ""), 64),
        "current_question": _trim_text(plan.get("current_question", ""), 220),
        "proposal": {
            "intent_summary": _trim_text(proposal.get("intent_summary", ""), 160),
            "proposal_summary": _trim_text(proposal.get("proposal_summary", ""), 180),
            "target_nodes": [str(x).strip() for x in proposal.get("target_nodes", []) if str(x).strip()][:6],
            "proposals": relevant_items,
        },
        "knowledge_model": {
            "focus_nodes": [str(x).strip() for x in knowledge_model.get("focus_nodes", []) if str(x).strip()][:6],
            "domain_hierarchy": _filter_domain_nodes(knowledge_model.get("domain_hierarchy", []), needle=dimension, max_nodes=4),
        },
    }


def _compact_domain_hierarchy(raw_nodes: list[Any], max_nodes: int) -> list[dict[str, Any]]:
    compact = []
    for item in raw_nodes[: max_nodes * 2]:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("id", "")).strip()
        name = str(item.get("name", "")).strip()
        if not node_id or not name:
            continue
        compact.append(
            {
                "id": node_id,
                "name": name,
                "node_type": _trim_text(item.get("node_type", ""), 24),
                "status": _trim_text(item.get("status", ""), 32),
                "open_gaps": [str(x).strip() for x in item.get("open_gaps", []) if str(x).strip()][:2],
            }
        )
        if len(compact) >= max_nodes:
            break
    return compact


def _compact_proposals(raw_proposals: list[Any], max_items: int) -> list[dict[str, Any]]:
    compact = []
    for item in raw_proposals[: max_items * 2]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "id": str(item.get("id", "")).strip(),
                "title": _trim_text(item.get("title", ""), 120),
                "objective": _trim_text(item.get("objective", ""), 180),
                "target_node_ids": [str(x).strip() for x in item.get("target_node_ids", []) if str(x).strip()][:4],
                "benchmark_role": _trim_text(item.get("benchmark_role", ""), 24),
            }
        )
        if len(compact) >= max_items:
            break
    return compact


def _filter_domain_nodes(raw_nodes: list[Any], needle: str, max_nodes: int) -> list[dict[str, Any]]:
    needle_l = needle.lower().strip()
    matched = []
    fallback = []
    for item in raw_nodes[: max_nodes * 3]:
        if not isinstance(item, dict):
            continue
        compact = {
            "id": str(item.get("id", "")).strip(),
            "name": str(item.get("name", "")).strip(),
            "description": _trim_text(item.get("description", ""), 160),
            "status": _trim_text(item.get("status", ""), 32),
            "open_gaps": [str(x).strip() for x in item.get("open_gaps", []) if str(x).strip()][:2],
        }
        haystack = " ".join([compact["id"], compact["name"], compact["description"]]).lower()
        if needle_l and needle_l in haystack:
            matched.append(compact)
        elif len(fallback) < max_nodes:
            fallback.append(compact)
        if len(matched) >= max_nodes:
            break
    return (matched or fallback)[:max_nodes]


def _trim_planner_payload(payload: dict[str, Any], _target_chars: int) -> dict[str, Any]:
    out = json.loads(json.dumps(payload))
    kb = out.get("knowledge_base", {}) if isinstance(out.get("knowledge_base", {}), dict) else {}
    if isinstance(kb.get("research_history"), list):
        kb["research_history"] = kb["research_history"][:1]
    if isinstance(kb.get("history"), list):
        kb["history"] = kb["history"][:1]
    model = kb.get("current_knowledge_model", {}) if isinstance(kb.get("current_knowledge_model", {}), dict) else {}
    if isinstance(model.get("domain_hierarchy"), list):
        model["domain_hierarchy"] = model["domain_hierarchy"][:4]
    proposal = kb.get("current_proposal", {}) if isinstance(kb.get("current_proposal", {}), dict) else {}
    if isinstance(proposal.get("proposals"), list):
        proposal["proposals"] = proposal["proposals"][:2]
    return out


def _trim_codegen_payload(payload: dict[str, Any], _target_chars: int) -> dict[str, Any]:
    out = json.loads(json.dumps(payload))
    kb = out.get("knowledge_base", {}) if isinstance(out.get("knowledge_base", {}), dict) else {}
    nodes = kb.get("relevant_knowledge_nodes")
    if isinstance(nodes, list):
        kb["relevant_knowledge_nodes"] = nodes[:2]
    plan = out.get("plan", {}) if isinstance(out.get("plan", {}), dict) else {}
    proposal = plan.get("proposal", {}) if isinstance(plan.get("proposal", {}), dict) else {}
    if isinstance(proposal.get("proposals"), list):
        proposal["proposals"] = proposal["proposals"][:1]
    return out


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


def _proposal_reason_from_memo(memo: str) -> str:
    section = _extract_numbered_section(memo, "Why This First")
    return section or "Planner produced proposal memo."


def _codegen_reason_from_memo(memo: str) -> str:
    section = _extract_numbered_section(memo, "Implementation Summary")
    return _first_nonempty_line(section) or "Code generator produced implementation memo."


def _proposal_from_memo(memo: str, intent: str, focus_nodes: list[str], iteration: int) -> dict[str, Any]:
    target_dimension = _first_nonempty_line(_extract_numbered_section(memo, "Target Dimension"))
    why_first = _extract_numbered_section(memo, "Why This First")
    benchmark_idea = _extract_numbered_section(memo, "Benchmark Idea")
    required_evidence = _extract_bullets_or_lines(_extract_numbered_section(memo, "Required Evidence"))
    success_unlocks = _extract_bullets_or_lines(_extract_numbered_section(memo, "What Success Unlocks"))
    if not target_dimension:
        return _default_proposal(intent=intent, focus_dimensions=focus_nodes, iteration=iteration)
    target_node = _slugify_dimension(target_dimension) or (focus_nodes[0] if focus_nodes else f"feature_{iteration}")
    return {
        "intent_summary": intent,
        "proposal_summary": benchmark_idea or f"Initial benchmark proposal for {target_dimension}.",
        "target_nodes": [target_node],
        "proposals": [
            {
                "id": f"proposal_{iteration}_0",
                "title": f"Baseline benchmark for {target_dimension}",
                "objective": benchmark_idea or f"Establish first evidence for {target_dimension}.",
                "target_node_ids": [target_node],
                "priority": "high",
                "benchmark_role": "baseline",
                "description": benchmark_idea,
                "hypothesis": why_first,
                "required_evidence": required_evidence,
                "rationale": why_first,
                "prerequisites": [],
                "next_if_success": success_unlocks,
                "next_if_failure": [f"Simplify or clarify the benchmark design for {target_dimension}."],
            }
        ],
        "planner_notes": _trim_text(memo, 1200),
        "generated_at": "",
    }


def _benchmark_from_memo(
    memo: str,
    intent: str,
    dimension: str,
    iteration: int,
    benchmark_index: int,
    proposal: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _ = intent
    proposal_items = proposal.get("proposals", []) if isinstance(proposal.get("proposals", []), list) else []
    item = proposal_items[0] if proposal_items else {}
    summary = _extract_numbered_section(memo, "Implementation Summary")
    source_section = _extract_numbered_section(memo, "CUDA Source File")
    command_section = _extract_numbered_section(memo, "Build and Run Command")
    validation_section = _extract_numbered_section(memo, "Validation Checks")
    feasibility_section = _extract_numbered_section(memo, "Feasibility and Risks")

    source_path = _extract_labeled_value(source_section, "Path") or f"generated/iter_{iteration:02d}/benchmark_{benchmark_index:02d}.cu"
    if not source_path.endswith(".cu"):
        source_path = f"generated/iter_{iteration:02d}/benchmark_{benchmark_index:02d}.cu"
    truncated_source = _has_unclosed_code_fence(source_section)
    source_code = _extract_first_code_block(source_section, preferred_langs=("cuda", "cpp", "c++", "c"))
    source_missing = not bool(source_code)
    command = _extract_first_code_block(command_section, preferred_langs=("bash", "sh", "shell"))
    if not command:
        binary_path = f"./generated/iter_{iteration:02d}/benchmark_{benchmark_index:02d}"
        command = f"nvcc -O3 {source_path} -o {binary_path} && {binary_path}"
    command = " ".join(command.split())

    validation_checks = _extract_bullets_or_lines(validation_section)
    feasibility = _extract_labeled_value(feasibility_section, "Feasibility").lower()
    complexity = _extract_labeled_value(feasibility_section, "Complexity").lower()
    if feasibility not in {"feasible", "feasible_with_revision", "not_feasible"}:
        feasibility = "feasible"
    if complexity not in {"low", "medium", "high", "excessive"}:
        complexity = "medium"
    risks = _extract_remaining_lines(
        feasibility_section,
        exclude_prefixes=("Feasibility:", "Complexity:", "Risks:", "- Feasibility:", "- Complexity:", "- Risks:"),
    )

    amendment_list = []
    if truncated_source or source_missing:
        feasibility = "not_feasible"
        complexity = "excessive" if truncated_source else ("high" if complexity in {"low", "medium"} else complexity)
        amendment_list.append(
            {
                "path": "implementation.cuda_source",
                "change": "Return one complete fenced CUDA code block in section 2 and ensure the file is not truncated.",
                "rationale": "The codegen response did not include a complete CUDA source block." if truncated_source else "The codegen response did not include a CUDA source block.",
                "priority": "high",
            }
        )
    if feasibility in {"feasible_with_revision", "not_feasible"} or complexity in {"high", "excessive"}:
        amendment_list.append(
            {
                "path": "proposal.proposals[0]",
                "change": "Clarify or simplify the implementation scope before the next codegen attempt.",
                "rationale": "; ".join(risks) if risks else f"Implementation memo reported {feasibility} with {complexity} complexity.",
                "priority": "high" if feasibility == "not_feasible" or complexity == "excessive" else "medium",
            }
        )

    benchmark_id = str(item.get("id", "")).strip() or f"proposal_{iteration}_{benchmark_index}"
    title = str(item.get("title", "")).strip() or f"Benchmark for {dimension}"
    hypothesis = str(item.get("hypothesis", "")).strip() or _first_nonempty_line(summary) or title
    analysis_method = {
        "summary": _first_nonempty_line(summary) or "Validate the benchmark run and inspect emitted evidence.",
        "metrics": validation_checks[:6] or ["stdout-reported throughput", "runtime_sec"],
        "decision_logic": "; ".join(validation_checks[:4]) or "Use successful build/run plus reported measurements to judge benchmark validity.",
    }
    scores = _scores_from_codegen_assessment(feasibility=feasibility, complexity=complexity, validation_checks=validation_checks)
    benchmark = {
        "id": benchmark_id,
        "command": "" if truncated_source or source_missing else command,
        "hypothesis": hypothesis,
        "dimensions": [dimension],
        "analysis_method": analysis_method,
        "scores": scores,
        "files": []
        if truncated_source or source_missing
        else [
            {
                "path": source_path,
                "type": "cu",
                "content": source_code,
            }
        ],
    }
    return benchmark, amendment_list


def _benchmark_is_codegen_ready(benchmark: dict[str, Any]) -> bool:
    if not isinstance(benchmark, dict):
        return False
    command = str(benchmark.get("command", "")).strip()
    files = benchmark.get("files", [])
    return bool(command) and _has_cuda_source(_sanitize_files(files)) and _is_cuda_build_or_run_command(command)


def _render_codegen_repair_prompt(
    *,
    original_prompt: str,
    previous_memo: str,
    amendments: list[dict[str, Any]],
) -> str:
    fixes = []
    for item in amendments:
        if not isinstance(item, dict):
            continue
        change = str(item.get("change", "")).strip()
        rationale = str(item.get("rationale", "")).strip()
        if change and rationale:
            fixes.append(f"- {change} Reason: {rationale}")
        elif change:
            fixes.append(f"- {change}")
    if not fixes:
        fixes.append("- Return one complete five-section implementation memo with a fenced CUDA source block and a runnable nvcc command.")
    return "\n".join(
        [
            "Your previous implementation memo could not be converted into a runnable benchmark.",
            "Return one corrected implementation memo in the exact same five-section Markdown format.",
            "Do not summarize. Do not omit section 2 or section 3. Output the full corrected memo only.",
            "",
            "Fix these issues:",
            *fixes[:6],
            "",
            "Original task:",
            _trim_text(original_prompt, 4000),
            "",
            "Previous memo to repair:",
            _trim_text(previous_memo, 5000),
        ]
    ).strip()


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


def _extract_numbered_section(text: str, title: str) -> str:
    pattern = re.compile(
        rf"(?ims)^\s*(?:#+\s*)?\d+\.\s*{re.escape(title)}\s*$\n(?P<body>.*?)(?=^\s*(?:#+\s*)?\d+\.\s*[^\n]+\s*$|\Z)"
    )
    match = pattern.search(text or "")
    return match.group("body").strip() if match else ""


def _first_nonempty_line(text: str) -> str:
    for line in str(text or "").splitlines():
        clean = line.strip().lstrip("-*").strip()
        if clean:
            return clean
    return ""


def _extract_bullets_or_lines(text: str) -> list[str]:
    out: list[str] = []
    for line in str(text or "").splitlines():
        clean = line.strip()
        if not clean:
            continue
        clean = clean.lstrip("-*").strip()
        if clean and clean not in out:
            out.append(clean)
    return out[:6]


def _extract_labeled_value(text: str, label: str) -> str:
    match = re.search(rf"(?im)^\s*-?\s*{re.escape(label)}\s*:\s*(.+?)\s*$", str(text or ""))
    if not match:
        return ""
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'`', '"', "'"}:
        value = value[1:-1].strip()
    return value


def _extract_first_code_block(text: str, preferred_langs: tuple[str, ...] = ()) -> str:
    blocks = re.findall(r"(?is)```([a-zA-Z0-9_+-]*)\n(.*?)```", str(text or ""))
    if not blocks:
        return ""
    normalized = {lang.lower() for lang in preferred_langs}
    for lang, body in blocks:
        if not normalized or lang.lower() in normalized:
            return body.strip()
    return blocks[0][1].strip()


def _has_unclosed_code_fence(text: str) -> bool:
    return str(text or "").count("```") % 2 == 1


def _extract_remaining_lines(text: str, exclude_prefixes: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    excluded = tuple(prefix.lower() for prefix in exclude_prefixes)
    for line in str(text or "").splitlines():
        clean = line.strip()
        if not clean:
            continue
        if clean.lower().startswith(excluded):
            continue
        clean = clean.lstrip("-*").strip()
        if clean and clean not in out:
            out.append(clean)
    return out[:6]


def _slugify_dimension(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return text.strip("_")[:64]


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


def _scores_from_codegen_assessment(feasibility: str, complexity: str, validation_checks: list[str]) -> dict[str, Any]:
    coverage = 0.7
    observability = 0.78 if validation_checks else 0.7
    implementability = 0.9
    if complexity == "medium":
        implementability = 0.84
    elif complexity == "high":
        implementability = 0.68
    elif complexity == "excessive":
        implementability = 0.45
    if feasibility == "feasible_with_revision":
        implementability = min(implementability, 0.72)
        observability = max(observability, 0.7)
        coverage = min(coverage, 0.66)
    elif feasibility == "not_feasible":
        implementability = min(implementability, 0.35)
        observability = min(observability, 0.55)
        coverage = min(coverage, 0.45)
    rationale = f"Memo-derived assessment: feasibility={feasibility}, complexity={complexity}."
    return {
        "coverage_gain_score": round(coverage, 2),
        "implementability_score": round(implementability, 2),
        "observability_score": round(observability, 2),
        "rationale": rationale,
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
    legacy_planner = contract.get("planner_output", {}) if isinstance(contract.get("planner_output", {}), dict) else {}
    merged = {
        "version": str(contract.get("version", "1.0")),
        "negotiation_policy": _policy_from_contract(contract),
        "research_request_output": contract.get("research_request_output", {}),
        "proposal_output": contract.get("proposal_output", {}),
        "implementation_output": contract.get("implementation_output", {}),
        "analysis_output": contract.get("analysis_output", {}),
    }
    if not isinstance(merged["research_request_output"], dict):
        merged["research_request_output"] = {}
    if not isinstance(merged["proposal_output"], dict):
        merged["proposal_output"] = {}
    if legacy_planner:
        merged["proposal_output"] = {
            **legacy_planner,
            **merged["proposal_output"],
        }
    if not merged["research_request_output"]:
        merged["research_request_output"] = {"required_keys": ["reason", "research_request"]}
    if not merged["proposal_output"]:
        merged["proposal_output"] = {"required_keys": ["reason", "proposal"]}
    for key in ["implementation_output", "analysis_output"]:
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
