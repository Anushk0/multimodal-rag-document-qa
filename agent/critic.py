"""Self-critique - the "reflect" step of this agent. Checks whether a
generated answer actually stays within the retrieved context, independent
of whether retrieval itself scored well.

This targets a different failure mode than rag_pipeline's grounding gate:
the gate checks whether the right evidence was FOUND (a retrieval-side
question); this checks whether the generated answer actually STUCK to that
evidence (a generation-side question) - good context can still produce an
answer that embellishes or invents a detail the context never stated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field

from rag.rag_pipeline import TOP_K_FINAL, RetrievalResult, _format_source

from .state import SubQuestion

logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    passed: bool
    reason: str


class _CritiqueVerdict(BaseModel):
    passed: bool = Field(description="True only if every factual claim in the answer is directly supported by the context.")
    reason: str = Field(description="One brief sentence explaining the verdict.")


def _build_context_text(sub_results: list[tuple[SubQuestion, RetrievalResult]], top_k: int = TOP_K_FINAL) -> str:
    bits = []
    for _sub_q, result in sub_results:
        for doc, _score in result.reranked_docs[:top_k]:
            bits.append(f"{doc.page_content}\n(Source: {_format_source(doc)})")
    return "\n\n---\n\n".join(bits)


def critique_answer(
    question: str,
    answer: str,
    sub_results: list[tuple[SubQuestion, RetrievalResult]],
    llm,
) -> CritiqueResult:
    """The one LLM call this module makes. If the critic itself errors out,
    passes the answer through rather than blocking a perfectly good answer
    on an infrastructure failure - the simple pipeline makes the same
    choice by surfacing LLM errors as text rather than silently discarding
    a real answer, and a broken critic shouldn't make agentic mode strictly
    worse than not having one."""
    context = _build_context_text(sub_results)

    prompt = f"""You are a strict fact-checker. Given a question, the context it was \
answered from, and the answer, decide whether EVERY factual claim in the answer is \
directly supported by the context. Minor rephrasing or summarizing is fine; adding \
any detail not present in the context is not.

Question:
{question}

Context:
{context}

Answer to check:
{answer}"""

    try:
        structured_llm = llm.with_structured_output(_CritiqueVerdict)
        verdict: _CritiqueVerdict = structured_llm.invoke(prompt)
        return CritiqueResult(passed=verdict.passed, reason=verdict.reason)
    except Exception:
        logger.exception("Critique call failed; passing the answer through unverified")
        return CritiqueResult(passed=True, reason="critique unavailable, passed through")
