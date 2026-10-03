"""Runs retrieval for each planned sub-question, scoped to its target
document(s) if the planner specified one, falling back to the question's
full selected-sources set otherwise.

A sub-question whose scoped retrieval comes back weak gets ONE broadened
retry across the full selected set, in case the planner mis-assigned which
document actually has the answer. That retry is a second local vectorstore
query, not an LLM call - free with respect to the quota that actually
constrains this app (see rag_pipeline.retrieve_and_rerank; all retrieval
compute is local).

Sub-questions are retrieved sequentially, not in parallel, on purpose: see
agent/graph.py's module docstring for why (rag_pipeline's process-wide
_llm_cache/_vision_cache caches aren't built for concurrent access, and
there's no latency win anyway since retrieval is local/free while the only
real cost - the sequential LLM calls elsewhere in the graph - wouldn't be
reduced by parallelizing this).
"""

from __future__ import annotations

from typing import Callable

from rag.rag_pipeline import (
    RetrievalResult,
    _build_retrieval_query,
    _emit_step,
    retrieve_and_rerank,
)

from .state import SubQuestion


def retrieve_for_subquestion(
    sub_q: SubQuestion,
    vectorstore,
    all_selected_sources: list[str],
    history,
    trace: dict | None = None,
    on_step: Callable[[dict], None] | None = None,
) -> RetrievalResult:
    retrieval_query = _build_retrieval_query(sub_q["text"], history)
    target_sources = sub_q.get("target_sources") or all_selected_sources

    result = retrieve_and_rerank(
        sub_q["text"], retrieval_query, vectorstore, target_sources, trace=trace, on_step=on_step
    )

    scoped_narrower = sub_q.get("target_sources") and set(target_sources) != set(all_selected_sources)
    if not result.grounded and scoped_narrower:
        _emit_step(
            trace, on_step, "Weak match, broadening search",
            f"'{sub_q['text']}' scored low against {target_sources} - retrying across all selected documents",
        )
        result = retrieve_and_rerank(
            sub_q["text"], retrieval_query, vectorstore, all_selected_sources, trace=trace, on_step=on_step
        )

    return result


def retrieve_for_all(
    sub_questions: list[SubQuestion],
    vectorstore,
    all_selected_sources: list[str],
    history,
    trace: dict | None = None,
    on_step: Callable[[dict], None] | None = None,
) -> list[tuple[SubQuestion, RetrievalResult]]:
    return [
        (
            sub_q,
            retrieve_for_subquestion(
                sub_q, vectorstore, all_selected_sources, history, trace=trace, on_step=on_step
            ),
        )
        for sub_q in sub_questions
    ]
