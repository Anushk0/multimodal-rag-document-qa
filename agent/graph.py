"""LangGraph wiring for the agentic Q&A path: plan -> retrieve -> synthesize
-> critique, with a single bounded retry on critique failure.

This is the opt-in path behind app.py's "Agentic mode" toggle. The default
(toggle off) pipeline is rag_pipeline.answer_question/stream_answer,
completely untouched by anything in this module - agentic mode is additive,
never a replacement.

Deliberate scope limits, stated once here rather than re-justified in every
node:
- Synchronous only (StateGraph.invoke, not .ainvoke) - Streamlit's own
  execution model is a synchronous script rerun, not natively async.
- No checkpointing/MemorySaver - nothing needs to persist across turns,
  since each Streamlit rerun is already a complete fresh request/response.
- No prebuilt tool-calling agent executor (create_react_agent) - these are
  deterministic, hand-controlled nodes, not an open-ended tool loop.
- No parallel (Send-based) fan-out for sub-questions - see
  agent/retriever.py's docstring: rag_pipeline's process-wide
  _llm_cache/_vision_cache caches aren't built for concurrent access, and
  parallelizing retrieval wouldn't reduce the actual cost driver anyway
  (the sequential LLM calls: plan, synthesize, critique).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from rag.rag_pipeline import (
    MIN_RELEVANCE_SCORE,
    NO_RELEVANT_CONTEXT_MESSAGE,
    _emit_step,
    _format_source,
    get_image_descriptions,
    get_vectorstore,
    ingest_documents,
)

from .critic import critique_answer
from .planner import needs_decomposition, plan_subquestions
from .retriever import retrieve_for_all
from .state import AgentState, SubQuestion
from .synthesizer import synthesize_answer

# Same reasoning as rag_pipeline.answer_question/stream_answer's identical
# choice - see the comment there. Kept in sync deliberately, not imported,
# since a future change to one temperature shouldn't silently retune the
# other without a conscious decision.
_TEMPERATURE = 0.1
_MODEL = "gemini-3-flash-preview"


def _node_plan(state: AgentState) -> dict:
    question = state["question"]
    selected_sources = state["selected_sources"]
    trace, on_step = state.get("trace"), state.get("on_step")

    if needs_decomposition(question, selected_sources):
        _emit_step(trace, on_step, "Planning", "question looks compound or multi-document - decomposing")
        sub_questions = plan_subquestions(question, selected_sources, state["llm"])
        _emit_step(
            trace, on_step, "Plan ready",
            f"{len(sub_questions)} sub-question(s): " + "; ".join(sq["text"] for sq in sub_questions),
        )
    else:
        sub_questions = [SubQuestion(text=question, target_sources=None)]

    return {"sub_questions": sub_questions}


def _node_retrieve(state: AgentState) -> dict:
    trace, on_step = state.get("trace"), state.get("on_step")

    sub_results = retrieve_for_all(
        state["sub_questions"], state["vectorstore"], state["selected_sources"], state.get("history"),
        trace=trace, on_step=on_step,
    )
    image_descriptions = get_image_descriptions(
        state["question"], state["pdf_paths"], state["llm"], trace=trace, on_step=on_step
    )
    grounded = bool(image_descriptions) or any(result.grounded for _sq, result in sub_results)

    return {"sub_results": sub_results, "image_descriptions": image_descriptions, "grounded": grounded}


def _node_synthesize(state: AgentState) -> dict:
    trace, on_step = state.get("trace"), state.get("on_step")
    strict = state.get("retry_count", 0) > 0

    _emit_step(
        trace, on_step, "Generating answer",
        "regenerating with stricter grounding instructions" if strict else "combining retrieved evidence",
    )
    answer = synthesize_answer(
        state["question"], state["sub_results"], state["image_descriptions"], state["llm"],
        history=state.get("history"), strict=strict,
    )
    return {"answer": answer}


def _node_critique(state: AgentState) -> dict:
    trace, on_step = state.get("trace"), state.get("on_step")

    _emit_step(trace, on_step, "Self-checking answer", "verifying the answer only uses the given context")
    result = critique_answer(state["question"], state["answer"], state["sub_results"], state["llm"])
    _emit_step(
        trace, on_step, "Self-check result",
        ("passed" if result.passed else "failed") + f" - {result.reason}",
    )
    return {"critique_passed": result.passed, "critique_reason": result.reason}


def _node_bump_retry(state: AgentState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def _route_after_retrieve(state: AgentState) -> str:
    return "synthesize" if state.get("grounded") else "ungrounded"


def _route_after_critique(state: AgentState) -> str:
    if state.get("critique_passed"):
        return "done"
    if state.get("retry_count", 0) == 0:
        return "retry"
    return "give_up"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", _node_plan)
    graph.add_node("retrieve", _node_retrieve)
    graph.add_node("synthesize", _node_synthesize)
    graph.add_node("critique", _node_critique)
    graph.add_node("bump_retry", _node_bump_retry)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_conditional_edges("retrieve", _route_after_retrieve, {"synthesize": "synthesize", "ungrounded": END})
    graph.add_edge("synthesize", "critique")
    graph.add_conditional_edges(
        "critique", _route_after_critique, {"done": END, "retry": "bump_retry", "give_up": END}
    )
    graph.add_edge("bump_retry", "synthesize")

    return graph.compile()


_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def agentic_answer_question(
    question: str,
    pdf_paths,
    session_id: str | None = None,
    history=None,
    trace: dict | None = None,
    on_step: Callable[[dict], None] | None = None,
) -> str:
    if not pdf_paths:
        return "No documents selected. Upload and select at least one PDF first."

    t_start = time.perf_counter()
    vectorstore = get_vectorstore(session_id)
    ingest_documents(pdf_paths, vectorstore)
    selected_sources = sorted({Path(p).name for p in pdf_paths})
    llm = ChatGoogleGenerativeAI(model=_MODEL, temperature=_TEMPERATURE)

    initial_state: AgentState = {
        "question": question,
        "selected_sources": selected_sources,
        "pdf_paths": pdf_paths,
        "session_id": session_id,
        "history": history,
        "vectorstore": vectorstore,
        "llm": llm,
        "trace": trace,
        "on_step": on_step,
        "retry_count": 0,
    }

    final_state = _get_graph().invoke(initial_state)

    if trace is not None:
        _populate_summary_trace_fields(trace, final_state, t_start)

    if not final_state.get("grounded"):
        return NO_RELEVANT_CONTEXT_MESSAGE
    if final_state.get("critique_passed") is False:
        return NO_RELEVANT_CONTEXT_MESSAGE
    return final_state.get("answer") or NO_RELEVANT_CONTEXT_MESSAGE


def _populate_summary_trace_fields(trace: dict, final_state: AgentState, t_start: float) -> None:
    """Fill in the same top-level trace keys rag_pipeline._prepare_prompt
    sets for the simple pipeline (top_score, grounded, sources_used, ...),
    aggregated across every sub-question's retrieval - so app.py's existing
    _confidence_badge/_render_trace work unchanged for an agentic answer
    too, instead of needing a second, agentic-specific rendering path."""
    sub_results = final_state.get("sub_results") or []
    all_scores = [r.top_score for _sq, r in sub_results if r.top_score is not None]

    merged_sources: dict[str, float] = {}
    for _sq, result in sub_results:
        for doc, score in result.reranked_docs:
            key = _format_source(doc)
            if key not in merged_sources or score > merged_sources[key]:
                merged_sources[key] = float(score)
    top_sources = sorted(merged_sources.items(), key=lambda kv: kv[1], reverse=True)[:6]

    trace.update(
        {
            "agentic": True,
            "sub_question_count": len(final_state.get("sub_questions") or []),
            "critique_passed": final_state.get("critique_passed"),
            "documents_searched": final_state.get("selected_sources"),
            "total_chunks_indexed": max((r.total_chunks_indexed for _sq, r in sub_results), default=0),
            "semantic_hits": sum(r.semantic_hit_count for _sq, r in sub_results),
            "bm25_hits": sum(r.bm25_hit_count for _sq, r in sub_results),
            "candidates_after_merge": sum(len(r.reranked_docs) for _sq, r in sub_results),
            "vision_triggered": bool(final_state.get("image_descriptions")),
            "top_score": round(max(all_scores), 2) if all_scores else None,
            "grounding_threshold": MIN_RELEVANCE_SCORE,
            "grounded": bool(final_state.get("grounded")),
            "sources_used": [{"source": source, "score": round(score, 2)} for source, score in top_sources],
            "total_ms": round((time.perf_counter() - t_start) * 1000, 1),
        }
    )


def agentic_stream_answer(
    question: str,
    pdf_paths,
    session_id: str | None = None,
    history=None,
    trace: dict | None = None,
    on_step: Callable[[dict], None] | None = None,
):
    """Non-streaming internally, unlike rag_pipeline.stream_answer:
    self-critique needs the complete draft answer before deciding whether
    the user should see it at all, so there's no meaningful way to
    token-stream a draft and then retract it. The approved text is yielded
    in one piece once the graph finishes - matching how stream_answer's own
    cache-hit branch already yields a complete answer in one piece too,
    just for a different reason (nothing to stream) rather than this one
    (can't safely stream something that might get rejected)."""
    yield agentic_answer_question(
        question, pdf_paths, session_id=session_id, history=history, trace=trace, on_step=on_step
    )
