"""Shared state/types for the agentic (LangGraph) Q&A path.

Kept separate from rag_pipeline.py's own types (RetrievalResult etc.)
because this is agent-graph-specific shape, not something the simple
(non-agentic) pipeline or api.py need to know about.
"""

from __future__ import annotations

from typing import Any, Callable, TypedDict


class SubQuestion(TypedDict):
    text: str
    # None means "search across all of the question's selected_sources";
    # a non-None list scopes retrieval to just those document(s), as
    # decided by the planner (e.g. one sub-question per compared document).
    target_sources: list[str] | None


class AgentState(TypedDict, total=False):
    # Inputs - set once when the graph is invoked, read-only after.
    question: str
    selected_sources: list[str]
    pdf_paths: list[Any]
    session_id: str | None
    history: list[tuple[str, str]] | None
    vectorstore: Any
    llm: Any
    trace: dict | None
    on_step: Callable[[dict], None] | None

    # Produced by the graph's nodes as it runs.
    sub_questions: list[SubQuestion]
    sub_results: list[tuple[SubQuestion, Any]]  # Any = rag_pipeline.RetrievalResult
    image_descriptions: list[str]
    grounded: bool
    answer: str | None
    critique_passed: bool
    critique_reason: str
    retry_count: int
