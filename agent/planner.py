"""Decides whether a question needs to be split into multiple retrieval
sub-tasks - either because it spans several selected documents, or because
it's a compound question with more than one distinct ask - and if so, what
each sub-question should target.

Deliberately gated behind a free heuristic (needs_decomposition) before the
one LLM call this makes: most real questions are simple single-document
lookups, and planning every question would burn Gemini quota - already a
hard constraint this project has hit directly (rag_pipeline.py's
MIN_RELEVANCE_SCORE calibration notes and eval/README.md both exist partly
because of how tightly that budget is used) - for no benefit on the common
case, since a single-sub-question plan behaves identically to not planning
at all.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from .state import SubQuestion

logger = logging.getLogger(__name__)

_COMPOUND_SIGNS = re.compile(
    r"\b(and|compared to|versus|vs\.?|relative to|as well as)\b", re.IGNORECASE
)
_WH_WORD = re.compile(r"\b(what|who|when|where|why|how|which)\b", re.IGNORECASE)


def needs_decomposition(question: str, selected_sources: list[str]) -> bool:
    """Zero-LLM-cost fast path. Only questions that look genuinely
    compound or multi-document get planned; everything else skips straight
    to a single-sub-question retrieval, at identical cost to not having a
    planner at all."""
    if len(selected_sources) > 1:
        return True
    if question.count("?") > 1:
        return True
    if len(_WH_WORD.findall(question)) > 1:
        return True
    return bool(_COMPOUND_SIGNS.search(question))


class _SubQuestionItem(BaseModel):
    text: str = Field(
        description=(
            "A single, standalone, specific question - answerable on its own. Do NOT "
            "mention any document filename in this text (put that in target_document "
            "instead) - naming the file here measurably hurts retrieval, since it's "
            "matched against document content, not filenames."
        )
    )
    target_document: str | None = Field(
        default=None,
        description=(
            "The exact filename (from the provided list) this sub-question should be "
            "searched against, if it clearly concerns just one document. Use null if it "
            "could apply to any/all of the selected documents. This is the ONLY place "
            "the filename should appear - never repeat it inside `text`."
        ),
    )


class _SubQuestionPlan(BaseModel):
    sub_questions: list[_SubQuestionItem] = Field(
        description="2 or more focused, standalone sub-questions that together fully cover the original question."
    )


def plan_subquestions(question: str, selected_sources: list[str], llm) -> list[SubQuestion]:
    """The one LLM call this module makes. Always returns at least one
    SubQuestion - if the model call fails or returns something degenerate,
    falls back to treating the original question as a single sub-question
    rather than leaving the caller with nothing to retrieve for."""
    prompt = f"""Break the following question into standalone sub-questions that, \
together, fully answer it. If the question is already a single, simple ask, \
return just one sub-question (the original question, unchanged).

IMPORTANT: each sub-question's text must read like a normal question about the \
subject matter only - e.g. "What is the sodium level?", never "What is the sodium \
level in report.pdf?". Filenames go ONLY in target_document, never inside the \
question text itself.

Available documents: {", ".join(selected_sources)}

Question: {question}"""

    try:
        structured_llm = llm.with_structured_output(_SubQuestionPlan)
        plan: _SubQuestionPlan = structured_llm.invoke(prompt)
    except Exception:
        logger.exception("Planner failed to decompose question; falling back to a single sub-question")
        return [SubQuestion(text=question, target_sources=None)]

    sub_questions: list[SubQuestion] = []
    for item in plan.sub_questions:
        target = None
        if item.target_document and item.target_document in selected_sources:
            target = [item.target_document]
        sub_questions.append(SubQuestion(text=item.text, target_sources=target))

    return sub_questions or [SubQuestion(text=question, target_sources=None)]
