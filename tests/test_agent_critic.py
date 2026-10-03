"""Tests for agent/critic.py using a stub LLM (no network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from agent.critic import _CritiqueVerdict, critique_answer
from agent.state import SubQuestion
from rag.rag_pipeline import RetrievalResult


def _result_with(doc_text: str) -> RetrievalResult:
    doc = Document(page_content=doc_text, metadata={"source": "a.pdf"})
    return RetrievalResult(reranked_docs=[(doc, 1.0)], total_chunks_indexed=1, semantic_hit_count=1, bm25_hit_count=0)


class _StubStructuredLLM:
    def __init__(self, response):
        self._response = response

    def invoke(self, prompt):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _StubLLM:
    def __init__(self, response):
        self._response = response

    def with_structured_output(self, schema):
        return _StubStructuredLLM(self._response)


def test_critique_passes_through_verdict():
    sub_results = [(SubQuestion(text="q", target_sources=None), _result_with("Sodium is 140 mmol/L."))]
    llm = _StubLLM(_CritiqueVerdict(passed=True, reason="matches context"))

    result = critique_answer("q", "The sodium level is 140 mmol/L.", sub_results, llm)

    assert result.passed is True
    assert result.reason == "matches context"


def test_critique_reports_failure_verdict():
    sub_results = [(SubQuestion(text="q", target_sources=None), _result_with("Sodium is 140 mmol/L."))]
    llm = _StubLLM(_CritiqueVerdict(passed=False, reason="answer invents a detail not in context"))

    result = critique_answer("q", "The sodium level is 140 mmol/L and trending upward.", sub_results, llm)

    assert result.passed is False


def test_critique_passes_through_on_llm_error_rather_than_blocking():
    sub_results = [(SubQuestion(text="q", target_sources=None), _result_with("Sodium is 140 mmol/L."))]
    llm = _StubLLM(RuntimeError("simulated API failure"))

    result = critique_answer("q", "The sodium level is 140 mmol/L.", sub_results, llm)

    assert result.passed is True
    assert "unavailable" in result.reason
