"""Tests for agent/planner.py - zero LLM cost for the fast-path heuristic,
and a stub LLM (no network) for the decomposition call itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.planner import _SubQuestionItem, _SubQuestionPlan, needs_decomposition, plan_subquestions


def test_single_document_simple_question_skips_decomposition():
    assert needs_decomposition("What is the sodium level?", ["report.pdf"]) is False


def test_multiple_selected_documents_needs_decomposition():
    assert needs_decomposition("What is the sodium level?", ["a.pdf", "b.pdf"]) is True


def test_compound_question_needs_decomposition():
    assert needs_decomposition("What is the sodium level and is it flagged?", ["report.pdf"]) is True


def test_multiple_question_marks_needs_decomposition():
    assert needs_decomposition("What is sodium? What is potassium?", ["report.pdf"]) is True


class _StubStructuredLLM:
    def __init__(self, response):
        self._response = response

    def invoke(self, prompt):
        return self._response


class _StubLLM:
    def __init__(self, structured_response):
        self._structured_response = structured_response

    def with_structured_output(self, schema):
        return _StubStructuredLLM(self._structured_response)


def test_plan_subquestions_maps_target_document_when_valid():
    stub_plan = _SubQuestionPlan(
        sub_questions=[
            _SubQuestionItem(text="What is the sodium level in report_a.pdf?", target_document="report_a.pdf"),
            _SubQuestionItem(text="What is the sodium level in report_b.pdf?", target_document="report_b.pdf"),
        ]
    )
    llm = _StubLLM(stub_plan)

    result = plan_subquestions("Compare sodium between report_a and report_b", ["report_a.pdf", "report_b.pdf"], llm)

    assert len(result) == 2
    assert result[0]["target_sources"] == ["report_a.pdf"]
    assert result[1]["target_sources"] == ["report_b.pdf"]


def test_plan_subquestions_ignores_hallucinated_target_document():
    # The model named a file that isn't actually in selected_sources -
    # should fall back to searching everything, not silently scope to a
    # document that was never selected.
    stub_plan = _SubQuestionPlan(
        sub_questions=[_SubQuestionItem(text="What is the sodium level?", target_document="made_up.pdf")]
    )
    llm = _StubLLM(stub_plan)

    result = plan_subquestions("What is the sodium level?", ["report.pdf"], llm)

    assert result[0]["target_sources"] is None


def test_plan_subquestions_falls_back_to_original_question_on_llm_error():
    class _FailingLLM:
        def with_structured_output(self, schema):
            raise RuntimeError("simulated API failure")

    result = plan_subquestions("What is the sodium level?", ["report.pdf"], _FailingLLM())

    assert result == [{"text": "What is the sodium level?", "target_sources": None}]
