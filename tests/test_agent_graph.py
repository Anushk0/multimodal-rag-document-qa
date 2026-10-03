"""Tests for agent/graph.py's actual routing/control-flow: does the fast
path skip planning, does one failed critique trigger exactly one bounded
regeneration, does an ungrounded retrieval short-circuit before ever
calling synthesize/critique.

Retrieval uses a real local vectorstore (zero-cost - embeddings/Chroma/
reranking are all local, per rag_pipeline.retrieve_and_rerank) with real
ingested test documents, matching test_agent_retriever.py. The plan/
synthesize/critique steps use a stub LLM injected directly into the initial
state, so these tests call build_graph().invoke(state) directly rather than
through agentic_answer_question (which always constructs a real
ChatGoogleGenerativeAI internally, mirroring rag_pipeline.answer_question's
existing pattern) - this is the graph's own control flow under test, not
the thin wrapper around it.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag.rag_pipeline as rag_module
from agent.critic import _CritiqueVerdict
from agent.graph import build_graph


def _make_pdf(path: Path, line: str, repeats: int = 20) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for _ in range(repeats):
        page.insert_text((72, y), line, fontsize=11)
        y += 14
    doc.save(str(path))
    doc.close()


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    db_path = tmp_path / "chroma_db"
    monkeypatch.setattr(rag_module, "BASE_DB_PATH", str(db_path))
    return db_path


@pytest.fixture
def indexed_vectorstore(tmp_path, isolated_index):
    pdf_path = tmp_path / "report.pdf"
    _make_pdf(pdf_path, "The quarterly revenue report shows strong growth in Q3 2024.")
    vectorstore = rag_module.get_vectorstore()
    rag_module.ingest_documents([pdf_path], vectorstore)
    return vectorstore


class _StubStructuredLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, prompt):
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(response, Exception):
            raise response
        return response


class _StubLLM:
    """Handles both .invoke() (used by synthesize) and
    .with_structured_output() (used by critique/planner). Each call to
    .invoke() pops the next scripted answer text; critique_responses are
    handed to a fresh _StubStructuredLLM per with_structured_output() call
    so critique can be scripted to return different verdicts across a
    retry."""

    def __init__(self, answers=("stub answer",), critique_verdicts=(_CritiqueVerdict(passed=True, reason="ok"),)):
        self._answers = list(answers)
        self._critique_verdicts = list(critique_verdicts)

    def invoke(self, prompt):
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        return SimpleNamespace(content=answer)

    def with_structured_output(self, schema):
        verdict = self._critique_verdicts.pop(0) if len(self._critique_verdicts) > 1 else self._critique_verdicts[0]
        return _StubStructuredLLM([verdict])


def _base_state(vectorstore, question="What does the revenue report show?"):
    return {
        "question": question,
        "selected_sources": ["report.pdf"],
        "pdf_paths": [Path("report.pdf")],
        "session_id": None,
        "history": None,
        "vectorstore": vectorstore,
        "trace": {},
        "on_step": None,
        "retry_count": 0,
    }


def test_simple_question_skips_planning_and_passes_through(indexed_vectorstore):
    state = _base_state(indexed_vectorstore)
    state["llm"] = _StubLLM(critique_verdicts=[_CritiqueVerdict(passed=True, reason="ok")])

    final_state = build_graph().invoke(state)

    assert len(final_state["sub_questions"]) == 1  # fast path: no decomposition
    assert final_state["grounded"] is True
    assert final_state["critique_passed"] is True
    assert final_state["answer"] == "stub answer"
    assert final_state["retry_count"] == 0  # no retry needed


def test_failed_critique_triggers_exactly_one_bounded_retry(indexed_vectorstore):
    state = _base_state(indexed_vectorstore)
    state["llm"] = _StubLLM(
        answers=["first draft", "regenerated draft"],
        critique_verdicts=[
            _CritiqueVerdict(passed=False, reason="invented a detail"),
            _CritiqueVerdict(passed=True, reason="fixed"),
        ],
    )

    final_state = build_graph().invoke(state)

    assert final_state["retry_count"] == 1
    assert final_state["critique_passed"] is True
    assert final_state["answer"] == "regenerated draft"


def test_repeatedly_failing_critique_gives_up_after_one_retry_not_forever(indexed_vectorstore):
    state = _base_state(indexed_vectorstore)
    state["llm"] = _StubLLM(
        answers=["first draft", "still bad draft"],
        critique_verdicts=[
            _CritiqueVerdict(passed=False, reason="bad"),
            _CritiqueVerdict(passed=False, reason="still bad"),
        ],
    )

    final_state = build_graph().invoke(state)

    # Bounded: exactly one retry attempted, then the graph stops rather
    # than looping - critique_passed stays False so the caller
    # (agentic_answer_question) falls back to NO_RELEVANT_CONTEXT_MESSAGE.
    assert final_state["retry_count"] == 1
    assert final_state["critique_passed"] is False


def test_ungrounded_retrieval_short_circuits_before_synthesize_or_critique(indexed_vectorstore):
    state = _base_state(indexed_vectorstore, question="What is the capital of France?")

    class _FailIfCalled:
        def invoke(self, prompt):
            raise AssertionError("synthesize should not be called when nothing was retrieved")

        def with_structured_output(self, schema):
            raise AssertionError("critique should not be called when nothing was retrieved")

    state["llm"] = _FailIfCalled()

    final_state = build_graph().invoke(state)

    assert final_state["grounded"] is False
    assert "answer" not in final_state or final_state.get("answer") is None
