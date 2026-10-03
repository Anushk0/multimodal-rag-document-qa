"""Tests for agent/retriever.py against a real (local, no-LLM) vector
store - retrieval itself is zero-cost, so these use real ingestion/
embeddings/reranking rather than stubs, matching how test_rag_pipeline.py
already tests retrieval directly.
"""

import sys
from pathlib import Path

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag.rag_pipeline as rag_module
from agent.retriever import retrieve_for_all, retrieve_for_subquestion
from agent.state import SubQuestion


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
def two_docs(tmp_path, isolated_index):
    pdf_a = tmp_path / "revenue.pdf"
    pdf_b = tmp_path / "onboarding.pdf"
    _make_pdf(pdf_a, "The quarterly revenue report shows strong growth in Q3 2024.")
    _make_pdf(pdf_b, "The onboarding guide explains how new employees set up their accounts.")

    vectorstore = rag_module.get_vectorstore()
    rag_module.ingest_documents([pdf_a, pdf_b], vectorstore)
    return vectorstore, ["revenue.pdf", "onboarding.pdf"]


def test_retrieve_for_subquestion_scopes_to_target_source(two_docs):
    vectorstore, all_sources = two_docs
    sub_q = SubQuestion(text="What does the revenue report show?", target_sources=["revenue.pdf"])

    result = retrieve_for_subquestion(sub_q, vectorstore, all_sources, history=None)

    assert result.grounded
    assert all(doc.metadata.get("source") == "revenue.pdf" for doc, _score in result.reranked_docs)


def test_retrieve_for_subquestion_broadens_when_scoped_search_is_weak(two_docs):
    vectorstore, all_sources = two_docs
    # Deliberately mis-scoped: the planner would have gotten this wrong -
    # the onboarding question has nothing to do with revenue.pdf.
    sub_q = SubQuestion(text="How do new employees set up their accounts?", target_sources=["revenue.pdf"])

    result = retrieve_for_subquestion(sub_q, vectorstore, all_sources, history=None)

    # After broadening, the correct document should actually be found.
    assert result.grounded
    assert any(doc.metadata.get("source") == "onboarding.pdf" for doc, _score in result.reranked_docs)


def test_retrieve_for_all_returns_one_result_per_subquestion(two_docs):
    vectorstore, all_sources = two_docs
    sub_questions = [
        SubQuestion(text="What does the revenue report show?", target_sources=["revenue.pdf"]),
        SubQuestion(text="How do new employees set up accounts?", target_sources=["onboarding.pdf"]),
    ]

    results = retrieve_for_all(sub_questions, vectorstore, all_sources, history=None)

    assert len(results) == 2
    assert results[0][0] is sub_questions[0]
    assert results[1][0] is sub_questions[1]
