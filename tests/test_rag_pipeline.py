"""Tests for the ingestion, retrieval and reranking logic in rag_pipeline.

These exercise the pipeline's internals directly (no Streamlit, no real
Gemini calls) so they run fast and don't touch the network or spend API
quota. A tiny PDF with real, extractable text is generated on the fly with
PyMuPDF so the tests don't depend on OCR/poppler being installed.
"""

import hashlib
import sys
from pathlib import Path

import fitz
import pytest
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag.rag_pipeline as rag_module


def _make_pdf(path: Path, line: str, repeats: int = 20) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for _ in range(repeats):
        page.insert_text((72, y), line, fontsize=11)
        y += 14
        if y > 750:
            page = doc.new_page()
            y = 72
    doc.save(str(path))
    doc.close()


@pytest.fixture
def sample_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(
        pdf_path,
        "Retrieval augmented generation combines a retriever with a language model.",
    )
    return pdf_path


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    """Point the pipeline's persistent index at a throwaway directory so
    tests never touch the real chroma_db/ used by the running app."""
    db_path = tmp_path / "chroma_db"
    monkeypatch.setattr(rag_module, "BASE_DB_PATH", str(db_path))
    return db_path


def test_ingest_is_idempotent(sample_pdf, isolated_index):
    vectorstore = rag_module.get_vectorstore()

    added_first, failed_first = rag_module.ingest_documents([sample_pdf], vectorstore)
    assert added_first > 0
    assert failed_first == []

    added_second, failed_second = rag_module.ingest_documents([sample_pdf], vectorstore)
    assert added_second == 0
    assert failed_second == []


def test_ingested_chunks_carry_source_metadata(sample_pdf, isolated_index):
    vectorstore = rag_module.get_vectorstore()
    rag_module.ingest_documents([sample_pdf], vectorstore)

    docs = rag_module._get_all_indexed_documents(vectorstore)
    assert docs
    assert all(d.metadata.get("source") == sample_pdf.name for d in docs)


def test_reingest_after_chunk_config_change_replaces_old_chunks(
    sample_pdf, isolated_index, monkeypatch
):
    vectorstore = rag_module.get_vectorstore()
    rag_module.ingest_documents([sample_pdf], vectorstore)

    monkeypatch.setattr(rag_module, "_CLASSIFIER_VERSION", "different-version")
    added, failed = rag_module.ingest_documents([sample_pdf], vectorstore)
    after = rag_module._get_all_indexed_documents(vectorstore)

    assert added > 0
    assert failed == []
    # Old chunks were deleted before re-ingesting, not left behind as duplicates.
    assert len(after) == added


def test_source_filter_scopes_retrieval_to_selected_documents(tmp_path, isolated_index):
    pdf_a = tmp_path / "doc_a.pdf"
    pdf_b = tmp_path / "doc_b.pdf"
    _make_pdf(pdf_a, "The quarterly revenue report shows strong growth in Q3.")
    _make_pdf(pdf_b, "The onboarding guide explains how new employees set up accounts.")

    vectorstore = rag_module.get_vectorstore()
    rag_module.ingest_documents([pdf_a, pdf_b], vectorstore)

    only_a = rag_module._get_all_indexed_documents(
        vectorstore, where={"source": {"$in": [pdf_a.name]}}
    )
    assert only_a
    assert all(d.metadata["source"] == pdf_a.name for d in only_a)

    both = rag_module._get_all_indexed_documents(
        vectorstore, where={"source": {"$in": [pdf_a.name, pdf_b.name]}}
    )
    assert {d.metadata["source"] for d in both} == {pdf_a.name, pdf_b.name}


def test_sessions_get_isolated_indexes(sample_pdf, isolated_index):
    vs_a = rag_module.get_vectorstore(session_id="user-a")
    vs_b = rag_module.get_vectorstore(session_id="user-b")

    rag_module.ingest_documents([sample_pdf], vs_a)

    assert rag_module._get_all_indexed_documents(vs_a)
    assert rag_module._get_all_indexed_documents(vs_b) == []


def test_no_session_id_uses_shared_default_index(sample_pdf, isolated_index):
    vectorstore = rag_module.get_vectorstore()
    assert Path(vectorstore._persist_directory) == isolated_index


def test_bm25_search_ranks_relevant_doc_first():
    # BM25's IDF term degenerates to exactly 0 for a 2-doc corpus where each
    # query term is unique to one doc (log(1.5) - log(1.5) == 0), so this
    # needs a 3rd distractor document to get real discriminating power.
    docs = [
        Document(page_content="cats are small domestic animals"),
        Document(page_content="the weather forecast predicts rain tomorrow"),
        Document(page_content="retrieval augmented generation improves LLM answers"),
    ]
    bm25 = BM25Okapi([d.page_content.split() for d in docs])
    results = rag_module.bm25_search(
        "retrieval augmented generation", docs, bm25, top_k=1
    )
    assert "retrieval" in results[0].page_content.lower()


def test_rerank_documents_orders_by_relevance():
    docs = [
        Document(page_content="The weather today is sunny with a light breeze."),
        Document(
            page_content="Retrieval augmented generation retrieves documents before answering."
        ),
    ]
    reranked = rag_module.rerank_documents(
        "What is retrieval augmented generation?", docs
    )
    top_doc, top_score = reranked[0]
    assert "retrieval augmented generation" in top_doc.page_content.lower()
    assert top_score > reranked[1][1]


def test_rerank_documents_relevance_score_separates_relevant_from_irrelevant():
    docs = [
        Document(page_content="Lithium-ion cells were cycled at 25 degrees Celsius."),
        Document(page_content="The default rate limit is 1000 requests per minute."),
    ]
    reranked = rag_module.rerank_documents("What is the API rate limit?", docs)
    relevant_score = next(
        score for doc, score in reranked if "rate limit" in doc.page_content
    )
    irrelevant_score = next(
        score for doc, score in reranked if "Lithium-ion" in doc.page_content
    )
    assert relevant_score > rag_module.MIN_RELEVANCE_SCORE
    assert irrelevant_score < rag_module.MIN_RELEVANCE_SCORE


def test_format_history_renders_recent_turns_and_caps_length():
    history = [("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4")]
    block = rag_module._format_history(history, max_turns=2)
    assert "q3" in block and "a3" in block
    assert "q4" in block and "a4" in block
    assert "q1" not in block and "q2" not in block


def test_format_history_empty_when_no_history():
    assert rag_module._format_history(None) == ""
    assert rag_module._format_history([]) == ""


def test_stream_answer_skips_llm_call_when_nothing_relevant_retrieved(
    sample_pdf, isolated_index, monkeypatch
):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called when nothing relevant was retrieved")

    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI.stream", _fail_if_called)

    chunks = list(
        rag_module.stream_answer(
            "What is the airspeed velocity of an unladen swallow?", [sample_pdf]
        )
    )
    assert chunks == [rag_module.NO_RELEVANT_CONTEXT_MESSAGE]


def test_stream_answer_yields_cached_answer_without_streaming(sample_pdf, isolated_index, monkeypatch):
    vectorstore = rag_module.get_vectorstore()
    rag_module.ingest_documents([sample_pdf], vectorstore)

    llm = rag_module.ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
    final_prompt = rag_module._prepare_prompt(
        "What does retrieval augmented generation combine?", [sample_pdf], llm
    )
    assert final_prompt is not None
    key = hashlib.md5(final_prompt.encode()).hexdigest()
    rag_module._llm_cache.set(key, "a pre-cached answer")

    def _fail_if_streamed(*args, **kwargs):
        raise AssertionError("Should not call llm.stream() on a cache hit")

    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI.stream", _fail_if_streamed)

    chunks = list(
        rag_module.stream_answer(
            "What does retrieval augmented generation combine?", [sample_pdf]
        )
    )
    assert chunks == ["a pre-cached answer"]


def test_answer_question_skips_llm_call_when_nothing_relevant_retrieved(
    sample_pdf, isolated_index, monkeypatch
):
    vectorstore = rag_module.get_vectorstore()
    rag_module.ingest_documents([sample_pdf], vectorstore)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called when nothing relevant was retrieved")

    monkeypatch.setattr(rag_module, "cached_llm_call", _fail_if_called)

    answer = rag_module.answer_question(
        "What is the airspeed velocity of an unladen swallow?", [sample_pdf]
    )
    assert answer == rag_module.NO_RELEVANT_CONTEXT_MESSAGE


def test_lru_cache_evicts_oldest_entry_beyond_maxsize():
    cache = rag_module._LRUCache(maxsize=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)

    assert cache.get("a", rag_module._MISSING) is rag_module._MISSING
    assert cache.get("b") == 2
    assert cache.get("c") == 3
