"""Tests for the FastAPI backend (api.py).

Uses FastAPI's TestClient (no real server/socket needed) and generates a
tiny in-memory PDF with PyMuPDF, so these run fast and offline. The /ask
flow is exercised through the grounding gate (an intentionally irrelevant
question), which never reaches the Gemini API - same trick used in
test_rag_pipeline.py - so no network access or API quota is needed here.
"""

import sys
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api  # noqa: E402
import rag.rag_pipeline as rag_module  # noqa: E402
import storage  # noqa: E402

client = TestClient(api.app)


def _make_pdf_bytes(line: str, repeats: int = 20) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for _ in range(repeats):
        page.insert_text((72, y), line, fontsize=11)
        y += 14
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_module, "BASE_DB_PATH", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(storage, "UPLOADED_ROOT", tmp_path / "uploaded_docs")


def test_full_session_flow(isolated_storage):
    session_id = client.post("/sessions").json()["session_id"]

    pdf_bytes = _make_pdf_bytes(
        "Retrieval augmented generation combines a retriever with a language model."
    )
    upload_resp = client.post(
        f"/sessions/{session_id}/documents",
        files={"files": ("sample.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload_resp.status_code == 200
    body = upload_resp.json()
    assert body["uploaded"] == ["sample.pdf"]
    assert body["new_chunks"] > 0

    docs = client.get(f"/sessions/{session_id}/documents").json()
    assert [d["name"] for d in docs] == ["sample.pdf"]

    # An irrelevant question hits the grounding gate and returns without a
    # real Gemini call - see MIN_RELEVANCE_SCORE in rag_pipeline.py.
    ask_resp = client.post(
        "/ask",
        json={
            "session_id": session_id,
            "question": "What is the airspeed velocity of an unladen swallow?",
        },
    )
    assert ask_resp.status_code == 200
    assert ask_resp.json()["answer"] == rag_module.NO_RELEVANT_CONTEXT_MESSAGE

    clear_resp = client.delete(f"/sessions/{session_id}/index")
    assert clear_resp.status_code == 200
    assert clear_resp.json()["status"] == "cleared"


def test_sessions_are_isolated(isolated_storage):
    session_a = client.post("/sessions").json()["session_id"]
    session_b = client.post("/sessions").json()["session_id"]
    assert session_a != session_b

    pdf_bytes = _make_pdf_bytes("Document belonging only to session A.")
    client.post(
        f"/sessions/{session_a}/documents",
        files={"files": ("a.pdf", pdf_bytes, "application/pdf")},
    )

    assert len(client.get(f"/sessions/{session_a}/documents").json()) == 1
    assert client.get(f"/sessions/{session_b}/documents").json() == []


def test_ask_with_history_still_hits_grounding_gate(isolated_storage):
    session_id = client.post("/sessions").json()["session_id"]
    pdf_bytes = _make_pdf_bytes(
        "Retrieval augmented generation combines a retriever with a language model."
    )
    client.post(
        f"/sessions/{session_id}/documents",
        files={"files": ("sample.pdf", pdf_bytes, "application/pdf")},
    )

    resp = client.post(
        "/ask",
        json={
            "session_id": session_id,
            "question": "What is the airspeed velocity of an unladen swallow?",
            "history": [{"question": "earlier question", "answer": "earlier answer"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["answer"] == rag_module.NO_RELEVANT_CONTEXT_MESSAGE


def test_ask_stream_returns_grounding_gate_message(isolated_storage):
    session_id = client.post("/sessions").json()["session_id"]
    pdf_bytes = _make_pdf_bytes(
        "Retrieval augmented generation combines a retriever with a language model."
    )
    client.post(
        f"/sessions/{session_id}/documents",
        files={"files": ("sample.pdf", pdf_bytes, "application/pdf")},
    )

    resp = client.post(
        "/ask/stream",
        json={
            "session_id": session_id,
            "question": "What is the airspeed velocity of an unladen swallow?",
        },
    )
    assert resp.status_code == 200
    assert resp.text == rag_module.NO_RELEVANT_CONTEXT_MESSAGE


def test_ask_without_documents_returns_400(isolated_storage):
    session_id = client.post("/sessions").json()["session_id"]
    resp = client.post("/ask", json={"session_id": session_id, "question": "anything"})
    assert resp.status_code == 400


def test_ask_with_unknown_document_name_returns_404(isolated_storage):
    session_id = client.post("/sessions").json()["session_id"]
    pdf_bytes = _make_pdf_bytes("Some content.")
    client.post(
        f"/sessions/{session_id}/documents",
        files={"files": ("real.pdf", pdf_bytes, "application/pdf")},
    )
    resp = client.post(
        "/ask",
        json={
            "session_id": session_id,
            "question": "anything",
            "document_names": ["does_not_exist.pdf"],
        },
    )
    assert resp.status_code == 404


def test_upload_rejects_non_pdf(isolated_storage):
    session_id = client.post("/sessions").json()["session_id"]
    resp = client.post(
        f"/sessions/{session_id}/documents",
        files={"files": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400
