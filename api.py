"""FastAPI backend for the RAG pipeline.

Wraps the same rag_pipeline module the Streamlit UI (app.py) uses, so the
two never drift out of sync - Streamlit is one client of this logic, not
the only one. Run it standalone:

    uvicorn api:app --reload --port 8000

Then open http://localhost:8000/docs for interactive Swagger docs.
"""

import uuid
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import rag.rag_pipeline as rag_module
import storage

app = FastAPI(
    title="RAG Document QA API",
    description=(
        "HTTP API for a multimodal RAG pipeline over uploaded PDFs: hybrid "
        "retrieval (semantic + BM25), cross-encoder reranking, OCR fallback, "
        "and Gemini multimodal answering. Backs the Streamlit UI in app.py."
    ),
    version="1.0.0",
)


class SessionResponse(BaseModel):
    session_id: str


class DocumentInfo(BaseModel):
    name: str


class IngestResponse(BaseModel):
    session_id: str
    uploaded: List[str]
    new_chunks: int = Field(
        description="Chunks newly embedded by this call. 0 means every "
        "uploaded file was already indexed (unchanged content + chunk config)."
    )
    failed: List[str] = Field(
        default_factory=list,
        description="Uploaded files that produced no extractable text (native "
        "text and OCR both empty/failed) and so aren't searchable.",
    )


class HistoryTurn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    session_id: str
    question: str
    document_names: Optional[List[str]] = Field(
        default=None,
        description="Subset of this session's uploaded filenames to search. "
        "Omit to search every document uploaded in this session.",
    )
    history: Optional[List[HistoryTurn]] = Field(
        default=None,
        description="Prior (question, answer) turns from this session, oldest "
        "first. The API is otherwise stateless - the caller carries history.",
    )


class AskResponse(BaseModel):
    answer: str


def _resolve_pdf_paths(session_id: str, document_names: Optional[List[str]]):
    upload_dir = storage.session_upload_dir(session_id)
    available = {p.name: p for p in storage.list_uploaded_pdfs(upload_dir)}

    if not available:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded for this session yet. POST to "
            "/sessions/{session_id}/documents first.",
        )

    if document_names:
        missing = set(document_names) - available.keys()
        if missing:
            raise HTTPException(status_code=404, detail=f"Unknown document(s): {sorted(missing)}")
        return [available[name] for name in document_names]

    return list(available.values())


@app.post("/sessions", response_model=SessionResponse)
def create_session() -> SessionResponse:
    """Mint a new session id. Each session has its own isolated vector index
    and uploaded-document set (see rag_pipeline's per-session Chroma
    collections)."""
    return SessionResponse(session_id=uuid.uuid4().hex)


@app.post("/sessions/{session_id}/documents", response_model=IngestResponse)
async def upload_documents(session_id: str, files: List[UploadFile] = File(...)) -> IngestResponse:
    upload_dir = storage.session_upload_dir(session_id)
    saved_paths = []

    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename} is not a PDF")
        data = await f.read()
        saved_paths.append(storage.save_uploaded_pdf_bytes(f.filename, data, upload_dir))

    new_chunks, failed = rag_module.ingest_uploaded_pdfs(saved_paths, session_id=session_id)

    return IngestResponse(
        session_id=session_id,
        uploaded=[p.name for p in saved_paths],
        new_chunks=new_chunks,
        failed=failed,
    )


@app.get("/sessions/{session_id}/documents", response_model=List[DocumentInfo])
def list_documents(session_id: str) -> List[DocumentInfo]:
    upload_dir = storage.session_upload_dir(session_id)
    return [DocumentInfo(name=p.name) for p in storage.list_uploaded_pdfs(upload_dir)]


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    pdf_paths = _resolve_pdf_paths(payload.session_id, payload.document_names)
    history = [(t.question, t.answer) for t in payload.history] if payload.history else None

    answer = rag_module.answer_question(
        payload.question, pdf_paths, session_id=payload.session_id, history=history
    )
    return AskResponse(answer=answer)


@app.post("/ask/stream")
def ask_stream(payload: AskRequest) -> StreamingResponse:
    """Same as /ask, but streams the answer as plain text chunks (SSE-style
    incremental delivery) instead of waiting for the full response."""
    pdf_paths = _resolve_pdf_paths(payload.session_id, payload.document_names)
    history = [(t.question, t.answer) for t in payload.history] if payload.history else None

    def _generate():
        for piece in rag_module.stream_answer(
            payload.question, pdf_paths, session_id=payload.session_id, history=history
        ):
            yield piece

    return StreamingResponse(_generate(), media_type="text/plain")


@app.delete("/sessions/{session_id}/index")
def clear_index(session_id: str) -> dict:
    rag_module.reset_index(session_id=session_id)
    return {"status": "cleared", "session_id": session_id}
