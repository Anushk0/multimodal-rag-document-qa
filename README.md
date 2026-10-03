# Multimodal RAG Document QA System

A Retrieval-Augmented Generation (RAG) system that answers questions over
multiple uploaded PDFs — native text extraction, OCR fallback for scanned
pages, figure/diagram understanding via Gemini's multimodal vision, hybrid
retrieval, and cross-encoder reranking — served through both a Streamlit
chat UI and a standalone FastAPI backend.

## How it works

1. **Ingestion** (`rag/rag_pipeline.py: ingest_documents`) — each uploaded
   PDF is loaded page-by-page (`PyPDFLoader`); if a PDF has little to no
   extractable native text (e.g. a scan), it's rendered to images and run
   through Tesseract OCR (with grayscale/contrast/sharpen preprocessing,
   `image_processing.py`). Pages are chunked (`RecursiveCharacterTextSplitter`)
   and embedded (`sentence-transformers/all-MiniLM-L6-v2`) into a persistent
   Chroma vector store. Each file's content hash *and* chunking config are
   tracked in a per-index registry, so re-running ingestion on unchanged
   files/config is a no-op — OCR and embedding are never redone unnecessarily
   (and a config change cleanly re-embeds, replacing old chunks rather than
   duplicating them).
2. **Retrieval** — a query combines semantic search (MMR over the vector
   store), BM25 keyword search, and a cross-encoder reranker
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to select the most relevant
   chunks, scoped to whichever documents the caller selects. See
   [eval/README.md](eval/README.md) for a benchmark of this hybrid pipeline
   against naive semantic-only search.
3. **Grounding gate** — the reranker's own relevance score (already computed
   to *order* results) doubles as a free confidence signal: if nothing
   retrieved clears `MIN_RELEVANCE_SCORE`, the LLM is never called and the
   system says so directly, instead of generating an answer from irrelevant
   context.
4. **Vision** — if the question mentions a figure/diagram/chart, embedded
   images are extracted from the PDF (`vision.py`, PyMuPDF) and described by
   Gemini's multimodal vision in a cached call.
5. **Answering** — retrieved text + any figure descriptions + recent
   conversation turns are combined into one prompt sent to Gemini in a
   single multimodal call, streamed back token-by-token. The vision call and
   the final answer are both cached (MD5-keyed, LRU-bounded) to avoid
   redundant API usage on repeated questions.

Every session (one browser tab, or one API session) gets its own isolated
vector index and uploaded-document set - concurrent users never see each
other's documents or answers.

## Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and
  on your `PATH` (or set `pytesseract.pytesseract.tesseract_cmd` explicitly)
- [Poppler](https://poppler.freedesktop.org/) installed and on your `PATH`
  (required by `pdf2image` to rasterize PDF pages for OCR)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/)

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements-dev.txt   # includes pytest/httpx; use requirements.txt for runtime only
```

Create a `.env` file in the project root:

```
API_KEY=<your Gemini API key>
```

`.env` is git-ignored — never commit it. (Deploying to Streamlit Community
Cloud instead? Use its Secrets manager — see `.streamlit/secrets.toml.example`.)

## Run

**Streamlit chat UI:**

```bash
streamlit run app.py
```

Upload PDFs in the sidebar, pick which ones to search, and chat. Answers
stream in and prior turns stay visible for follow-up questions.

**FastAPI backend** (a second, independent client of the same pipeline):

```bash
uvicorn api:app --reload --port 8000
```

Open `http://localhost:8000/docs` for interactive Swagger docs. Endpoints:
`POST /sessions`, `POST/GET /sessions/{id}/documents`, `POST /ask`,
`POST /ask/stream` (SSE-style streaming), `DELETE /sessions/{id}/index`.

**Docker** (runs both, with Tesseract/Poppler baked in):

```bash
docker compose up --build
```

UI at `localhost:8501`, API at `localhost:8000`.

## Tests & evaluation

```bash
pytest tests/ -v          # 22 tests: pipeline + API, all offline, no API key spend
python eval/run_eval.py   # retrieval benchmark + CI regression gate, also offline
```

Both run in CI on every push ([.github/workflows/ci.yml](.github/workflows/ci.yml)) —
retrieval quality is checked automatically, not just at demo time.

## Project structure

```
app.py                  Streamlit chat UI: upload, document selection, streaming Q&A
api.py                  FastAPI backend exposing the same pipeline over HTTP
storage.py              Shared per-session upload storage (used by both app.py and api.py)
rag/rag_pipeline.py     Ingestion, hybrid retrieval, reranking, grounding gate, caching, LLM calls
image_processing.py     OCR image preprocessing (grayscale/contrast/sharpen)
vision.py               Embedded-image extraction + Gemini vision analysis
tests/                  pytest suite (pipeline + API), all offline
eval/                   Retrieval benchmark: synthetic docs, QA set, naive-vs-hybrid comparison
Dockerfile, docker-compose.yml   Containerized UI + API with Tesseract/Poppler
.github/workflows/ci.yml         Test suite + retrieval regression gate on every push
```

## Known limitations

- `USE_VISION` analyzes only one image per PDF per vision-triggered question,
  to keep latency and API usage bounded.
- Conversation history is included in the generation prompt but not used to
  rewrite the retrieval query itself (would need an extra LLM call, which
  conflicts with this project's single-call-per-answer design - see the
  docstring on `_format_history` in `rag_pipeline.py`). A multi-turn
  follow-up that depends entirely on unstated context from earlier in the
  conversation can therefore retrieve worse than a fully explicit question.
- Session data (uploads, vector index) lives on local disk under
  `uploaded_docs/` and `chroma_db/sessions/<id>/` with no automatic
  expiry - fine for a demo, would need a cleanup job for long-running
  multi-user deployment.
