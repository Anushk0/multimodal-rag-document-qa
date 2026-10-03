import os
import json
import shutil
import time
import gc
import hashlib
import logging
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import chromadb
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from pdf2image import convert_from_path
import pytesseract

from image_processing import preprocess_image
from vision import extract_images_from_pdf, analyze_image_with_gemini
from document_classifier import ClassificationResult, classify_pdf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _find_tesseract_cmd() -> str | None:
    """Locate the Tesseract binary without relying on the process's PATH,
    which on Windows often doesn't reflect a PATH edit made after this
    process (or its parent shell) started."""
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def _find_poppler_bin() -> str | None:
    """Locate the Poppler bin directory (for pdf2image) the same way -
    checks PATH first, then the winget install location."""
    found = shutil.which("pdftoppm")
    if found:
        return str(Path(found).parent)
    winget_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget_root.exists():
        for candidate in winget_root.glob("*Poppler*/poppler-*/Library/bin/pdftoppm.exe"):
            return str(candidate.parent)
    return None


_TESSERACT_CMD = _find_tesseract_cmd()
if _TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
else:
    logger.warning("Tesseract binary not found; OCR fallback for scanned PDFs will fail.")

_POPPLER_PATH = _find_poppler_bin()
if not _POPPLER_PATH:
    logger.warning("Poppler binaries not found; OCR fallback for scanned PDFs will fail.")

load_dotenv()

api_key = os.getenv("API_KEY")
if not api_key:
    raise RuntimeError(
        "Missing API_KEY. Create a .env file in the project root with "
        "API_KEY=<your Gemini API key> before running the app."
    )
os.environ["GOOGLE_API_KEY"] = api_key

USE_VISION = True

# Every session gets its own subdirectory under BASE_DB_PATH (its own Chroma
# collection + ingestion registry + uploaded_docs), so concurrent users never
# see each other's documents or answers. Passing session_id=None collapses
# back to a single shared index at BASE_DB_PATH itself - used by the eval
# harness, tests, and any single-user/CLI use of this module.
BASE_DB_PATH = "./chroma_db"

# Bumped whenever document_classifier's classification/chunking logic
# changes, so already-ingested files are correctly recognized as needing
# re-chunking under the new logic - the per-document analogue of the old
# single global "did CHUNK_SIZE/CHUNK_OVERLAP change" check, now that chunk
# config is chosen per-document by classify_pdf() rather than fixed.
_CLASSIFIER_VERSION = "v1"

TOP_K_FINAL = 6

# Cross-encoder relevance score below which retrieved context is treated as
# "nothing relevant found" and the LLM call is skipped entirely, instead of
# risking a hallucinated answer over irrelevant context. Empirically,
# ms-marco-MiniLM-L-6-v2 scores genuinely relevant (query, chunk) pairs from
# clean prose around 1 to 3+, but relevant pairs from noisy OCR'd/scanned
# documents (garbled characters, tabular/form layout instead of prose) score
# much lower, around -3 to -6.6 - the reranker just isn't well-calibrated for
# that kind of text. Clearly irrelevant pairs stay around -10 to -11
# regardless of document type, so -8.0 keeps a comfortable margin below real
# matches (even OCR'd ones) while still well above the irrelevant baseline -
# see eval/README.md for the measurements this threshold is based on.
MIN_RELEVANCE_SCORE = -8.0
NO_RELEVANT_CONTEXT_MESSAGE = (
    "I couldn't find information relevant to this question in the selected "
    "document(s). Try rephrasing your question, or check that you've selected "
    "the right file(s) above."
)

_text_splitters: dict[tuple[int, int], RecursiveCharacterTextSplitter] = {}


def _get_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Small cache so repeated (chunk_size, chunk_overlap) pairs - the
    common case, since most documents classify the same way - don't
    construct a new splitter object per document."""
    key = (chunk_size, chunk_overlap)
    if key not in _text_splitters:
        _text_splitters[key] = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
    return _text_splitters[key]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


_MISSING = object()


class _LRUCache:
    """Small size-bounded cache so a long-running Streamlit process doesn't
    grow these dicts forever."""

    def __init__(self, maxsize: int):
        self._maxsize = maxsize
        self._data: "OrderedDict" = OrderedDict()

    def get(self, key, default=None):
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def set(self, key, value) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)


# Vision cache is keyed by image content (a figure's description doesn't
# depend on the question asked about it); LLM cache by the full rendered
# prompt.
_llm_cache = _LRUCache(maxsize=200)
_vision_cache = _LRUCache(maxsize=100)


def needs_vision(query: str) -> bool:
    keywords = ["figure", "diagram", "graph", "image", "chart"]
    return any(k in query.lower() for k in keywords)


def analyze_query(query: str) -> dict:
    q = query.lower()
    return {
        "needs_vision": any(k in q for k in ["figure", "diagram", "graph"]),
        "is_short": len(q.split()) < 4,
        "is_definition": "what is" in q or "define" in q,
        "is_explanatory": "explain" in q or "how" in q,
        "type": "figure" if "figure" in q else "normal",
    }


def bm25_search(query, docs, bm25, top_k=8):
    if not docs or bm25 is None:
        return []
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def cached_llm_call(prompt: str, llm):
    key = hashlib.md5(prompt.encode()).hexdigest()
    cached = _llm_cache.get(key, _MISSING)
    if cached is not _MISSING:
        return cached
    response = llm.invoke(prompt)
    _llm_cache.set(key, response)
    return response


def cached_vision_call(image_bytes: bytes, image_ext: str, llm):
    key = hashlib.md5(image_bytes).hexdigest()
    cached = _vision_cache.get(key, _MISSING)
    if cached is not _MISSING:
        return cached
    description = analyze_image_with_gemini(image_bytes, image_ext, llm)
    _vision_cache.set(key, description)
    return description


def extract_text_with_ocr(pdf_path: str) -> str:
    images = convert_from_path(pdf_path, poppler_path=_POPPLER_PATH)
    ocr_text = ""
    for img in images:
        processed_img = preprocess_image(img)
        text = pytesseract.image_to_string(processed_img, config="--psm 6")
        ocr_text += text + "\n"
    return ocr_text


def rerank_documents(query, docs):
    """Rerank docs by cross-encoder relevance. Returns (doc, score) tuples,
    highest relevance first. The score doubles as a free grounding signal:
    a low top score means nothing relevant was actually retrieved (see
    MIN_RELEVANCE_SCORE in answer_question)."""
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _registry_path(db_path: str) -> str:
    return os.path.join(db_path, "ingested_files.json")


def _load_registry(db_path: str) -> dict:
    registry_path = _registry_path(db_path)
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_registry(db_path: str, registry: dict) -> None:
    os.makedirs(db_path, exist_ok=True)
    with open(_registry_path(db_path), "w") as f:
        json.dump(registry, f)


def _session_db_path(session_id: str | None) -> str:
    if not session_id:
        return BASE_DB_PATH
    safe_id = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_")) or "default"
    return os.path.join(BASE_DB_PATH, "sessions", safe_id)


def _load_pdf_as_documents(pdf_path: str) -> tuple[list[Document], ClassificationResult]:
    """Load one PDF's pages, falling back to OCR when native text is too
    sparse, and classify the document's overall type so the caller can pick
    a chunking strategy suited to it (see document_classifier.py).

    Returns (page_docs, classification) rather than just page_docs, so
    ingest_documents can choose per-document chunk size without a second
    pass over the file.
    """
    loader = PyPDFLoader(pdf_path)
    page_docs = loader.load()

    total_text = sum(len(d.page_content.strip()) for d in page_docs)
    used_ocr = total_text < 500
    if used_ocr:
        try:
            ocr_text = extract_text_with_ocr(pdf_path)
            if ocr_text.strip():
                page_docs.append(
                    Document(page_content=ocr_text, metadata={"page": "ocr"})
                )
        except Exception:
            logger.exception("OCR failed for %s", pdf_path)

    source_name = Path(pdf_path).name
    for doc in page_docs:
        doc.metadata["source"] = source_name

    classification = classify_pdf(pdf_path, page_docs, used_ocr=used_ocr)
    logger.info("Classified %s as %s (%s)", source_name, classification.doc_type.value, classification.reason)

    return page_docs, classification


def get_vectorstore(session_id: str | None = None) -> Chroma:
    db_path = _session_db_path(session_id)
    os.makedirs(db_path, exist_ok=True)
    return Chroma(persist_directory=db_path, embedding_function=embeddings)


def reset_index(session_id: str | None = None) -> None:
    """Wipe the persistent vector store and ingestion registry from disk."""
    db_path = _session_db_path(session_id)
    if not os.path.exists(db_path):
        return

    # chromadb caches System instances (and their open sqlite/mmap handles)
    # keyed by persist path. A Chroma object created earlier in this process
    # (e.g. by the question just answered) can still hold those handles open
    # via a reference cycle that plain refcounting won't clear - which makes
    # the directory undeletable on Windows until they're released. gc.collect()
    # breaks the cycle so clear_system_cache()'s released handles actually
    # close; the retry loop is a defensive fallback in case release isn't
    # instantaneous.
    gc.collect()
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    gc.collect()

    for attempt in range(5):
        try:
            shutil.rmtree(db_path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2)


def ingest_documents(pdf_paths, vectorstore) -> tuple[int, list[str]]:
    """Embed any new, changed, or re-configured PDFs into the persistent
    vector store.

    Each file's content hash is tracked in a registry on disk (colocated
    with the vectorstore's own persist_directory, so each session's
    registry stays with its own index), alongside the classifier version
    that decided its chunking strategy (see document_classifier.py).
    Re-running this for an unchanged file under an unchanged classifier is
    a no-op (skips loading/OCR/classification/chunking/embedding entirely -
    deliberately checked *before* loading the file, so this stays cheap for
    the common "nothing changed" case). If a file's content changes, or
    _CLASSIFIER_VERSION is bumped, its old chunks are deleted before
    re-ingesting so the index never accumulates stale duplicates.

    Returns (new_chunk_count, failed_sources). A file that yields zero
    chunks (no native text and OCR failed or found nothing) is deliberately
    left out of the registry - so it's retried on the next ingest call
    instead of being permanently marked "already indexed" while actually
    contributing nothing to the index.
    """
    db_path = vectorstore._persist_directory or BASE_DB_PATH
    registry = _load_registry(db_path)
    new_chunks = []
    new_ids = []
    failed_sources = []

    for pdf_path in pdf_paths:
        path_str = str(pdf_path)
        source_name = Path(path_str).name
        file_hash = _file_hash(path_str)
        entry = registry.get(path_str)

        if (
            entry
            and entry.get("file_hash") == file_hash
            and entry.get("classifier_version") == _CLASSIFIER_VERSION
        ):
            continue

        if entry:
            stale = vectorstore.get(where={"source": source_name})
            if stale.get("ids"):
                vectorstore.delete(ids=stale["ids"])

        page_docs, classification = _load_pdf_as_documents(path_str)
        splitter = _get_text_splitter(classification.chunk_size, classification.chunk_overlap)
        chunks = splitter.split_documents(page_docs)

        if not chunks:
            logger.warning(
                "No extractable text in %s (native text and OCR both empty/failed); "
                "leaving out of registry so it's retried next time.",
                source_name,
            )
            failed_sources.append(source_name)
            continue

        for i, chunk in enumerate(chunks):
            content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:10]
            chunk_id = f"{source_name}::{chunk.metadata.get('page', 'na')}::{i}::{content_hash}"
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

        registry[path_str] = {
            "file_hash": file_hash,
            "classifier_version": _CLASSIFIER_VERSION,
            "doc_type": classification.doc_type.value,
            "chunk_config": f"{classification.chunk_size}-{classification.chunk_overlap}",
        }

    if new_chunks:
        vectorstore.add_documents(new_chunks, ids=new_ids)
        _save_registry(db_path, registry)
        logger.info("Ingested %d new chunk(s).", len(new_chunks))

    return len(new_chunks), failed_sources


def ingest_uploaded_pdfs(pdf_paths, session_id: str | None = None) -> tuple[int, list[str]]:
    """Convenience wrapper so the UI can index files right after upload,
    instead of waiting for the first question."""
    vectorstore = get_vectorstore(session_id)
    return ingest_documents(pdf_paths, vectorstore)


def _get_all_indexed_documents(vectorstore, where: dict | None = None):
    """Fetch chunks currently in the vector store (optionally filtered), so
    BM25 always searches the same document set as the semantic retriever."""
    data = vectorstore.get(where=where) if where else vectorstore.get()
    texts = data.get("documents") or []
    metadatas = data.get("metadatas") or [{}] * len(texts)
    return [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metadatas)]


def get_indexed_chunk_count(session_id: str | None = None) -> int:
    """Total chunks currently in a session's vector store - a small public
    entry point so callers (the UI) don't need to reach into the
    underscore-prefixed retrieval internals just to show a chunk count."""
    vectorstore = get_vectorstore(session_id)
    return len(_get_all_indexed_documents(vectorstore))


def _format_source(doc: Document) -> str:
    source = doc.metadata.get("source", "unknown document")
    page = doc.metadata.get("page")
    if page == "ocr":
        return f"{source} (OCR-extracted)"
    if page is not None:
        try:
            return f"{source}, page {int(page) + 1}"
        except (TypeError, ValueError):
            return f"{source}, page {page}"
    return source


def _format_history(history, max_turns: int = 3) -> str:
    """Render the last few (question, answer) turns for prompt context.

    This is separate from _build_retrieval_query, which augments only the
    single most recent turn for retrieval/reranking. Here we show the LLM
    up to max_turns turns for generation, since more context helps it write
    a coherent conversational answer even though it doesn't affect what got
    retrieved.
    """
    if not history:
        return ""
    recent = list(history)[-max_turns:]
    turns = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in recent)
    return (
        "Previous conversation in this session (for context only - answer "
        f"the NEW question below, not these):\n{turns}\n\n---\n\n"
    )


def _build_retrieval_query(question: str, history) -> str:
    """Augment the query used for retrieval, BM25, and reranking with the
    most recent 2 turns, so an elliptical follow-up ("is that normal?", "is it
    higher than usual?") that shares no keywords with the document can
    still surface the chunk its previous answer was grounded in.

    Without this, retrieval/reranking/the grounding gate all see only the
    bare follow-up text, score it as irrelevant to every chunk, and reject
    the question before the LLM - which has the history and could actually
    resolve the pronoun - ever gets a turn. The final generation prompt
    still shows the LLM the raw current question via _format_history/the
    prompt template; this only changes what retrieval and the gate see.
    """
    if not history:
        return question
    recent = list(history)[-2:]
    prior_turns = "\n\n".join(f"Previous question: {q}\nPrevious answer: {a}" for q, a in recent)
    return f"{prior_turns}\n\nFollow-up question: {question}"


def _extract_text(response) -> str:
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, list):
            return " ".join(item.get("text", "") for item in content if isinstance(item, dict))
        return str(content)
    return str(response)


def _emit_step(
    trace: dict | None,
    on_step: Callable[[dict], None] | None,
    label: str,
    detail: str = "",
    **extra,
) -> None:
    """Record one step of "what the pipeline just did" for display.

    Appends to trace["steps"] (a running history, shown after the fact in
    the trace panel) *and* calls on_step with the same event (for a caller
    that wants to render progress live, as it happens - e.g. a Streamlit
    st.status() block - rather than only once the whole answer is done).
    Both are optional and independent: a caller can want one, both, or
    neither.
    """
    event = {"label": label, "detail": detail, **extra}
    if trace is not None:
        trace.setdefault("steps", []).append(event)
    if on_step is not None:
        on_step(event)


def get_image_descriptions(
    question: str,
    pdf_paths,
    llm,
    trace: dict | None = None,
    on_step: Callable[[dict], None] | None = None,
) -> list[str]:
    """Vision (image/figure) understanding: if the question mentions a
    visual element, pull one image per selected PDF and describe it.

    Runs once per top-level question over the full selected pdf_paths, not
    per sub-question - image descriptions aren't source-scoped the way text
    retrieval is, so there's no reason a future multi-hop/decomposed query
    would need to repeat this per sub-question.
    """
    image_descriptions: list[str] = []
    if not (USE_VISION and needs_vision(question)):
        return image_descriptions

    _emit_step(trace, on_step, "Checking for figures/diagrams", "question mentions a visual element")
    for pdf_path in pdf_paths:
        try:
            images = extract_images_from_pdf(str(pdf_path))
            for image_bytes, image_ext in images[:1]:  # only 1 image per PDF
                desc = cached_vision_call(image_bytes, image_ext, llm)
                if desc:
                    image_descriptions.append(f"[{Path(pdf_path).name}] {desc}")
        except Exception:
            logger.exception("Vision extraction failed for %s", pdf_path)
    _emit_step(
        trace, on_step, "Figure analysis done",
        f"{len(image_descriptions)} image(s) described" if image_descriptions else "no usable images found",
    )
    return image_descriptions


@dataclass
class RetrievalResult:
    """Reranked (doc, score) pairs for one retrieval, highest relevance
    first, plus the counts needed for trace/eval reporting. `grounded`
    reflects only this retrieval's own score against MIN_RELEVANCE_SCORE -
    a caller with an independent grounding signal (e.g. a figure
    description) is expected to OR it in itself, the same way
    _prepare_prompt already does today."""

    reranked_docs: list[tuple[Document, float]]
    total_chunks_indexed: int
    semantic_hit_count: int
    bm25_hit_count: int

    @property
    def top_score(self) -> float | None:
        return float(self.reranked_docs[0][1]) if self.reranked_docs else None

    @property
    def grounded(self) -> bool:
        score = self.top_score
        return score is not None and score >= MIN_RELEVANCE_SCORE

    def final_docs(self, top_k: int = TOP_K_FINAL) -> list[Document]:
        return [doc for doc, _ in self.reranked_docs[:top_k]]


def retrieve_and_rerank(
    question: str,
    retrieval_query: str,
    vectorstore,
    selected_sources: list[str],
    trace: dict | None = None,
    on_step: Callable[[dict], None] | None = None,
) -> RetrievalResult:
    """Hybrid retrieval (MMR semantic + BM25) over the given sources,
    merged and reranked by cross-encoder relevance. This is the reusable
    core shared by the simple pipeline, a future multi-hop/decomposed agent
    (once per sub-question, each with its own retrieval_query and possibly
    a narrower selected_sources), and eval/run_ragas_eval.py (which needs
    raw retrieved chunk text, not just the formatted display strings the
    trace panel stores).

    `question` (raw) and `retrieval_query` (history-folded via
    _build_retrieval_query, or otherwise pre-built by a caller) are kept
    separate deliberately: `question` only drives analyze_query()'s
    figure-type branch weighting below, while `retrieval_query` is what
    actually gets embedded/matched/reranked - preserving the exact split
    the original single-pass pipeline used.
    """
    source_filter = {"source": {"$in": selected_sources}}
    all_docs = _get_all_indexed_documents(vectorstore, where=source_filter)

    tokenized_corpus = [doc.page_content.split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    enhanced_query = f"""
    User question: {retrieval_query}

    This question may refer to:
    - figures
    - diagrams
    - images
    - charts

    Interpret references like "figure 1" carefully.
    """

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 15,
            "fetch_k": 30,
            "lambda_mult": 0.5,
            "filter": source_filter,
        },
    )

    _emit_step(trace, on_step, "Retrieving", f"searching {len(selected_sources)} document(s) for: {question}")
    semantic_docs = retriever.invoke(enhanced_query)
    bm25_docs = bm25_search(retrieval_query, all_docs, bm25)

    plan = analyze_query(question)
    if plan["type"] == "figure":
        combined_docs = semantic_docs[:10] + bm25_docs[:4]
    else:
        combined_docs = semantic_docs[:8] + bm25_docs[:5]

    unique_docs = list({doc.page_content: doc for doc in combined_docs}.values())
    _emit_step(
        trace, on_step, "Reranking candidates",
        f"{len(semantic_docs)} semantic + {len(bm25_docs)} keyword hits -> {len(unique_docs)} unique candidate(s)",
    )
    reranked_docs = rerank_documents(retrieval_query, unique_docs)

    result = RetrievalResult(
        reranked_docs=reranked_docs,
        total_chunks_indexed=len(all_docs),
        semantic_hit_count=len(semantic_docs),
        bm25_hit_count=len(bm25_docs),
    )
    top_score = result.top_score
    _emit_step(
        trace, on_step, "Checking relevance",
        (
            f"top match score {round(top_score, 2)} vs threshold {MIN_RELEVANCE_SCORE} -> "
            + ("passed" if result.grounded else "below threshold")
        )
        if top_score is not None
        else "no candidates retrieved",
    )
    return result


def _build_final_prompt(
    question: str,
    final_docs: list[Document],
    image_descriptions: list[str],
    history,
) -> str:
    """Assemble the context block and generation prompt from already-chosen
    final_docs/image_descriptions. Split out so the (common) single-retrieval
    case in a future multi-hop agent can build a provably-identical prompt
    to today's, rather than a reimplementation that could silently drift."""
    context_parts = []
    if image_descriptions:
        context_parts.append(
            "IMPORTANT VISUAL INFORMATION FROM FIGURE:\n" + "\n\n".join(image_descriptions)
        )

    text_section = "\n\n".join(
        f"{doc.page_content}\n(Source: {_format_source(doc)})" for doc in final_docs
    )
    context_parts.append("TEXT INFORMATION:\n" + text_section)
    context = "\n\n---\n\n".join(context_parts)

    history_block = _format_history(history)

    final_prompt = f"""
{history_block}You are an AI assistant.
You are given TWO types of information:
1. VISUAL DESCRIPTION (from image/figure)
2. TEXT DESCRIPTION (from document)

IMPORTANT INSTRUCTIONS:

- If a figure is mentioned:
  - PRIORITIZE visual information over text
  - Describe diagram components (boxes, arrows, flow)
  - Then combine with text
  - If visual and text differ, trust visual more

- If visual information is incomplete:
  -> Use text to fill missing details

- If text is incomplete:
  -> Use visual description to infer structure

- Do NOT ignore either source
- Explain clearly and step-by-step when describing diagrams, describe components and flow

Answer the question using the context below, but:
- Explain in simple terms
- Do NOT copy text directly
- Summarize and make it easy to understand
- Answer based only on the context.
- At the end, mention the source (page number or document name) shown after each piece of context above.
- Do NOT add your own knowledge.
- Do NOT generalize or expand.

Context:
{context}

Question:
{question}

Answer:
"""
    logger.info("Context built (%d chars, %d source chunk(s))", len(context), len(final_docs))
    return final_prompt


def _prepare_prompt(
    question: str,
    pdf_paths,
    llm,
    session_id=None,
    history=None,
    trace: dict | None = None,
    on_step: Callable[[dict], None] | None = None,
):
    """Runs ingestion, hybrid retrieval, reranking, vision, and the
    grounding gate, and returns the final prompt ready for the LLM - or
    None if nothing relevant was retrieved (caller should show
    NO_RELEVANT_CONTEXT_MESSAGE instead of calling the LLM).

    A thin wrapper over get_image_descriptions/retrieve_and_rerank/
    _build_final_prompt - kept as a single entry point since it's still the
    simple (non-agentic) pipeline's only caller-facing function, but the
    actual work is in those three, each independently reusable.

    If `trace` is given, it's mutated in place with diagnostics about this
    call (documents searched, retrieval hit counts, rerank scores, timing).
    This is an out-parameter rather than part of the return value so
    existing callers (including tests calling this directly) don't have to
    change - and so this module stays UI-agnostic, with no knowledge of how
    a caller might display the trace.
    """
    t_start = time.perf_counter()
    vectorstore = get_vectorstore(session_id)
    ingest_documents(pdf_paths, vectorstore)

    # Scope retrieval to only the documents the user selected for this
    # question, not every PDF ever uploaded to the app.
    selected_sources = sorted({Path(p).name for p in pdf_paths})

    all_docs_count = len(_get_all_indexed_documents(vectorstore, where={"source": {"$in": selected_sources}}))
    _emit_step(
        trace, on_step, "Preparing documents",
        f"{len(selected_sources)} document(s), {all_docs_count} chunk(s) indexed",
    )

    image_descriptions = get_image_descriptions(question, pdf_paths, llm, trace=trace, on_step=on_step)

    retrieval_query = _build_retrieval_query(question, history)
    result = retrieve_and_rerank(
        question, retrieval_query, vectorstore, selected_sources, trace=trace, on_step=on_step
    )
    retrieval_ms = round((time.perf_counter() - t_start) * 1000, 1)

    grounded = bool(image_descriptions) or result.grounded

    if trace is not None:
        trace.update(
            {
                "documents_searched": selected_sources,
                "total_chunks_indexed": result.total_chunks_indexed,
                "semantic_hits": result.semantic_hit_count,
                "bm25_hits": result.bm25_hit_count,
                "candidates_after_merge": len(result.reranked_docs),
                "vision_triggered": bool(image_descriptions),
                "top_score": round(result.top_score, 2) if result.top_score is not None else None,
                "grounding_threshold": MIN_RELEVANCE_SCORE,
                "grounded": grounded,
                "retrieval_ms": retrieval_ms,
            }
        )

    # Grounding gate: if nothing relevant was retrieved and there's no figure
    # description to fall back on, don't spend an LLM call generating an
    # answer from irrelevant context - just say so.
    if not grounded:
        logger.info(
            "Grounding gate: top relevance score %s below threshold %.1f; skipping LLM call.",
            result.top_score,
            MIN_RELEVANCE_SCORE,
        )
        _emit_step(trace, on_step, "Stopping", "nothing relevant found - skipping the LLM call")
        return None

    final_docs = result.final_docs()

    if trace is not None:
        trace["sources_used"] = [
            {"source": _format_source(doc), "score": round(float(score), 2)}
            for doc, score in result.reranked_docs[:TOP_K_FINAL]
        ]
    _emit_step(
        trace, on_step, "Building context",
        f"using {len(final_docs)} chunk(s): " + ", ".join(_format_source(d) for d in final_docs),
    )

    return _build_final_prompt(question, final_docs, image_descriptions, history)


def answer_question(
    question: str,
    pdf_paths,
    session_id=None,
    history=None,
    trace: dict | None = None,
    on_step: Callable[[dict], None] | None = None,
) -> str:
    if not pdf_paths:
        return "No documents selected. Upload and select at least one PDF first."

    t_start = time.perf_counter()

    # Low temperature: the prompt asks for strict extraction from the given
    # context ("answer based only on the context... do not add your own
    # knowledge"), which is closer to a deterministic task than creative
    # writing. At 0.5, the same correct context occasionally produced a
    # wrong "not found in the document" answer for data that was right
    # there in the prompt - observed directly while testing follow-up
    # questions. Lower temperature doesn't make this impossible, but makes
    # the model far more likely to actually read the context it was given.
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)
    final_prompt = _prepare_prompt(
        question, pdf_paths, llm, session_id=session_id, history=history, trace=trace, on_step=on_step
    )

    if final_prompt is None:
        if trace is not None:
            trace["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        return NO_RELEVANT_CONTEXT_MESSAGE

    try:
        key = hashlib.md5(final_prompt.encode()).hexdigest()
        cache_hit = _llm_cache.get(key, _MISSING) is not _MISSING
        _emit_step(trace, on_step, "Generating answer", "served from cache" if cache_hit else "calling the model")
        response = cached_llm_call(final_prompt, llm)
        if trace is not None:
            trace["cache_hit"] = cache_hit
            trace["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        return _extract_text(response).strip()
    except Exception as e:
        logger.exception("LLM call failed")
        return f"LLM Error: {e}"


def stream_answer(
    question: str,
    pdf_paths,
    session_id=None,
    history=None,
    trace: dict | None = None,
    on_step: Callable[[dict], None] | None = None,
):
    """Generator yielding the answer incrementally as it's generated.

    A cache hit yields the whole cached answer in one piece (nothing to
    stream - it's already complete); a cache miss streams token-by-token
    from Gemini and caches the fully assembled answer under the same key
    cached_llm_call would have used, so a repeat of this exact question
    (same prompt hash) is instant next time either way.

    If `trace` is given, it's mutated in place with retrieval/timing
    diagnostics for the caller to display (see _prepare_prompt). If
    `on_step` is given, it's called live as each pipeline step happens
    (see _emit_step) - e.g. to drive a Streamlit st.status() block.
    """
    t_start = time.perf_counter()

    if not pdf_paths:
        yield "No documents selected. Upload and select at least one PDF first."
        return

    # Low temperature: the prompt asks for strict extraction from the given
    # context ("answer based only on the context... do not add your own
    # knowledge"), which is closer to a deterministic task than creative
    # writing. At 0.5, the same correct context occasionally produced a
    # wrong "not found in the document" answer for data that was right
    # there in the prompt - observed directly while testing follow-up
    # questions. Lower temperature doesn't make this impossible, but makes
    # the model far more likely to actually read the context it was given.
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)
    final_prompt = _prepare_prompt(
        question, pdf_paths, llm, session_id=session_id, history=history, trace=trace, on_step=on_step
    )

    if final_prompt is None:
        if trace is not None:
            trace["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        yield NO_RELEVANT_CONTEXT_MESSAGE
        return

    key = hashlib.md5(final_prompt.encode()).hexdigest()
    cached = _llm_cache.get(key, _MISSING)
    if cached is not _MISSING:
        _emit_step(trace, on_step, "Generating answer", "served from cache")
        if trace is not None:
            trace["cache_hit"] = True
            trace["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        yield _extract_text(cached).strip()
        return

    if trace is not None:
        trace["cache_hit"] = False
    _emit_step(trace, on_step, "Generating answer", "streaming from the model")

    try:
        accumulated = ""
        for chunk in llm.stream(final_prompt):
            piece = _extract_text(chunk)
            if piece:
                accumulated += piece
                yield piece
        _llm_cache.set(key, accumulated.strip())
        if trace is not None:
            trace["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
    except Exception as e:
        logger.exception("Streaming LLM call failed")
        yield f"LLM Error: {e}"
