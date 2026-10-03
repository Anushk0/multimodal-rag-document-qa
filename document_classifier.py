"""Cheap, zero-LLM-cost per-document classification, used to pick a chunking
strategy suited to how a document actually reads (clean digital prose vs.
noisy OCR'd scans vs. dense numeric tables) instead of applying one fixed
chunk size to every upload regardless of shape.

Deliberately never calls an LLM: this runs once per upload, and burning a
Gemini call per document would compete with the same daily quota the actual
Q&A needs (see eval/README.md's grounding-gate notes for how tight that
budget already is). Every signal here is a cheap heuristic over text/layout
already available from PyPDFLoader/PyMuPDF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import fitz  # PyMuPDF
from langchain_core.documents import Document


class DocType(str, Enum):
    BORN_DIGITAL = "born_digital"
    SCANNED_IMAGE = "scanned_image"
    DENSE_TABLE = "dense_table"


@dataclass
class ClassificationResult:
    doc_type: DocType
    chunk_size: int
    chunk_overlap: int
    reason: str


# BORN_DIGITAL matches this project's original fixed chunk config exactly,
# so a born-digital document chunks byte-identically to before this
# classifier existed - it's the "no unusual signal, behave as before" case.
_CHUNK_CONFIG: dict[DocType, tuple[int, int]] = {
    DocType.BORN_DIGITAL: (800, 150),
    # OCR'd text has noisier chunk boundaries (broken words, garbled
    # characters). eval/README.md's own measurements show relevant OCR'd
    # chunks already score much lower on the reranker than clean prose
    # (-3 to -6.6 vs +1 to +3) - a larger, more-overlapping chunk gives the
    # reranker more surrounding context per candidate to work with.
    DocType.SCANNED_IMAGE: (1000, 250),
    # Dense numeric tables (many short rows) benefit from smaller chunks so
    # one chunk doesn't straddle unrelated rows, diluting a specific value
    # with neighboring rows' numbers.
    DocType.DENSE_TABLE: (450, 60),
}

# Non-printable-ASCII "noise" characters - a rough proxy for OCR garbling in
# text that technically extracted natively but is actually low-quality
# (e.g. a PDF with an embedded but corrupted text layer).
_GARBLED_CHAR_PATTERN = re.compile(r"[^\x20-\x7E\n]")
_GARBLED_RATIO_THRESHOLD = 0.05
_NUMERIC_TOKEN_PATTERN = re.compile(r"^[\d.,%<>=/\-]+$")


def _garbled_ratio(text: str) -> float:
    if not text:
        return 0.0
    noise = len(_GARBLED_CHAR_PATTERN.findall(text))
    return noise / len(text)


def _looks_like_dense_table(pdf_path: str, max_pages: int = 3) -> bool:
    """A page that's mostly short numeric tokens on short lines reads as a
    table/form rather than prose - checked via PyMuPDF's word-level layout
    (fitz is already a project dependency via vision.py), not an LLM call."""
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return False
    try:
        numeric_tokens = 0
        total_tokens = 0
        short_lines = 0
        total_lines = 0
        for page in doc[:max_pages]:
            words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
            total_tokens += len(words)
            numeric_tokens += sum(1 for w in words if _NUMERIC_TOKEN_PATTERN.match(w[4]))
            lines = [ln for ln in page.get_text().splitlines() if ln.strip()]
            total_lines += len(lines)
            # 45, not ~25: a real table row (label + value + reference
            # range + unit, e.g. "SODIUM 140 136 - 146 mmol/L") commonly
            # runs 28-35 chars - calibrated against a synthetic table PDF
            # after an initial 25-char threshold missed it entirely despite
            # a correctly-high numeric-token ratio for the same rows.
            short_lines += sum(1 for ln in lines if len(ln.strip()) <= 45)
    finally:
        doc.close()

    if total_tokens < 20 or total_lines < 5:
        return False  # too little content on the sampled pages to judge
    return (numeric_tokens / total_tokens) > 0.25 and (short_lines / total_lines) > 0.5


def classify_pdf(pdf_path: str, page_docs: list[Document], used_ocr: bool) -> ClassificationResult:
    """Classify one PDF's overall shape from its already-loaded page
    Documents, plus a direct look at the file for the table-density signal
    (word-level layout isn't preserved in PyPDFLoader's plain-text output).

    `used_ocr` is passed in rather than re-derived from the (post-OCR) text,
    since by the time OCR has run successfully the text is no longer sparse
    - the caller (_load_pdf_as_documents) already knows definitively
    whether it had to fall back to OCR, so this reuses that decision
    instead of guessing at it a second time from a different signal.
    """
    if used_ocr:
        chunk_size, chunk_overlap = _CHUNK_CONFIG[DocType.SCANNED_IMAGE]
        return ClassificationResult(
            DocType.SCANNED_IMAGE, chunk_size, chunk_overlap,
            reason="native text was too sparse; OCR fallback was used",
        )

    full_text = "\n".join(d.page_content for d in page_docs)
    garbled = _garbled_ratio(full_text)
    if garbled > _GARBLED_RATIO_THRESHOLD:
        chunk_size, chunk_overlap = _CHUNK_CONFIG[DocType.SCANNED_IMAGE]
        return ClassificationResult(
            DocType.SCANNED_IMAGE, chunk_size, chunk_overlap,
            reason=f"{garbled:.0%} garbled/non-printable characters in native text",
        )

    if _looks_like_dense_table(pdf_path):
        chunk_size, chunk_overlap = _CHUNK_CONFIG[DocType.DENSE_TABLE]
        return ClassificationResult(
            DocType.DENSE_TABLE, chunk_size, chunk_overlap,
            reason="page layout looks like a dense numeric table/grid",
        )

    chunk_size, chunk_overlap = _CHUNK_CONFIG[DocType.BORN_DIGITAL]
    return ClassificationResult(
        DocType.BORN_DIGITAL, chunk_size, chunk_overlap,
        reason="normal extractable text, no unusual layout signals",
    )
