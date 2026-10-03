"""Tests for document_classifier.py - all synthetic PDFs, zero LLM cost,
zero network. Mirrors the classification checks run manually against this
project's two real test documents (a clean digital PDF and a scanned fax
needing OCR) while keeping those real files out of the repo.
"""

import sys
from pathlib import Path

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_classifier import DocType, classify_pdf


def _make_pdf(path: Path, lines: list[str], fontsize: int = 11) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=fontsize)
        y += fontsize + 3
        if y > 750:
            page = doc.new_page()
            y = 72
    doc.save(str(path))
    doc.close()


def _load_page_docs(path: Path):
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(str(path)).load()


def test_prose_classifies_as_born_digital_with_default_chunking(tmp_path):
    pdf_path = tmp_path / "prose.pdf"
    _make_pdf(
        pdf_path,
        ["Retrieval augmented generation combines a retriever with a language model."] * 20,
    )
    page_docs = _load_page_docs(pdf_path)

    result = classify_pdf(str(pdf_path), page_docs, used_ocr=False)

    assert result.doc_type == DocType.BORN_DIGITAL
    assert (result.chunk_size, result.chunk_overlap) == (800, 150)


def test_dense_numeric_table_classifies_as_dense_table(tmp_path):
    pdf_path = tmp_path / "table.pdf"
    _make_pdf(
        pdf_path,
        [f"SODIUM {140 + i}  3.7 - 5.4  mmol/L" for i in range(40)],
        fontsize=9,
    )
    page_docs = _load_page_docs(pdf_path)

    result = classify_pdf(str(pdf_path), page_docs, used_ocr=False)

    assert result.doc_type == DocType.DENSE_TABLE
    assert result.chunk_size < 800  # smaller than the prose default, on purpose


def test_ocr_fallback_forces_scanned_image_classification(tmp_path):
    # used_ocr=True should classify as SCANNED_IMAGE regardless of what the
    # (post-OCR) text looks like - the caller already knows definitively
    # that native extraction was too sparse, so this shouldn't be re-derived
    # from the OCR'd text's own density.
    pdf_path = tmp_path / "scanned.pdf"
    _make_pdf(pdf_path, ["Plenty of normal-looking prose text right here."] * 20)
    page_docs = _load_page_docs(pdf_path)

    result = classify_pdf(str(pdf_path), page_docs, used_ocr=True)

    assert result.doc_type == DocType.SCANNED_IMAGE
    assert result.chunk_size > 800  # larger than the prose default, on purpose


def test_garbled_native_text_classifies_as_scanned_image_without_ocr(tmp_path):
    from langchain_core.documents import Document

    garbled_docs = [
        Document(page_content="���� garbled ��� text ����" * 20)
    ]

    result = classify_pdf("unused-path.pdf", garbled_docs, used_ocr=False)

    assert result.doc_type == DocType.SCANNED_IMAGE
    assert "garbled" in result.reason
