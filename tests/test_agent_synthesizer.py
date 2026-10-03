"""Tests for agent/synthesizer.py using a stub LLM (no network) - verifies
merge/dedup control flow and the strict-regeneration prompt suffix, not
answer quality (that's eval/run_ragas_eval.py's job).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from agent.state import SubQuestion
from agent.synthesizer import synthesize_answer
from rag.rag_pipeline import RetrievalResult


class _RecordingLLM:
    def __init__(self):
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return SimpleNamespace(content="stub answer")


def _result_with(doc_text: str, source: str, score: float) -> RetrievalResult:
    doc = Document(page_content=doc_text, metadata={"source": source})
    return RetrievalResult(reranked_docs=[(doc, score)], total_chunks_indexed=1, semantic_hit_count=1, bm25_hit_count=0)


def test_synthesize_merges_and_dedupes_across_subquestions():
    sub_results = [
        (SubQuestion(text="q1", target_sources=None), _result_with("Sodium is 140 mmol/L.", "a.pdf", 2.0)),
        (SubQuestion(text="q2", target_sources=None), _result_with("Sodium is 140 mmol/L.", "a.pdf", 3.0)),  # dup, higher score
        (SubQuestion(text="q3", target_sources=None), _result_with("Potassium is 4.9 mmol/L.", "a.pdf", 1.0)),
    ]
    llm = _RecordingLLM()

    answer = synthesize_answer("q", sub_results, image_descriptions=[], llm=llm)

    assert answer == "stub answer"
    assert llm.last_prompt.count("Sodium is 140 mmol/L.") == 1  # deduped, not repeated
    assert "Potassium is 4.9 mmol/L." in llm.last_prompt


def test_synthesize_strict_mode_appends_stricter_instruction():
    sub_results = [(SubQuestion(text="q", target_sources=None), _result_with("fact.", "a.pdf", 1.0))]
    llm = _RecordingLLM()

    synthesize_answer("q", sub_results, image_descriptions=[], llm=llm, strict=False)
    non_strict_prompt = llm.last_prompt

    synthesize_answer("q", sub_results, image_descriptions=[], llm=llm, strict=True)
    strict_prompt = llm.last_prompt

    assert len(strict_prompt) > len(non_strict_prompt)
    assert "ONLY facts explicitly" in strict_prompt
