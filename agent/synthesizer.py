"""Combines retrieved evidence from one or more sub-questions into a single
generation call. Decomposition/multi-hop never multiplies the LLM
generation cost - only the (free, local) retrieval step fans out per
sub-question; synthesis always stays one call, regardless of how many
sub-questions were planned.

For the common single-sub-question case, this builds its prompt via
rag_pipeline._build_final_prompt - the exact function the simple pipeline
itself uses - so that case is provably identical to today's output rather
than a second implementation that could quietly drift from it.
"""

from __future__ import annotations

from rag.rag_pipeline import RetrievalResult, TOP_K_FINAL, _build_final_prompt, _extract_text

from .state import SubQuestion

_STRICT_SUFFIX = (
    "\n\nIMPORTANT: a previous answer to this exact question included information "
    "not present in the context above. Answer again using ONLY facts explicitly "
    "stated in the context. If the context doesn't fully answer the question, say "
    "so explicitly rather than filling in the gap."
)


def _merge_top_docs(sub_results: list[tuple[SubQuestion, RetrievalResult]], top_k: int = TOP_K_FINAL):
    """Merge each sub-question's top-k reranked docs into one list, deduped
    by content (the same chunk can legitimately answer more than one
    sub-question - e.g. a shared header/summary row), keeping the highest
    score seen for a chunk that appears more than once."""
    best: dict[str, tuple] = {}
    for _sub_q, result in sub_results:
        for doc, score in result.reranked_docs[:top_k]:
            key = doc.page_content
            if key not in best or score > best[key][1]:
                best[key] = (doc, score)
    merged = sorted(best.values(), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [doc for doc, _score in merged]


def synthesize_answer(
    question: str,
    sub_results: list[tuple[SubQuestion, RetrievalResult]],
    image_descriptions: list[str],
    llm,
    history=None,
    strict: bool = False,
) -> str:
    final_docs = _merge_top_docs(sub_results)
    prompt = _build_final_prompt(question, final_docs, image_descriptions, history)
    if strict:
        prompt += _STRICT_SUFFIX

    response = llm.invoke(prompt)
    return _extract_text(response).strip()
