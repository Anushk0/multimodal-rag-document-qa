# Retrieval Evaluation

`run_eval.py` benchmarks this project's retrieval pipeline (MMR semantic
search + BM25 + cross-encoder reranking) against a naive baseline (plain
vector `similarity_search`), using a small, reproducible, synthetic
question/answer set - not real user data, so this runs identically for
anyone who clones the repo.

Retrieval only - no LLM calls, no API key required, no cost. Safe to run
in CI on every push (see `.github/workflows/ci.yml`).

```bash
python eval/build_sample_docs.py   # regenerate eval/sample_docs/*.pdf (already committed)
python eval/run_eval.py            # prints + saves eval/results.json
```

## Corpus

Three short synthetic PDFs (`sample_docs/`): a company handbook, an API
product spec, and a research note - about a dozen chunks total at this
project's chunk size. Two of the documents include a deliberate **hard
negative**: a "historical/legacy" paragraph that reuses the same topic and
phrasing as the real answer but with a different, outdated number (e.g. a
superseded $75 reimbursement threshold next to the current $50 one). This is
a standard IR evaluation technique for stress-testing whether retrieval
grabs the semantically-similar-but-wrong chunk instead of the correct one.

## Metrics

At this corpus scale (~12 chunks), `recall@5` is trivially 1.0 for any
reasonable strategy - the whole topic cluster fits in the top 5. That's not
a useful signal here, so instead we report:

- **recall@1 / recall@3** - is the single most relevant chunk actually
  ranked first (or in the top 3)? This is what matters in practice, since
  a wrongly-ranked #1 chunk can dominate a short context window.
- **MRR** - how far down the ranking the first genuinely relevant chunk
  falls, averaged across questions.
- **avg_latency_ms** - wall-clock retrieval time per question.

## Latest results (20 questions, 15 standard + 5 adversarial/short-keyword)

| Strategy | recall@1 | recall@3 | MRR | avg latency |
|---|---|---|---|---|
| naive semantic-only | 0.95 | 1.00 | 0.975 | ~9 ms |
| hybrid (BM25 + MMR + rerank) | **1.00** | 1.00 | **1.00** | ~165 ms |

(Regenerate with `python eval/run_eval.py`; see `results.json` for the raw output.)

**Honest read of this result:** at this small, cleanly-separated-topic
corpus scale, plain semantic search is already very strong - it's wrong
only once, on `api-free-quota` ("How many free requests per month are
included in the Free tier?"), where it ranks the document's generic
title/intro chunk above the specific pricing chunk. The hybrid pipeline
gets every question right in the top 1, at roughly 17x the latency (~165ms
vs ~9ms) because of the extra BM25 pass and, mainly, the cross-encoder
rerank step.

This is a genuine but modest gap, not a dramatic one - and that's the
honest result at this scale. The value of BM25 + reranking should widen on
larger, noisier, multi-topic corpora (more plausible near-duplicate
chunks to confuse a pure embedding lookup) and on short, keyword-style
queries (closer to how people actually type into a search box); the 5
adversarial cases in this dataset are a first step toward measuring that,
not a full-scale proof of it.

## Grounding gate

`rag_pipeline.MIN_RELEVANCE_SCORE` (-8.0) is the cross-encoder relevance
score below which `answer_question` skips the LLM call entirely and
returns a "nothing relevant found" message instead of risking a
hallucinated answer over irrelevant context. The threshold comes from
measuring the reranker's score distribution directly:

| Query | Chunk | Score |
|---|---|---|
| "What is the rate limit?" | (relevant: rate-limit paragraph, clean prose) | **+1.06** |
| "How many vacation days do employees get?" | (relevant: leave-policy paragraph, clean prose) | **+3.09** |
| "What is the collection time?" | (relevant: OCR'd scanned lab report) | **-3.09** |
| "Sodium level?" | (relevant: OCR'd scanned lab report) | **-6.00** |
| "Name of the patient?" | (relevant: OCR'd scanned lab report) | **-6.18** |
| "What is the rate limit?" | (irrelevant: battery test paragraph) | **-10.70** |
| "What is the capital of France?" | (irrelevant: pricing paragraph, or OCR'd lab report corpus) | **-11.0 to -11.19** |

Relevant pairs from clean prose score roughly +1 to +3, but relevant pairs
from noisy OCR'd/scanned documents (garbled characters, tabular/form
layout instead of prose) score much lower - down to -6.6 in testing - even
when the retrieved chunk is exactly right, because the reranker just isn't
well-calibrated for that kind of text. Clearly irrelevant pairs stay
around -10 to -11 regardless of document type or how the text was
extracted. -8.0 sits in the gap between the worst-case relevant OCR score
and the irrelevant baseline, so it accepts genuine matches from scanned
documents without meaningfully increasing the risk of admitting truly
irrelevant context (which still scores ~2.5+ points below it). This costs
nothing extra: the reranker score was already being computed to *order*
the results and had previously been discarded after sorting.

