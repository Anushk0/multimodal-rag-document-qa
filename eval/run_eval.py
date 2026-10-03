"""Retrieval evaluation harness.

Runs the ground-truth question set in qa_dataset.json against two retrieval
strategies:

  naive_semantic_only    - plain vector similarity_search(k=5), the kind of
                            retrieval most "chat with your PDF" tutorials stop at.
  hybrid_bm25_mmr_rerank - this project's actual pipeline: MMR semantic search
                            + BM25 + cross-encoder reranking.

For each question we check whether a chunk from the expected source document
containing at least one expected keyword appears in the top-k results
(recall@k), and how highly it's ranked (MRR). This only exercises retrieval -
no LLM calls are made, so it costs nothing to run and is safe to run in CI on
every push.

Usage:
    python eval/run_eval.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag.rag_pipeline as rag_module  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DOCS_DIR = EVAL_DIR / "sample_docs"
DATASET_PATH = EVAL_DIR / "qa_dataset.json"
RESULTS_PATH = EVAL_DIR / "results.json"

# This corpus is intentionally small (a dozen or so chunks) so it stays fast
# and dependency-free for CI. At that scale recall@5 is trivially perfect for
# any strategy - every chunk from the right document fits in the top 5. What
# actually distinguishes retrieval quality here is whether the single most
# relevant chunk is ranked first, so we report recall@1/@3 and MRR instead.
EVAL_KS = (1, 3)

# Regression thresholds for the pipeline's actual (hybrid) strategy - used as
# a CI gate below. If a retrieval change drops quality below these, the
# build fails instead of silently shipping a worse retriever.
MIN_HYBRID_RECALL_AT_1 = 0.85
MIN_HYBRID_MRR = 0.85


def _is_relevant(doc, expected_source: str, expected_keywords: list[str]) -> bool:
    if doc.metadata.get("source") != expected_source:
        return False
    text = doc.page_content.lower()
    return any(kw.lower() in text for kw in expected_keywords)


def _recall_and_rank(ranked_docs, expected_source, expected_keywords, k):
    for i, doc in enumerate(ranked_docs[:k]):
        if _is_relevant(doc, expected_source, expected_keywords):
            return 1, i + 1
    return 0, None


def _evaluate(strategy_name, retrieve_fn, dataset, ks=EVAL_KS):
    recalls_by_k = {k: [] for k in ks}
    reciprocal_ranks, latencies = [], []
    max_k = max(ks)

    for item in dataset:
        start = time.perf_counter()
        ranked = retrieve_fn(item["question"])
        latencies.append(time.perf_counter() - start)

        for k in ks:
            hit, _ = _recall_and_rank(
                ranked, item["expected_source"], item["expected_keywords"], k
            )
            recalls_by_k[k].append(hit)

        _, rank = _recall_and_rank(
            ranked, item["expected_source"], item["expected_keywords"], max_k
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    result = {"strategy": strategy_name}
    for k in ks:
        result[f"recall@{k}"] = round(statistics.mean(recalls_by_k[k]), 3)
    result["mrr"] = round(statistics.mean(reciprocal_ranks), 3)
    result["avg_latency_ms"] = round(statistics.mean(latencies) * 1000, 1)
    result["n_questions"] = len(dataset)
    return result


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text())
    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(
            "No sample PDFs found. Run `python eval/build_sample_docs.py` first."
        )

    # Use a dedicated index so this never touches the app's real chroma_db/.
    rag_module.BASE_DB_PATH = str(EVAL_DIR / "eval_chroma_db")

    vectorstore = rag_module.get_vectorstore()
    rag_module.ingest_documents(pdf_paths, vectorstore)
    all_docs = rag_module._get_all_indexed_documents(vectorstore)
    bm25 = rag_module.BM25Okapi([d.page_content.split() for d in all_docs])

    def naive_semantic_only(question, k=max(EVAL_KS)):
        return vectorstore.similarity_search(question, k=k)

    def hybrid_bm25_mmr_rerank(question):
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 15, "fetch_k": 30, "lambda_mult": 0.5},
        )
        semantic_docs = retriever.invoke(question)
        bm25_docs = rag_module.bm25_search(question, all_docs, bm25, top_k=5)
        combined = semantic_docs[:8] + bm25_docs[:5]
        unique = list({d.page_content: d for d in combined}.values())
        scored = rag_module.rerank_documents(question, unique)
        return [doc for doc, _score in scored]

    results = [
        _evaluate("naive_semantic_only", naive_semantic_only, dataset),
        _evaluate("hybrid_bm25_mmr_rerank", hybrid_bm25_mmr_rerank, dataset),
    ]

    print(json.dumps(results, indent=2))
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {RESULTS_PATH}")

    hybrid = next(r for r in results if r["strategy"] == "hybrid_bm25_mmr_rerank")
    if hybrid["recall@1"] < MIN_HYBRID_RECALL_AT_1 or hybrid["mrr"] < MIN_HYBRID_MRR:
        raise SystemExit(
            f"\nRetrieval regression: hybrid recall@1={hybrid['recall@1']} "
            f"(min {MIN_HYBRID_RECALL_AT_1}), mrr={hybrid['mrr']} (min {MIN_HYBRID_MRR})"
        )


if __name__ == "__main__":
    main()
