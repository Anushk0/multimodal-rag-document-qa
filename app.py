from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Iterable, List

import streamlit as st

# Bridge Streamlit Cloud's secrets manager into the environment before
# rag_pipeline reads API_KEY at import time. Locally, st.secrets simply has
# no secrets.toml and raises - that's fine, rag_pipeline's own load_dotenv()
# picks up .env instead. This keeps rag_pipeline itself UI-agnostic (it has
# no Streamlit dependency, since api.py also depends on it).
try:
    if "API_KEY" in st.secrets:
        os.environ.setdefault("API_KEY", st.secrets["API_KEY"])
except Exception:
    pass

import rag.rag_pipeline as rag_module  # noqa: E402
import storage  # noqa: E402
from agent.graph import agentic_stream_answer  # noqa: E402


APP_TITLE = "Agentic Document QA System"
MAX_HISTORY_TURNS = 10

# Light, theme-safe styling - relies on Streamlit's own CSS variables for
# colors so it doesn't fight the user's light/dark mode, and only tweaks
# spacing/rounding rather than overriding Streamlit's DOM structure.
_CUSTOM_CSS = """
<style>
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 0.25rem 0.9rem;
    margin-bottom: 0.35rem;
    background: var(--secondary-background-color);
}
.stExpander {
    border-radius: 10px;
    border: 1px solid rgba(128, 128, 128, 0.25);
}
.rag-badge {
    display: inline-block;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.1rem 0.6rem;
    border-radius: 999px;
    margin-bottom: 0.4rem;
}
.rag-badge-grounded { background: rgba(34, 197, 94, 0.15); color: #16a34a; }
.rag-badge-weak { background: rgba(234, 179, 8, 0.18); color: #b45309; }
.rag-badge-none { background: rgba(239, 68, 68, 0.15); color: #dc2626; }
.rag-stat-strip {
    display: flex;
    gap: 1.25rem;
    font-size: 0.85rem;
    opacity: 0.85;
    margin: -0.5rem 0 1rem 0;
}
</style>
"""


def save_uploaded_pdfs(files: Iterable[Any], upload_dir: Path) -> List[Path]:
    return [
        storage.save_uploaded_pdf_bytes(
            getattr(f, "name", "document.pdf"), bytes(f.getbuffer()), upload_dir
        )
        for f in files
    ]


def _confidence_badge(trace: dict | None) -> str:
    """Small HTML badge summarizing how well-grounded an answer was, derived
    from the same relevance score the grounding gate itself used - so the
    badge never claims more confidence than the gate actually had."""
    if not trace or trace.get("top_score") is None:
        if trace and trace.get("vision_triggered"):
            return '<span class="rag-badge rag-badge-grounded">🟢 Grounded (figure/diagram)</span>'
        return '<span class="rag-badge rag-badge-none">🔴 No relevant content found</span>'
    score = trace["top_score"]
    threshold = trace.get("grounding_threshold", rag_module.MIN_RELEVANCE_SCORE)
    if not trace.get("grounded", score >= threshold):
        return '<span class="rag-badge rag-badge-none">🔴 No relevant content found</span>'
    # "Weak" band: passed the gate but not by much - still an honest signal,
    # not a hard cutoff backed by further calibration.
    if score < threshold + 3:
        return f'<span class="rag-badge rag-badge-weak">🟠 Weak match (score {score})</span>'
    return f'<span class="rag-badge rag-badge-grounded">🟢 Grounded (score {score})</span>'


def _render_step_timeline(trace: dict | None) -> None:
    """The step-by-step reasoning log ("watch it think", after the fact) -
    the same events that, for a live answer, are also streamed in real
    time into an st.status() box (see the on_step wiring in main())."""
    steps = trace.get("steps") if trace else None
    if not steps:
        return
    st.caption("Reasoning steps:")
    for i, step in enumerate(steps, start=1):
        line = f"{i}. **{step['label']}**"
        if step.get("detail"):
            line += f" — {step['detail']}"
        st.markdown(line)


def _render_trace(trace: dict | None) -> None:
    """Expandable retrieval/timing diagnostics for one answer - the
    "tracing" panel: what was searched, what was found, how confident the
    match was, and how long each stage took."""
    if not trace:
        return
    with st.expander("🔍 How this was answered"):
        _render_step_timeline(trace)
        docs = trace.get("documents_searched") or []
        st.caption(f"Searched {len(docs)} document(s): {', '.join(docs) or 'none'}")

        cols = st.columns(4)
        cols[0].metric("Chunks indexed", trace.get("total_chunks_indexed", "-"))
        cols[1].metric("Semantic hits", trace.get("semantic_hits", "-"))
        cols[2].metric("BM25 hits", trace.get("bm25_hits", "-"))
        top_score = trace.get("top_score")
        cols[3].metric("Top rerank score", top_score if top_score is not None else "-")

        timing_bits = []
        if trace.get("retrieval_ms") is not None:
            timing_bits.append(f"retrieval {trace['retrieval_ms']} ms")
        if trace.get("total_ms") is not None:
            timing_bits.append(f"total {trace['total_ms']} ms")
        if trace.get("cache_hit"):
            timing_bits.append("served from cache")
        if trace.get("vision_triggered"):
            timing_bits.append("vision/figure analysis used")
        if timing_bits:
            st.caption(" · ".join(timing_bits))

        sources = trace.get("sources_used")
        if sources:
            st.caption("Chunks used to answer (highest relevance first):")
            st.table(
                [{"source": s["source"], "relevance score": s["score"]} for s in sources]
            )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    st.title(f"📄 {APP_TITLE}")
    st.caption(
        "Ask questions grounded in your own PDFs - answers are only generated when "
        "genuinely relevant content is found, never guessed."
    )

    # Each browser session gets its own id, so its uploads, vector index, and
    # answers never mix with another concurrent user's (see rag_pipeline's
    # per-session Chroma collections, and storage.py's per-session upload dirs).
    session_id = st.session_state.setdefault("session_id", uuid.uuid4().hex)
    upload_dir = storage.session_upload_dir(session_id)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("traces", [])

    with st.sidebar:
        st.subheader("📤 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            processed = st.session_state.setdefault("processed_uploads", set())
            new_files = [f for f in uploaded_files if f.file_id not in processed]
            if new_files:
                saved = save_uploaded_pdfs(new_files, upload_dir)
                st.success(f"Saved {len(saved)} file(s).")
                with st.spinner("Indexing new document(s) (OCR + embeddings if needed)..."):
                    new_chunks, failed = rag_module.ingest_uploaded_pdfs(saved, session_id=session_id)
                if new_chunks:
                    st.info(f"Indexed {new_chunks} new chunk(s).")
                if failed:
                    st.warning(
                        "Could not extract any text from: "
                        + ", ".join(failed)
                        + ". These files won't be searchable until this is fixed - "
                        "check the app logs for the OCR error."
                    )
                processed.update(f.file_id for f in new_files)

        pdf_paths = storage.list_uploaded_pdfs(upload_dir)
        st.divider()
        st.subheader("📚 Documents")

        selected_paths: List[Path] = []
        if pdf_paths:
            total_chunks = rag_module.get_indexed_chunk_count(session_id)
            st.markdown(
                f'<div class="rag-stat-strip">'
                f'<span>📄 {len(pdf_paths)} document(s)</span>'
                f'<span>🧩 {total_chunks} chunk(s) indexed</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

            name_to_path = {p.name: p for p in pdf_paths}
            selected_names = st.multiselect(
                "Search within",
                options=list(name_to_path.keys()),
                default=list(name_to_path.keys()),
                help="Only these documents are used to answer your question.",
            )
            selected_paths = [name_to_path[name] for name in selected_names]

            col1, col2 = st.columns(2)
            if col1.button("🗑️ Clear index", use_container_width=True):
                rag_module.reset_index(session_id=session_id)
                st.success("Index cleared - it rebuilds on your next question.")
            if col2.button("🧹 Clear chat", use_container_width=True):
                st.session_state["history"] = []
                st.session_state["traces"] = []
                st.rerun()
        else:
            st.caption("No PDFs uploaded yet.")

        st.divider()
        agentic_mode = st.toggle(
            "🧠 Agentic mode (beta)",
            value=False,
            help=(
                "Plans multi-part questions into sub-questions, retries weak "
                "retrievals, and self-checks its answer before returning it. "
                "Uses 2-4x more API calls per question than normal mode, "
                "since it makes several model calls (planning, generation, "
                "self-check) instead of one - off by default to conserve "
                "your daily quota."
            ),
        )
        if agentic_mode:
            st.caption("⚠️ Uses more of your daily API quota per question than normal mode.")

    if not st.session_state["history"]:
        st.info(
            "💡 **Tip:** ask about specific facts, numbers, names, or sections in your "
            "document(s) rather than broad requests like \"summarize this\" - the more "
            "specific the question, the more reliably it can be matched to the right "
            "part of your document."
        )

    for i, (prior_question, prior_answer) in enumerate(st.session_state["history"]):
        with st.chat_message("user"):
            st.write(prior_question)
        with st.chat_message("assistant"):
            trace = st.session_state["traces"][i] if i < len(st.session_state["traces"]) else None
            if trace:
                st.markdown(_confidence_badge(trace), unsafe_allow_html=True)
            st.write(prior_answer)
            _render_trace(trace)

    question = st.chat_input(
        "Ask a question about the selected documents...",
        disabled=not selected_paths,
    )

    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            trace: dict = {}
            # Created (not entered via `with`) so on_step can write into it
            # live, interleaved with retrieval happening synchronously
            # inside stream_answer, while the actual answer text below
            # renders in the normal chat-message flow, not nested inside
            # the collapsible status box.
            status = st.status("🧠 Thinking...", expanded=True, state="running")

            def _on_step(event: dict) -> None:
                line = f"**{event['label']}**"
                if event.get("detail"):
                    line += f" — {event['detail']}"
                status.write(line)

            answer_stream = (
                agentic_stream_answer if agentic_mode else rag_module.stream_answer
            )(
                question,
                selected_paths,
                session_id=session_id,
                history=st.session_state["history"],
                trace=trace,
                on_step=_on_step,
            )
            answer = st.write_stream(answer_stream)
            status.update(label="✅ Done thinking", state="complete", expanded=False)
            # The badge can only be computed once the stream has fully run
            # (trace is populated by _prepare_prompt as a side effect of the
            # generator being consumed) - so, unlike historical turns above,
            # it renders below the answer here rather than above it. On the
            # next rerun this turn becomes historical and reflows to match.
            st.markdown(_confidence_badge(trace), unsafe_allow_html=True)
            _render_trace(trace)
        st.session_state["history"].append((question, answer))
        st.session_state["traces"].append(trace)
        st.session_state["history"] = st.session_state["history"][-MAX_HISTORY_TURNS:]
        st.session_state["traces"] = st.session_state["traces"][-MAX_HISTORY_TURNS:]


if __name__ == "__main__":
    main()
