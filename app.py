"""
RAG Application — Streamlit Frontend
Provides: document upload/indexing, chat, admin panel, and document tools.
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import config, logger, compute_file_hash, human_size, UPLOAD_DIR
from src.document_loader import load_document, load_documents_batch
from src.chunker import ChunkConfig, chunk_documents
from src.embeddings import get_embedder
from src.vector_store import VectorStore
from src.retriever import Retriever, RetrieverConfig
from src.llm import (
    get_llm,
    stream_rag_answer,
    summarise_text,
    generate_quiz,
    generate_insights,
    compare_documents,
)
from src.chat import ChatSession, ChatMessage

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
[data-testid="stSidebar"] { background: #0f1117; }
.main .block-container { padding-top: 1.5rem; max-width: 1200px; }

/* Chat bubbles */
.user-bubble {
    background: #1e3a5f; color: #e8f4fd; border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1rem; margin: 0.5rem 0; max-width: 85%;
    margin-left: auto; font-size: 0.95rem;
}
.assistant-bubble {
    background: #1a1d2e; color: #e2e8f0; border-radius: 18px 18px 18px 4px;
    padding: 0.75rem 1rem; margin: 0.5rem 0; max-width: 90%;
    border-left: 3px solid #3b82f6; font-size: 0.95rem;
}
.source-card {
    background: #111827; border: 1px solid #374151; border-radius: 8px;
    padding: 0.6rem 0.9rem; margin: 0.3rem 0; font-size: 0.8rem;
    color: #9ca3af;
}
.source-card strong { color: #60a5fa; }
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155; border-radius: 10px; padding: 1rem;
    text-align: center;
}
.stProgress > div > div { background: linear-gradient(90deg, #3b82f6, #8b5cf6); }
.tag {
    display: inline-block; background: #1e3a5f; color: #60a5fa;
    border-radius: 4px; padding: 0.1rem 0.5rem; font-size: 0.75rem;
    margin: 0.1rem; font-family: monospace;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────

def _init_session_state():
    defaults = {
        "chat_session": ChatSession(),
        "vector_store": None,
        "retriever": None,
        "indexing_log": [],
        "api_key_verified": False,
        "active_tab": "chat",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_vector_store(embedding_model: str) -> VectorStore:
    return VectorStore(embedding_model=embedding_model)


def get_retriever_cached(embedding_model: str) -> Retriever:
    vs = get_vector_store(embedding_model)
    return Retriever(vs)


def _check_api_key() -> bool:
    key = st.session_state.get("google_api_key_input", "") or config.GOOGLE_API_KEY
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        config.GOOGLE_API_KEY = key
        return True
    return False


def _format_source_cards(sources: list[dict]) -> str:
    html_parts = []
    for src in sources[:5]:
        name = src.get("source", "?")
        page = src.get("page", "?")
        score = src.get("score", 0)
        section = src.get("section", "")
        sec_str = f" · {section}" if section else ""
        html_parts.append(
            f'<div class="source-card">'
            f'📄 <strong>{name}</strong>{sec_str} — '
            f'Page <strong>{page}</strong> '
            f'<span style="color:#4ade80">▲ {score:.2%}</span>'
            f'</div>'
        )
    return "".join(html_parts)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 DocMind RAG")
        st.caption("Production-grade document intelligence")
        st.divider()

        # API Key
        with st.expander("🔑 API Configuration", expanded=not _check_api_key()):
            api_key = st.text_input(
                "Google API Key",
                value=config.GOOGLE_API_KEY,
                type="password",
                key="google_api_key_input",
                help="Get a key at https://aistudio.google.com/app/apikey",
            )
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
                config.GOOGLE_API_KEY = api_key

        st.divider()

        # Model settings
        st.markdown("### ⚙️ Model Settings")
        embedding_model = st.selectbox(
            "Embedding Model",
            config.AVAILABLE_EMBEDDING_MODELS,
            index=0,
            key="embedding_model",
        )
        llm_model = st.selectbox(
            "LLM Model",
            config.AVAILABLE_LLM_MODELS,
            index=0,
            key="llm_model",
        )
        temperature = st.slider(
            "Temperature", 0.0, 1.0, config.TEMPERATURE, 0.05, key="temperature"
        )

        st.divider()

        # Chunking
        st.markdown("### ✂️ Chunking")
        chunk_strategy = st.radio(
            "Strategy",
            ["recursive", "semantic"],
            horizontal=True,
            key="chunk_strategy",
        )
        chunk_size = st.slider("Chunk Size", 256, 4000, config.DEFAULT_CHUNK_SIZE, 64, key="chunk_size")
        chunk_overlap = st.slider("Chunk Overlap", 0, 800, config.DEFAULT_CHUNK_OVERLAP, 32, key="chunk_overlap")

        st.divider()

        # Retrieval
        st.markdown("### 🔍 Retrieval")
        retrieval_mode = st.radio(
            "Mode", ["hybrid", "dense", "sparse"], horizontal=True, key="retrieval_mode"
        )
        top_k = st.slider("Top-K Results", 1, 20, config.DEFAULT_TOP_K, 1, key="top_k")
        sim_threshold = st.slider(
            "Similarity Threshold", 0.0, 1.0, config.DEFAULT_SIMILARITY_THRESHOLD, 0.05,
            key="sim_threshold"
        )
        use_rerank = st.toggle("Re-rank Results", True, key="use_rerank")
        use_mmr = st.toggle("MMR Diversity", False, key="use_mmr")

        st.divider()

        # Quick stats
        try:
            vs = get_vector_store(embedding_model)
            stats = vs.get_collection_stats()
            col1, col2 = st.columns(2)
            col1.metric("📄 Docs", stats["total_documents"])
            col2.metric("🧩 Chunks", stats["total_chunks"])
            st.caption(f"💾 {stats['storage_size']}")
        except Exception:
            st.caption("No documents indexed yet")


# ── Upload & Indexing Tab ─────────────────────────────────────────────────────

def render_upload_tab():
    st.markdown("## 📤 Upload & Index Documents")
    st.markdown(
        "Upload **PDF, DOCX, TXT, Markdown, or HTML** files. "
        "Files are automatically deduplicated and indexed incrementally."
    )

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt", "md", "html", "htm"],
        accept_multiple_files=True,
        help=f"Max {config.MAX_UPLOAD_SIZE_MB} MB per file",
    )

    if not uploaded_files:
        st.info("👆 Upload one or more documents to get started.", icon="ℹ️")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Files Selected", len(uploaded_files))
    col2.metric(
        "Total Size",
        human_size(sum(f.size for f in uploaded_files)),
    )
    col3.metric(
        "Est. Pages",
        "~" + str(sum(max(1, f.size // 3000) for f in uploaded_files)),
    )

    st.divider()

    if st.button("🚀 Index Documents", type="primary", use_container_width=True):
        embedding_model = st.session_state.get("embedding_model", config.EMBEDDING_MODEL)
        chunk_cfg = ChunkConfig(
            strategy=st.session_state.get("chunk_strategy", "recursive"),
            chunk_size=st.session_state.get("chunk_size", config.DEFAULT_CHUNK_SIZE),
            chunk_overlap=st.session_state.get("chunk_overlap", config.DEFAULT_CHUNK_OVERLAP),
        )

        vs = get_vector_store(embedding_model)
        indexed_hashes = vs.get_indexed_hashes()

        progress_bar = st.progress(0, text="Preparing…")
        status_area = st.empty()
        log_lines: list[str] = []

        def log(msg: str, icon: str = ""):
            full = f"{icon} {msg}".strip()
            log_lines.append(full)
            status_area.markdown(
                "\n".join(f"- {l}" for l in log_lines[-10:])
            )

        saved_paths: list[Path] = []
        duplicate_count = 0

        # Save uploaded files to disk
        for uf in uploaded_files:
            raw = uf.read()
            file_hash = compute_file_hash(raw)
            if file_hash in indexed_hashes:
                log(f"Skipped (duplicate): {uf.name}", "⏭️")
                duplicate_count += 1
                continue
            dest = UPLOAD_DIR / uf.name
            dest.write_bytes(raw)
            saved_paths.append(dest)
            log(f"Saved: {uf.name}", "✅")

        if not saved_paths:
            st.warning(f"All {duplicate_count} file(s) already indexed.")
            progress_bar.progress(1.0, text="Done")
            return

        total_chunks_added = 0
        total_files = len(saved_paths)

        for file_idx, (path, result) in enumerate(load_documents_batch(saved_paths)):
            file_progress = (file_idx) / total_files
            progress_bar.progress(file_progress, text=f"Loading {path.name}…")

            if isinstance(result, Exception):
                log(f"Error loading {path.name}: {result}", "❌")
                continue

            log(f"Loaded {path.name}: {len(result)} pages/sections", "📄")

            # Determine embedder for semantic chunking
            embedder = None
            if chunk_cfg.strategy == "semantic":
                embedder = get_embedder(embedding_model)

            chunks = chunk_documents(result, cfg=chunk_cfg, embedder=embedder)
            log(f"Chunked {path.name}: {len(chunks)} chunks", "✂️")

            progress_bar.progress(
                (file_idx + 0.5) / total_files,
                text=f"Indexing {path.name}…",
            )

            def embed_progress(done, total):
                inner = (file_idx + 0.5 + (done / total) * 0.5) / total_files
                progress_bar.progress(min(inner, 0.99), text=f"Embedding {path.name}…")

            added = vs.add_documents(chunks, progress_callback=embed_progress)
            total_chunks_added += added
            log(f"Indexed {path.name}: {added} chunks stored", "🗄️")

        # Invalidate BM25 cache
        retriever = get_retriever_cached(embedding_model)
        retriever.invalidate_cache()

        progress_bar.progress(1.0, text="✅ Indexing complete!")
        st.success(
            f"✅ Indexed {total_files - duplicate_count} document(s) → "
            f"{total_chunks_added} chunks added. "
            + (f"{duplicate_count} duplicate(s) skipped." if duplicate_count else "")
        )
        st.balloons()


# ── Chat Tab ──────────────────────────────────────────────────────────────────

def render_chat_tab():
    st.markdown("## 💬 Ask Your Documents")

    embedding_model = st.session_state.get("embedding_model", config.EMBEDDING_MODEL)
    vs = get_vector_store(embedding_model)

    if vs.count() == 0:
        st.warning("📭 No documents indexed yet. Go to **Upload** to add documents first.", icon="⚠️")
        return

    session: ChatSession = st.session_state.chat_session

    # ── Chat controls ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            session.clear()
            st.rerun()
    with col2:
        export_fmt = st.selectbox("Export format", ["markdown", "json", "csv"], label_visibility="collapsed")
    with col3:
        if st.button("📥 Export Chat", use_container_width=True):
            if export_fmt == "json":
                data = session.export_json()
                mime = "application/json"
                ext = "json"
            elif export_fmt == "csv":
                data = session.export_csv()
                mime = "text/csv"
                ext = "csv"
            else:
                data = session.export_markdown()
                mime = "text/markdown"
                ext = "md"
            st.download_button(
                "⬇️ Download",
                data=data,
                file_name=f"chat_export.{ext}",
                mime=mime,
            )

    st.divider()

    # ── Render history ────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in session.messages:
            if msg.role == "user":
                st.markdown(
                    f'<div class="user-bubble">🧑 {msg.content}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="assistant-bubble">{msg.content}</div>',
                    unsafe_allow_html=True,
                )
                if msg.sources:
                    with st.expander(f"📚 {len(msg.sources)} source(s) cited", expanded=False):
                        for src in msg.sources:
                            col_a, col_b, col_c = st.columns([3, 1, 1])
                            col_a.markdown(f"**{src.get('source', '?')}**")
                            col_b.markdown(f"Page `{src.get('page', '?')}`")
                            col_c.markdown(f"Score `{src.get('score', 0):.2%}`")
                            if src.get("content"):
                                st.caption(src["content"][:300] + "…")
                            st.divider()

    # ── Input ─────────────────────────────────────────────────────────────────
    query = st.chat_input("Ask a question about your documents…")

    if query:
        if not _check_api_key():
            st.error("🔑 Set your Google API key in the sidebar first.")
            return

        session.add_user(query)
        st.markdown(
            f'<div class="user-bubble">🧑 {query}</div>',
            unsafe_allow_html=True,
        )

        retriever = get_retriever_cached(embedding_model)
        ret_cfg = RetrieverConfig(
            mode=st.session_state.get("retrieval_mode", "hybrid"),
            top_k=st.session_state.get("top_k", config.DEFAULT_TOP_K),
            similarity_threshold=st.session_state.get("sim_threshold", config.DEFAULT_SIMILARITY_THRESHOLD),
            rerank=st.session_state.get("use_rerank", True),
            mmr_diversity=st.session_state.get("use_mmr", False),
        )

        with st.spinner("🔍 Retrieving relevant passages…"):
            results = retriever.retrieve(query, cfg=ret_cfg)

        if not results:
            no_ans = (
                "I could not find relevant information in the indexed documents "
                "for this query. Try rephrasing or lowering the similarity threshold."
            )
            session.add_assistant(no_ans)
            st.markdown(f'<div class="assistant-bubble">{no_ans}</div>', unsafe_allow_html=True)
            st.rerun()
            return

        # Stream answer
        with st.empty():
            full_response = ""
            typing_placeholder = st.empty()
            typing_placeholder.markdown(
                '<div class="assistant-bubble">⌛ Thinking…</div>',
                unsafe_allow_html=True,
            )
            full_text = ""
            for chunk in stream_rag_answer(
                query=query,
                results=results,
                conversation_history=session.get_history_for_llm()[:-1],
                model_name=st.session_state.get("llm_model", config.DEFAULT_LLM_MODEL),
                temperature=st.session_state.get("temperature", config.TEMPERATURE),
            ):
                full_text += chunk
                typing_placeholder.markdown(
                    f'<div class="assistant-bubble">{full_text}▌</div>',
                    unsafe_allow_html=True,
                )

            typing_placeholder.markdown(
                f'<div class="assistant-bubble">{full_text}</div>',
                unsafe_allow_html=True,
            )

        sources = [r.to_dict() for r in results]
        session.add_assistant(
            full_text,
            sources=sources,
            model=st.session_state.get("llm_model", config.DEFAULT_LLM_MODEL),
            retrieval_mode=ret_cfg.mode,
        )

        # Show sources
        with st.expander(f"📚 {len(results)} source(s) retrieved", expanded=True):
            for r in results:
                col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
                col_a.markdown(f"**{r.source}**")
                col_b.markdown(f"Page `{r.page}`")
                col_c.markdown(f"Score `{r.score:.2%}`")
                col_d.markdown(f"`{r.retrieval_method}`")
                st.caption(r.document.page_content[:400] + "…")
                st.divider()


# ── Document Tools Tab ────────────────────────────────────────────────────────

def render_tools_tab():
    st.markdown("## 🛠️ Document Tools")

    embedding_model = st.session_state.get("embedding_model", config.EMBEDDING_MODEL)
    vs = get_vector_store(embedding_model)
    doc_list = vs.get_document_list()

    if not doc_list:
        st.warning("No documents indexed yet.")
        return

    if not _check_api_key():
        st.warning("🔑 Set your Google API key in the sidebar to use LLM tools.")
        return

    doc_names = [d["source"] for d in doc_list]

    tool_choice = st.radio(
        "Select Tool",
        ["📝 Summarise Document", "📄 Summarise Pages", "❓ Generate Quiz",
         "💡 Generate Insights", "🔄 Compare Documents", "🔎 Search Only"],
        horizontal=True,
    )

    st.divider()

    llm_model = st.session_state.get("llm_model", config.DEFAULT_LLM_MODEL)

    def _get_doc_text(source_name: str, page_range: tuple | None = None) -> str:
        """Pull raw text for a document from the vector store."""
        try:
            raw = vs._get_store().get(
                where={"source": source_name},
                include=["documents", "metadatas"],
            )
            pairs = list(zip(raw.get("documents", []), raw.get("metadatas", [])))
            if page_range:
                pairs = [
                    (d, m) for d, m in pairs
                    if page_range[0] <= int(m.get("page", 0) or 0) <= page_range[1]
                ]
            pairs.sort(key=lambda x: int(x[1].get("page", 0) or 0))
            return "\n\n".join(d for d, _ in pairs)
        except Exception as exc:
            return f"Error retrieving text: {exc}"

    # ── Summarise Document ────────────────────────────────────────────────────
    if tool_choice == "📝 Summarise Document":
        selected = st.selectbox("Select document", doc_names)
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Summarising…"):
                text = _get_doc_text(selected)
                if not text:
                    st.error("Could not retrieve document text.")
                    return
                result_placeholder = st.empty()
                full_text = ""
                for chunk in summarise_text(text, model_name=llm_model, streaming=True):
                    full_text += chunk
                    result_placeholder.markdown(full_text + "▌")
                result_placeholder.markdown(full_text)
                st.download_button(
                    "📥 Download Summary",
                    full_text,
                    file_name=f"summary_{selected}.md",
                    mime="text/markdown",
                )

    # ── Summarise Pages ───────────────────────────────────────────────────────
    elif tool_choice == "📄 Summarise Pages":
        selected = st.selectbox("Select document", doc_names)
        doc_info = next((d for d in doc_list if d["source"] == selected), {})
        total_pages = doc_info.get("total_pages", 1) or 1
        try:
            total_pages = int(total_pages)
        except (ValueError, TypeError):
            total_pages = 50

        page_range = st.slider(
            "Page range", 1, max(total_pages, 1), (1, min(10, total_pages))
        )
        if st.button("Summarise Selected Pages", type="primary"):
            with st.spinner(f"Summarising pages {page_range[0]}–{page_range[1]}…"):
                text = _get_doc_text(selected, page_range)
                if not text:
                    st.error("No content found for selected pages.")
                    return
                result = summarise_text(text, model_name=llm_model)
                st.markdown(result)

    # ── Generate Quiz ─────────────────────────────────────────────────────────
    elif tool_choice == "❓ Generate Quiz":
        selected = st.selectbox("Select document", doc_names)
        n_questions = st.slider("Number of questions", 3, 15, 5)
        if st.button("Generate Quiz", type="primary"):
            with st.spinner("Generating quiz questions…"):
                text = _get_doc_text(selected)
                result = generate_quiz(text, n=n_questions, model_name=llm_model)
                st.markdown(result)
                st.download_button(
                    "📥 Download Quiz",
                    result,
                    file_name=f"quiz_{selected}.md",
                    mime="text/markdown",
                )

    # ── Generate Insights ─────────────────────────────────────────────────────
    elif tool_choice == "💡 Generate Insights":
        selected = st.selectbox("Select document", doc_names)
        if st.button("Analyse Document", type="primary"):
            with st.spinner("Analysing…"):
                text = _get_doc_text(selected)
                result = generate_insights(text, model_name=llm_model)
                st.markdown(result)

    # ── Compare Documents ─────────────────────────────────────────────────────
    elif tool_choice == "🔄 Compare Documents":
        if len(doc_names) < 2:
            st.warning("Index at least 2 documents to use comparison.")
            return
        col1, col2 = st.columns(2)
        doc_a = col1.selectbox("Document A", doc_names, key="cmp_a")
        doc_b = col2.selectbox("Document B", [d for d in doc_names if d != doc_a], key="cmp_b")
        if st.button("Compare Documents", type="primary"):
            with st.spinner("Comparing…"):
                text_a = _get_doc_text(doc_a)
                text_b = _get_doc_text(doc_b)
                result = compare_documents(text_a, doc_a, text_b, doc_b, model_name=llm_model)
                st.markdown(result)

    # ── Search Only ───────────────────────────────────────────────────────────
    elif tool_choice == "🔎 Search Only":
        query = st.text_input("Semantic search query")
        k = st.slider("Results", 1, 20, 5, key="search_k")
        threshold = st.slider("Threshold", 0.0, 1.0, 0.2, 0.05, key="search_thresh")
        source_filter = st.selectbox("Filter by document", ["All"] + doc_names, key="search_filter")

        if st.button("Search", type="primary") and query:
            filter_dict = None if source_filter == "All" else {"source": source_filter}
            results = vs.similarity_search(
                query, k=k, filter_dict=filter_dict, score_threshold=threshold
            )
            if not results:
                st.info("No results above threshold.")
            for doc, score in results:
                with st.expander(
                    f"📄 {doc.metadata.get('source', '?')} — "
                    f"Page {doc.metadata.get('page', '?')} — "
                    f"Score {score:.2%}"
                ):
                    st.markdown(doc.page_content)
                    st.json(doc.metadata, expanded=False)


# ── Admin Panel Tab ───────────────────────────────────────────────────────────

def render_admin_tab():
    st.markdown("## ⚙️ Administration Panel")

    embedding_model = st.session_state.get("embedding_model", config.EMBEDDING_MODEL)
    vs = get_vector_store(embedding_model)
    stats = vs.get_collection_stats()

    # ── Overview metrics ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Documents", stats["total_documents"])
    c2.metric("🧩 Chunks", stats["total_chunks"])
    c3.metric("💾 Storage", stats["storage_size"])
    c4.metric("🤖 Model", stats["embedding_model"].split("/")[-1])

    st.divider()

    # ── Document table ────────────────────────────────────────────────────────
    st.markdown("### 📚 Indexed Documents")
    doc_list = stats.get("documents", [])

    if not doc_list:
        st.info("No documents indexed.")
    else:
        df = pd.DataFrame(doc_list)
        df["char_count"] = df["char_count"].apply(lambda x: human_size(int(x or 0)))
        df = df.rename(columns={
            "source": "Filename",
            "doc_type": "Type",
            "chunk_count": "Chunks",
            "total_pages": "Pages",
            "upload_date": "Uploaded",
            "char_count": "Size",
        })
        df["Uploaded"] = pd.to_datetime(df["Uploaded"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Delete a document
        st.markdown("#### 🗑️ Delete Document")
        doc_to_delete = st.selectbox(
            "Select document to remove",
            [d["source"] for d in doc_list],
            key="delete_doc_select",
        )
        col_del1, col_del2 = st.columns([2, 1])
        with col_del1:
            st.warning(f"This will remove all indexed chunks for **{doc_to_delete}**.")
        with col_del2:
            if st.button("🗑️ Delete", type="secondary", use_container_width=True):
                deleted = vs.delete_document(doc_to_delete)
                # Also remove from uploads folder
                upload_path = UPLOAD_DIR / doc_to_delete
                upload_path.unlink(missing_ok=True)
                retriever = get_retriever_cached(embedding_model)
                retriever.invalidate_cache()
                st.success(f"Deleted {deleted} chunks for '{doc_to_delete}'")
                st.rerun()

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    if doc_list:
        st.markdown("### 📊 Statistics")
        col_a, col_b = st.columns(2)

        with col_a:
            df_plot = pd.DataFrame(doc_list)
            if "doc_type" in df_plot.columns:
                type_counts = df_plot["doc_type"].value_counts().reset_index()
                type_counts.columns = ["Type", "Count"]
                fig = px.pie(
                    type_counts, names="Type", values="Count",
                    title="Documents by Type",
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if "chunk_count" in df_plot.columns and "source" in df_plot.columns:
                df_chunks = df_plot.nlargest(10, "chunk_count")
                fig2 = px.bar(
                    df_chunks, x="source", y="chunk_count",
                    title="Chunks per Document (Top 10)",
                    color="chunk_count",
                    color_continuous_scale="Blues",
                )
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_tickangle=-35,
                )
                st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Danger zone ───────────────────────────────────────────────────────────
    st.markdown("### ⚠️ Danger Zone")
    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        st.error(
            "**Rebuild Vector Database** — drops all indexed data. "
            "You will need to re-upload and re-index all documents."
        )
    with col_r2:
        if st.button("🔄 Rebuild DB", type="secondary", use_container_width=True):
            vs.rebuild_collection()
            get_vector_store.clear()
            st.session_state.chat_session.clear()
            st.success("Vector database rebuilt. Please re-index your documents.")
            st.rerun()


# ── Main layout ───────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    st.markdown(
        "<h1 style='margin-bottom:0'>🧠 DocMind RAG</h1>"
        "<p style='color:#6b7280;margin-top:0'>Intelligent document retrieval & generation</p>",
        unsafe_allow_html=True,
    )

    tab_upload, tab_chat, tab_tools, tab_admin = st.tabs([
        "📤 Upload & Index",
        "💬 Chat",
        "🛠️ Tools",
        "⚙️ Admin",
    ])

    with tab_upload:
        render_upload_tab()

    with tab_chat:
        render_chat_tab()

    with tab_tools:
        render_tools_tab()

    with tab_admin:
        render_admin_tab()


if __name__ == "__main__":
    main()
