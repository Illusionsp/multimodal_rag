"""
app.py
------
Streamlit front-end for the Multimodal RAG system.

Run with:  streamlit run app.py
"""
import logging
from pathlib import Path

import streamlit as st

import config
from src.rag_pipeline import MultimodalRAGPipeline

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="ReportIQ — Financial & Business Report Assistant", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run downloads them, please wait)...")
def get_pipeline() -> MultimodalRAGPipeline:
    return MultimodalRAGPipeline()


def init_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of (role, content, sources)
    if "ingest_log" not in st.session_state:
        st.session_state.ingest_log = []


def seed_sample_document(pipeline: MultimodalRAGPipeline):
    """Auto-ingest the bundled demo report on first run so the app is
    immediately testable without the visitor needing their own PDF."""
    sample_path = Path(__file__).parent / "sample_documents" / "demo_quarterly_business_report.pdf"
    if not sample_path.exists():
        return
    if sample_path.name in pipeline.list_sources():
        return
    with st.spinner("Loading sample report (tables + charts) for a quick demo..."):
        try:
            pipeline.ingest(str(sample_path))
        except Exception as e:
            logging.getLogger(__name__).warning("Could not seed sample document: %s", e)


def render_source(source, idx: int):
    with st.expander(f"Source {idx + 1} — {source.type.upper()} · {source.source} · page {source.page_number}"):
        if source.type == "image" and source.image_path and Path(source.image_path).exists():
            st.image(source.image_path, use_column_width=True)
            st.caption(source.content)
        elif source.type == "table":
            st.markdown(source.content)
        else:
            st.write(source.content)
        if source.distance is not None:
            st.caption(f"similarity distance: {source.distance:.4f}")


def sidebar(pipeline: MultimodalRAGPipeline):
    st.sidebar.title("📊 ReportIQ")
    st.sidebar.caption("Ask questions across financial reports, tables, charts, and scanned pages — no manual re-typing of numbers into spreadsheets.")

    st.sidebar.subheader("1. Upload documents")
    st.sidebar.caption("No PDF handy? A sample quarterly report is preloaded below — just ask a question.")
    uploaded_files = st.sidebar.file_uploader(
        "PDF files (tables, charts, scanned pages all supported)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.sidebar.button("Ingest uploaded PDFs", type="primary"):
        for uf in uploaded_files:
            dest = config.UPLOAD_DIR / uf.name
            dest.write_bytes(uf.getbuffer())
            with st.spinner(f"Parsing {uf.name} (tables, charts, OCR fallback)..."):
                try:
                    stats = pipeline.ingest(str(dest))
                    st.session_state.ingest_log.append(stats)
                    if stats.skipped_duplicate:
                        st.sidebar.info(f"{uf.name}: identical file already indexed, skipped re-ingestion.")
                    else:
                        st.sidebar.success(
                            f"{uf.name}: {stats.text_chunks} text, {stats.table_chunks} tables, "
                            f"{stats.image_chunks} charts/images, {stats.ocr_recovered_chunks} OCR-recovered chunks"
                        )
                except Exception as e:
                    st.sidebar.error(f"Failed to ingest {uf.name}: {e}")

    st.sidebar.subheader("2. Search settings")
    top_k = st.sidebar.slider("Chunks to retrieve", min_value=1, max_value=15, value=config.DEFAULT_TOP_K)
    type_options = st.sidebar.multiselect(
        "Restrict to content types", ["text", "table", "image"], default=[]
    )
    sources = pipeline.list_sources()
    source_filter = st.sidebar.selectbox("Restrict to document", ["(all documents)"] + sources)
    source_filter = None if source_filter == "(all documents)" else source_filter

    st.sidebar.subheader("3. Knowledge base")
    st.sidebar.write(f"Documents indexed: {len(sources)}")
    if st.sidebar.button("⚠️ Reset knowledge base"):
        pipeline.reset()
        st.session_state.chat_history = []
        st.session_state.ingest_log = []
        st.sidebar.warning("Knowledge base cleared.")
        st.rerun()

    with st.sidebar.expander("Backend status"):
        st.write(f"Groq configured: {'✅' if config.GROQ_API_KEY else '❌ (set GROQ_API_KEY)'}")
        st.write(f"HF fallback configured: {'✅' if config.HF_API_TOKEN else '❌ (set HF_API_TOKEN)'}")
        st.write(f"OCR engine: {config.OCR_ENGINE}")
        st.write(f"Parsing strategy: {config.UNSTRUCTURED_STRATEGY}")
        st.write(f"Hybrid search (BM25 + dense): {'✅' if config.ENABLE_HYBRID_SEARCH else '❌'}")
        st.write(f"Cross-encoder reranking: {'✅' if config.ENABLE_RERANKING else '❌'}")

    return top_k, type_options or None, source_filter


def main():
    init_state()
    pipeline = get_pipeline()
    seed_sample_document(pipeline)
    top_k, type_filter, source_filter = sidebar(pipeline)

    st.title("Financial & Business Report Q&A")
    st.caption(
        "Ask questions across text, complex tables, charts, and scanned pages in your reports. "
        "Try: \"What was North America's Q3 revenue and YoY growth?\" or \"What does the regional "
        "revenue pie chart show?\" against the preloaded sample report."
    )

    for role, content, srcs in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)
            if srcs:
                for i, s in enumerate(srcs):
                    render_source(s, i)

    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.chat_history.append(("user", question, None))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks and generating an answer..."):
                result = pipeline.answer(
                    question, top_k=top_k, type_filter=type_filter, source_filter=source_filter
                )
            st.markdown(result.answer)
            for i, s in enumerate(result.sources):
                render_source(s, i)
        st.session_state.chat_history.append(("assistant", result.answer, result.sources))


if __name__ == "__main__":
    main()
