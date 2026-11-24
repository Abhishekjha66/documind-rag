"""Streamlit UI for DocuMind - chat with your documents using RAG."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.embedder import get_embedder
from src.llm import PRESETS, LLMClient
from src.rag import RAGPipeline

st.set_page_config(page_title="DocuMind - RAG Document Q&A", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return get_embedder()


def secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("📚 DocuMind")
    st.caption("Retrieval-Augmented Generation over your own documents")

    st.subheader("LLM settings")
    provider = st.selectbox("Provider", list(PRESETS))
    base_url, default_model = PRESETS[provider]
    model = st.text_input("Model", default_model)
    api_key = st.text_input("API key", value=secret("LLM_API_KEY"), type="password",
                            help="Free key: console.groq.com. Leave empty to see retrieval only.")
    top_k = st.slider("Passages to retrieve (k)", 1, 10, 4)
    chunk_size = st.slider("Chunk size (chars)", 300, 2000, 800, 100)

    st.subheader("Documents")
    files = st.file_uploader("Upload PDF / DOCX / TXT / MD", type=["pdf", "docx", "txt", "md"],
                             accept_multiple_files=True)
    use_sample = st.button("Load sample documents")
    if st.button("Clear everything"):
        st.session_state.clear()
        st.rerun()

# ---------------- state ----------------
embedder = load_embedder()
if "rag" not in st.session_state or st.session_state.get("chunk_size") != chunk_size:
    st.session_state.rag = RAGPipeline(embedder, chunk_size=chunk_size, overlap=chunk_size // 7)
    st.session_state.chunk_size = chunk_size
    st.session_state.messages = []
rag: RAGPipeline = st.session_state.rag
rag.llm = LLMClient(api_key, base_url, model) if api_key or provider.startswith("Ollama") else None

with st.spinner("Indexing documents..."):
    for f in files or []:
        rag.ingest(f.name, f.getvalue())
    if use_sample:
        for p in sorted(Path("sample_docs").glob("*")):
            rag.ingest(p.name, p.read_bytes())

# ---------------- main ----------------
st.header("Ask questions about your documents")
c1, c2, c3 = st.columns(3)
c1.metric("Documents", len(rag.documents))
c2.metric("Chunks indexed", len(rag.store))
c3.metric("Embeddings", embedder.name.split("/")[-1])

if not rag.documents:
    st.info("Upload a document in the sidebar, or click **Load sample documents** to try it out.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            hits, stream = rag.answer_stream(question, k=top_k, history=history)
            answer = st.write_stream(stream)
        except Exception as e:  # bad key, rate limit, network...
            hits, answer = [], f"LLM call failed: `{e}`"
            st.error(answer)
        if hits:
            with st.expander(f"Sources ({len(hits)})"):
                for i, h in enumerate(hits, 1):
                    st.markdown(f"**[{i}] {h.chunk.source}** - page {h.chunk.page} - similarity `{h.score:.2f}`")
                    st.caption(h.chunk.text)
    st.session_state.messages.append({"role": "assistant", "content": answer})
