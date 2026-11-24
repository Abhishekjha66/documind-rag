# 📚 DocuMind - RAG Document Q&A

Chat with your own PDFs, Word files and text notes. DocuMind uses **Retrieval-Augmented Generation (RAG)**: it finds the most relevant passages in your documents and has an **LLM** answer *only* from them, with inline citations.

**🔗 Live demo:** `https://YOUR-APP-NAME.streamlit.app`  <!-- replace after deploying -->

## Features

- Upload PDF, DOCX, TXT, MD (multiple files)
- Recursive chunking with overlap
- Semantic embeddings (`BAAI/bge-small-en-v1.5` via ONNX/fastembed, CPU only, no GPU needed)
- Cosine-similarity retrieval + **MMR re-ranking** for diverse results
- Grounded generation with `[1] [2]` source citations and "I couldn't find that" refusals (reduces hallucination)
- Provider-agnostic: Groq (free), OpenAI, OpenRouter, or local Ollama via any OpenAI-compatible API
- Streaming responses, source viewer with similarity scores, conversation memory
- Works without an API key (shows retrieved passages)
- Unit tests + GitHub Actions CI

## Architecture

```
Upload -> Loader -> Chunker -> Embedder -> Vector store
                                               |
Question -> Embedder -> top-k + MMR  <---------+
                           |
                  Prompt (context + rules) -> LLM -> Answer + citations
```

| Module | Role |
|---|---|
| `src/loader.py` | PDF/DOCX/TXT parsing |
| `src/chunker.py` | Recursive splitter with overlap |
| `src/embedder.py` | fastembed (semantic) with hashing fallback |
| `src/vectorstore.py` | In-memory cosine search + MMR |
| `src/rag.py` | Pipeline, prompt, citations |
| `src/llm.py` | OpenAI-compatible streaming client |
| `app.py` | Streamlit UI |

## Run locally

```bash
git clone https://github.com/YOUR-USERNAME/documind-rag.git
cd documind-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Get a free API key at [console.groq.com](https://console.groq.com), paste it in the sidebar (or set `LLM_API_KEY` in `.streamlit/secrets.toml`).

## Tests

```bash
pytest -q
```

## Deploy free on Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) -> **New app** -> pick the repo, branch `main`, file `app.py`.
3. In **Advanced settings -> Secrets** add: `LLM_API_KEY = "your_groq_key"`
4. Deploy and copy the live URL into this README and your resume.

## Tech stack

Python, Streamlit, fastembed (BGE embeddings), NumPy, OpenAI-compatible LLM API (Llama 3.3 on Groq), pypdf, python-docx, pytest, GitHub Actions.

## Ideas to extend

Persistent vector DB (FAISS / Chroma), hybrid BM25 + dense search, cross-encoder re-ranking, RAG evaluation with RAGAS, OCR for scanned PDFs.

## License

MIT
