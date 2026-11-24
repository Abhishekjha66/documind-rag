import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunker import chunk_text
from src.embedder import HashingEmbedder
from src.loader import load_document
from src.rag import RAGPipeline
from src.vectorstore import VectorStore


class FakeLLM:
    def stream(self, messages):
        yield "Answer based on context [1]."


def test_chunker_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i}." for i in range(200))
    chunks = chunk_text(text, "a.txt", size=300, overlap=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 + 51 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", "a", size=10, overlap=10)


def test_loader_text_and_unsupported():
    assert load_document("n.txt", b"hello world") == [(1, "hello world")]
    with pytest.raises(ValueError):
        load_document("n.exe", b"x")


def test_vectorstore_ranking_and_mmr():
    emb = HashingEmbedder()
    chunks = chunk_text("cats purr softly", "a", 1) + chunk_text("stock markets fell today", "b", 1)
    store = VectorStore()
    store.add(chunks, emb.embed([c.text for c in chunks]))
    top = store.search(emb.embed(["why do cats purr"])[0], k=1, mmr_lambda=None)
    assert top[0][0].source == "a"
    assert len(store.search(emb.embed(["markets"])[0], k=2)) == 2


def test_pipeline_retrieves_correct_document():
    rag = RAGPipeline(HashingEmbedder(), llm=FakeLLM())
    rag.ingest("python.txt", b"Python is a programming language created by Guido van Rossum in 1991.")
    rag.ingest("cooking.txt", b"To make pasta, boil salted water and cook spaghetti for ten minutes.")
    ans = rag.answer("Who created Python?", k=1)
    assert ans.sources[0].chunk.source == "python.txt"
    assert "[1]" in ans.text


def test_pipeline_without_llm_returns_passages():
    rag = RAGPipeline(HashingEmbedder(), llm=None)
    rag.ingest("a.txt", b"Retrieval augmented generation grounds answers in documents.")
    assert "No LLM key" in rag.answer("what is retrieval augmented generation").text


def test_duplicate_ingest_is_ignored():
    rag = RAGPipeline(HashingEmbedder())
    assert rag.ingest("a.txt", b"some text here") > 0
    assert rag.ingest("a.txt", b"some text here") == 0
