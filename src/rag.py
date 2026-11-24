"""RAG pipeline: ingest -> embed -> retrieve -> grounded generation with citations."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from .chunker import Chunk, chunk_text
from .loader import load_document
from .vectorstore import VectorStore

SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions ONLY from the provided context.\n"
    "Rules:\n"
    "1. Use only the numbered context passages. Cite them inline like [1], [2].\n"
    "2. If the context does not contain the answer, reply exactly: "
    "\"I couldn't find that in the uploaded documents.\"\n"
    "3. Never invent facts. Be concise."
)


@dataclass
class Retrieval:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    text: str
    sources: list[Retrieval] = field(default_factory=list)


class RAGPipeline:
    def __init__(self, embedder, llm=None, chunk_size: int = 800, overlap: int = 120):
        self.embedder = embedder
        self.llm = llm
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.store = VectorStore()
        self.documents: set[str] = set()

    # ---------- ingestion ----------
    def ingest(self, name: str, data: bytes) -> int:
        if name in self.documents:
            return 0
        chunks: list[Chunk] = []
        for page, text in load_document(name, data):
            chunks += chunk_text(text, name, page, self.chunk_size, self.overlap, start_index=len(chunks))
        if not chunks:
            return 0
        self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
        self.documents.add(name)
        return len(chunks)

    def reset(self) -> None:
        self.store.clear()
        self.documents.clear()

    # ---------- retrieval ----------
    def retrieve(self, question: str, k: int = 4) -> list[Retrieval]:
        qv = self.embedder.embed([question])[0]
        return [Retrieval(c, s) for c, s in self.store.search(qv, k=k)]

    @staticmethod
    def build_messages(question: str, hits: list[Retrieval], history: list[dict] | None = None) -> list[dict]:
        context = "\n\n".join(
            f"[{i}] (source: {h.chunk.source}, page {h.chunk.page})\n{h.chunk.text}"
            for i, h in enumerate(hits, 1)
        )
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        msgs += (history or [])[-6:]
        msgs.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
        return msgs

    # ---------- generation ----------
    def answer_stream(self, question: str, k: int = 4, history: list[dict] | None = None
                      ) -> tuple[list[Retrieval], Iterator[str]]:
        hits = self.retrieve(question, k)
        if not hits:
            return hits, iter(["Please upload a document first."])
        if self.llm is None:  # no API key: extractive fallback so the demo still works
            body = "\n\n".join(f"**[{i}]** {h.chunk.text}" for i, h in enumerate(hits, 1))
            return hits, iter([f"_No LLM key set - showing the most relevant passages:_\n\n{body}"])
        return hits, self.llm.stream(self.build_messages(question, hits, history))

    def answer(self, question: str, k: int = 4) -> Answer:
        hits, stream = self.answer_stream(question, k)
        return Answer("".join(stream), hits)
