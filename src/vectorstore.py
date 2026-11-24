"""Tiny in-memory vector store with cosine search and MMR re-ranking."""
from __future__ import annotations

import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self.chunks.extend(chunks)
        self._matrix = embeddings if self._matrix is None else np.vstack([self._matrix, embeddings])

    def clear(self) -> None:
        self.chunks, self._matrix = [], None

    def search(self, query_vec: np.ndarray, k: int = 4, mmr_lambda: float | None = 0.7,
               fetch_k: int = 20) -> list[tuple[Chunk, float]]:
        """Top-k chunks by cosine similarity. If mmr_lambda is set, diversify with MMR."""
        if self._matrix is None or not len(self.chunks):
            return []
        q = query_vec.reshape(-1)
        scores = self._matrix @ q
        k = min(k, len(self.chunks))
        if mmr_lambda is None:
            idx = np.argsort(-scores)[:k]
            return [(self.chunks[i], float(scores[i])) for i in idx]

        cand = list(np.argsort(-scores)[: max(fetch_k, k)])
        selected: list[int] = []
        while cand and len(selected) < k:
            if not selected:
                best = cand[0]
            else:
                sel_mat = self._matrix[selected]
                best, best_val = cand[0], -1e9
                for c in cand:
                    redundancy = float(np.max(sel_mat @ self._matrix[c]))
                    val = mmr_lambda * float(scores[c]) - (1 - mmr_lambda) * redundancy
                    if val > best_val:
                        best, best_val = c, val
            selected.append(best)
            cand.remove(best)
        return [(self.chunks[i], float(scores[i])) for i in selected]
