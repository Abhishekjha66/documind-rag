"""Embedding backends.

FastEmbedder  - real semantic embeddings (BAAI/bge-small-en-v1.5, ONNX, CPU only).
HashingEmbedder - dependency-free lexical fallback, used for tests / offline mode.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np


class HashingEmbedder:
    name = "hashing-fallback"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in self._tokens(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class FastEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def get_embedder(prefer_semantic: bool = True):
    if prefer_semantic:
        try:
            return FastEmbedder()
        except Exception:  # model download / import failed -> degrade gracefully
            pass
    return HashingEmbedder()
