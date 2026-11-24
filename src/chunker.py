"""Recursive text chunking with overlap."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    index: int


_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    out: list[str] = []
    buf = ""
    for part in parts:
        piece = part if not buf else buf + sep + part
        if len(piece) <= size:
            buf = piece
            continue
        if buf:
            out.append(buf)
        if len(part) > size:
            out.extend(_split(part, size, rest))
            buf = ""
        else:
            buf = part
    if buf:
        out.append(buf)
    return out


def chunk_text(text: str, source: str, page: int = 1, size: int = 800, overlap: int = 120,
               start_index: int = 0) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    pieces = [p.strip() for p in _split(text, size, _SEPARATORS) if p.strip()]
    chunks: list[Chunk] = []
    for i, p in enumerate(pieces):
        if i > 0 and overlap:
            tail = pieces[i - 1][-overlap:]
            p = tail + " " + p
        chunks.append(Chunk(text=p, source=source, page=page, index=start_index + i))
    return chunks
