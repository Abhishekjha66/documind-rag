"""Document loading: turn uploaded files into (page, text) pairs."""
from __future__ import annotations

import io
from pathlib import Path

SUPPORTED = {".pdf", ".txt", ".md", ".docx"}


def load_document(name: str, data: bytes) -> list[tuple[int, str]]:
    """Return a list of (page_number, text). Non-paginated formats use page 1."""
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED)}")

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]
        return [(n, t) for n, t in pages if t]

    if ext == ".docx":
        from docx import Document

        doc = Document(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [(1, text)] if text.strip() else []

    text = data.decode("utf-8", errors="ignore").strip()
    return [(1, text)] if text else []
