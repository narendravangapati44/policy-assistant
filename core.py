"""
Shared configuration and helpers for the Policy Assistant.

Both ingest.py and rag.py import from here, so the embedding model, vector
database, chunking, and document loading live in one place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from anthropic import Anthropic

load_dotenv()

# --- Paths & configuration -------------------------------------------------
ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "documents"          # drop your PDFs / .txt / .md here
DB_PATH = str(ROOT / "qdrant_db")      # the on-disk vector database
COLLECTION = "policies"
MODEL = "claude-haiku-4-5-20251001"    # the LLM used to generate answers

# Chunking (characters). ~800 chars ≈ a couple of paragraphs, with overlap
# so meaning isn't lost at chunk boundaries.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Singletons (created once, reused) -------------------------------------
_embedder = None
_qdrant = None


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding()   # downloads a small local model on first use
    return _embedder


def embed(texts):
    """Turn a list of strings into a list of vectors."""
    return list(get_embedder().embed(texts))


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(path=DB_PATH)
    return _qdrant


def get_claude() -> Anthropic:
    return Anthropic()


# --- Document loading ------------------------------------------------------
def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def load_documents():
    """Yield (source_name, full_text) for each supported file in documents/."""
    if not DOCS_DIR.exists():
        return
    for path in sorted(DOCS_DIR.iterdir()):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = _read_pdf(path)
        elif suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8")
        else:
            continue
        if text.strip():
            yield path.name, text


# --- Chunking --------------------------------------------------------------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into chunks, respecting paragraphs, capped at `size` chars.

    Paragraphs are packed together until they'd exceed `size`. A single
    paragraph longer than `size` is hard-split with `overlap` between pieces.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
                current = ""
            if len(para) <= size:
                current = para
            else:
                step = max(size - overlap, 1)
                for i in range(0, len(para), step):
                    chunks.append(para[i:i + size])
    if current:
        chunks.append(current)
    return chunks
