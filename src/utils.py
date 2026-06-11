"""
Shared utilities, configuration management, and logging setup.
"""

import os
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
VECTOR_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", BASE_DIR / "vectorstore"))
LOG_DIR = BASE_DIR / "logs"

for _dir in (UPLOAD_DIR, VECTOR_DIR, LOG_DIR, BASE_DIR / "data"):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logger.remove()
logger.add(
    LOG_DIR / "rag_app.log",
    rotation="10 MB",
    retention="7 days",
    level=LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}",
)
logger.add(lambda msg: None, level="TRACE")  # suppress stdout noise in Streamlit


# ── Config ────────────────────────────────────────────────────────────────────
class Config:
    """Central configuration pulled from environment / .env."""

    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHROMA_PERSIST_DIR: str = str(VECTOR_DIR)
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", 1000))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 200))

    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", 5))
    DEFAULT_SIMILARITY_THRESHOLD: float = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", 0.3))

    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.0-flash")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", 8192))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.1))

    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", 500))

    SUPPORTED_EXTENSIONS: tuple = (".pdf", ".docx", ".txt", ".md", ".html", ".htm")

    AVAILABLE_EMBEDDING_MODELS: list = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    ]

    AVAILABLE_LLM_MODELS: list = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]


config = Config()


# ── File helpers ──────────────────────────────────────────────────────────────

def compute_file_hash(content: bytes) -> str:
    """SHA-256 hash of raw bytes — used for duplicate detection."""
    return hashlib.sha256(content).hexdigest()


def safe_filename(name: str) -> str:
    """Sanitise a filename: strip unsafe characters, truncate."""
    name = re.sub(r'[^\w\s\-.]', '_', name)
    name = re.sub(r'\s+', '_', name.strip())
    return name[:200]


def human_size(num_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def truncate(text: str, max_chars: int = 300) -> str:
    return text[:max_chars] + "…" if len(text) > max_chars else text


def flatten(nested: list[list[Any]]) -> list[Any]:
    return [item for sublist in nested for item in sublist]
