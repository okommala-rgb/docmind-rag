"""
Document loading: PDF (PyMuPDF), DOCX, TXT, Markdown, HTML.
Returns a list of LangChain Document objects with rich metadata.
"""

from __future__ import annotations

import io
import re
import time
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
import docx
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from src.utils import config, compute_file_hash, now_iso, logger


# ── Individual parsers ────────────────────────────────────────────────────────

def _load_pdf(path: Path, file_hash: str) -> list[Document]:
    """Extract text page-by-page from a PDF using PyMuPDF."""
    docs: list[Document] = []
    try:
        pdf = fitz.open(str(path))
        total_pages = len(pdf)
        for page_num in range(total_pages):
            page = pdf[page_num]
            text = page.get_text("text").strip()
            if not text:
                # Attempt OCR-friendly extraction for scanned pages
                text = page.get_text("blocks")
                text = " ".join(b[4] for b in text if isinstance(b[4], str)).strip()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "file_path": str(path),
                        "file_hash": file_hash,
                        "doc_type": "pdf",
                        "page": page_num + 1,
                        "total_pages": total_pages,
                        "upload_date": now_iso(),
                        "char_count": len(text),
                    },
                ))
        pdf.close()
        logger.info(f"PDF loaded: {path.name} | {total_pages} pages | {len(docs)} non-empty")
    except Exception as exc:
        logger.error(f"PDF load failed for {path.name}: {exc}")
        raise
    return docs


def _load_docx(path: Path, file_hash: str) -> list[Document]:
    """Extract text from a DOCX file, preserving heading structure."""
    docs: list[Document] = []
    try:
        document = docx.Document(str(path))
        sections: list[tuple[str, str]] = []  # (heading, body)
        current_heading = "Introduction"
        current_body: list[str] = []

        for para in document.paragraphs:
            style = para.style.name.lower()
            text = para.text.strip()
            if not text:
                continue
            if "heading" in style:
                if current_body:
                    sections.append((current_heading, "\n".join(current_body)))
                    current_body = []
                current_heading = text
            else:
                current_body.append(text)

        if current_body:
            sections.append((current_heading, "\n".join(current_body)))

        for idx, (heading, body) in enumerate(sections):
            docs.append(Document(
                page_content=f"[{heading}]\n{body}",
                metadata={
                    "source": path.name,
                    "file_path": str(path),
                    "file_hash": file_hash,
                    "doc_type": "docx",
                    "page": idx + 1,
                    "total_pages": len(sections),
                    "section_heading": heading,
                    "upload_date": now_iso(),
                    "char_count": len(body),
                },
            ))
        logger.info(f"DOCX loaded: {path.name} | {len(sections)} sections")
    except Exception as exc:
        logger.error(f"DOCX load failed for {path.name}: {exc}")
        raise
    return docs


def _load_txt(path: Path, file_hash: str, doc_type: str = "txt") -> list[Document]:
    """Load plain-text, markdown, or any text file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return []
        # Split into ~1000-char logical chunks at paragraph boundaries
        paragraphs = re.split(r"\n{2,}", text)
        docs: list[Document] = []
        chunk: list[str] = []
        chunk_idx = 1
        length = 0
        for para in paragraphs:
            chunk.append(para)
            length += len(para)
            if length >= 2000:
                docs.append(Document(
                    page_content="\n\n".join(chunk),
                    metadata={
                        "source": path.name,
                        "file_path": str(path),
                        "file_hash": file_hash,
                        "doc_type": doc_type,
                        "page": chunk_idx,
                        "total_pages": None,
                        "upload_date": now_iso(),
                        "char_count": length,
                    },
                ))
                chunk = []
                length = 0
                chunk_idx += 1
        if chunk:
            docs.append(Document(
                page_content="\n\n".join(chunk),
                metadata={
                    "source": path.name,
                    "file_path": str(path),
                    "file_hash": file_hash,
                    "doc_type": doc_type,
                    "page": chunk_idx,
                    "total_pages": chunk_idx,
                    "upload_date": now_iso(),
                    "char_count": length,
                },
            ))
        # back-fill total_pages
        for doc in docs:
            doc.metadata["total_pages"] = chunk_idx
        logger.info(f"TXT/MD loaded: {path.name} | {len(docs)} segments")
        return docs
    except Exception as exc:
        logger.error(f"TXT load failed for {path.name}: {exc}")
        raise


def _load_html(path: Path, file_hash: str) -> list[Document]:
    """Parse HTML, strip tags, extract clean text."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        tmp_doc = Document(
            page_content=text,
            metadata={
                "source": path.name,
                "file_path": str(path),
                "file_hash": file_hash,
                "doc_type": "html",
                "page": 1,
                "total_pages": 1,
                "upload_date": now_iso(),
                "char_count": len(text),
            },
        )
        # Reuse TXT chunking logic on the extracted text
        tmp_path = path.parent / (path.stem + "_extracted.txt")
        tmp_path.write_text(text, encoding="utf-8")
        docs = _load_txt(tmp_path, file_hash, doc_type="html")
        for doc in docs:
            doc.metadata["source"] = path.name
        tmp_path.unlink(missing_ok=True)
        logger.info(f"HTML loaded: {path.name}")
        return docs
    except Exception as exc:
        logger.error(f"HTML load failed for {path.name}: {exc}")
        raise


# ── Public interface ──────────────────────────────────────────────────────────

def load_document(path: Path, raw_bytes: bytes | None = None) -> list[Document]:
    """
    Load a single document from a Path (already saved to disk).
    Returns a list of LangChain Documents, one per page/section.
    """
    suffix = path.suffix.lower()
    if suffix not in config.SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    content_bytes = raw_bytes if raw_bytes else path.read_bytes()
    file_hash = compute_file_hash(content_bytes)

    if suffix == ".pdf":
        return _load_pdf(path, file_hash)
    elif suffix == ".docx":
        return _load_docx(path, file_hash)
    elif suffix in (".txt", ".md"):
        doc_type = "markdown" if suffix == ".md" else "txt"
        return _load_txt(path, file_hash, doc_type=doc_type)
    elif suffix in (".html", ".htm"):
        return _load_html(path, file_hash)
    else:
        raise ValueError(f"No loader registered for {suffix}")


def load_documents_batch(
    paths: list[Path],
    progress_callback=None,
) -> Generator[tuple[Path, list[Document] | Exception], None, None]:
    """
    Yield (path, documents_or_exception) for each file in paths.
    progress_callback(current, total, filename) is called per file.
    """
    total = len(paths)
    for idx, path in enumerate(paths):
        if progress_callback:
            progress_callback(idx, total, path.name)
        try:
            docs = load_document(path)
            yield path, docs
        except Exception as exc:
            logger.warning(f"Failed to load {path.name}: {exc}")
            yield path, exc
        time.sleep(0)  # yield control in tight loops
