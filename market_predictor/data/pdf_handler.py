"""
data/pdf_handler.py — PDF download and text extraction for S&P Global Platts reports.

Handles two sources:
  1. In-memory PDF bytes (from API response)
  2. PDF bytes fetched from a URL

Uses pdfplumber (preferred — better layout handling) with pypdf fallback.

Public API:
    extract_text_from_bytes(pdf_bytes)  -> str
    fetch_and_extract(url, headers)     -> str
"""
from __future__ import annotations

import io
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Extract all text from a PDF given as raw bytes.

    Tries pdfplumber first (better layout/table handling), falls back to pypdf.
    Returns empty string if both fail or bytes are not a valid PDF.
    """
    if not pdf_bytes or pdf_bytes[:4] != b"%PDF":
        return ""

    # Try pdfplumber first
    text = _extract_pdfplumber(pdf_bytes)
    if text:
        return text

    # Fallback: pypdf
    return _extract_pypdf(pdf_bytes)


def _extract_pdfplumber(pdf_bytes: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        return ""

    pages: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                    if text.strip():
                        pages.append(text.strip())
                except Exception:
                    continue
    except Exception:
        return ""

    return "\n".join(pages).strip()


def _extract_pypdf(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    pages: list[str] = []
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text.strip())
            except Exception:
                continue
    except Exception:
        return ""

    return "\n".join(pages).strip()


# ── URL fetch + extract ───────────────────────────────────────────────────────

def fetch_and_extract(url: str, headers: Optional[dict] = None,
                      timeout: int = 45) -> str:
    """
    Download a PDF from `url` and extract its text.

    Args:
        url:     Direct URL to a PDF file.
        headers: Optional HTTP headers (e.g. Authorization).
        timeout: Request timeout in seconds.

    Returns:
        Extracted text, or empty string on failure.
    """
    if not url:
        return ""
    try:
        import requests
        resp = requests.get(url, headers=headers or {}, timeout=timeout,
                            allow_redirects=True)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "").lower()
        if "pdf" not in ct and not url.lower().endswith(".pdf"):
            return ""
        return extract_text_from_bytes(resp.content)
    except Exception as exc:
        print(f"[pdf_handler] fetch failed for {url[:80]}: {exc}")
        return ""
