from __future__ import annotations

import re
from pathlib import Path

from .schema import TextChunk


PAGE_PATTERNS = [
    re.compile(r"^\s*(?:<!--\s*)?(?:page|页码|第)\s*[:：]?\s*(\d+)\s*(?:页)?\s*(?:-->)?\s*$", re.I),
    re.compile(r"^\s*[-_*]{0,3}\s*(?:第\s*)?(\d+)\s*页\s*[-_*]{0,3}\s*$"),
    re.compile(r"^\s*#{1,6}\s*(?:page|第)\s*[:：]?\s*(\d+)\s*(?:页)?\s*$", re.I),
]
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def load_markdown(path: str | Path) -> str:
    md_path = Path(path)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown input not found: {md_path}")
    return md_path.read_text(encoding="utf-8")


def _page_hint(line: str) -> str | None:
    for pattern in PAGE_PATTERNS:
        match = pattern.match(line)
        if match:
            return match.group(1)
    return None


def _split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current = ""
    for para in re.split(r"(\n\s*\n)", text):
        if len(current) + len(para) > max_chars and current.strip():
            parts.append(current.strip())
            current = para
        else:
            current += para
    if current.strip():
        parts.append(current.strip())

    final: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final.append(part)
            continue
        for i in range(0, len(part), max_chars):
            final.append(part[i : i + max_chars].strip())
    return [item for item in final if item]


def split_markdown_chunks(
    markdown_text: str,
    min_chars: int = 1500,
    max_chars: int = 3000,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    title_stack: list[str] = []
    page = ""
    buffer: list[str] = []
    buffer_title: list[str] = []
    buffer_page = ""

    def flush(force: bool = False) -> None:
        nonlocal buffer, buffer_title, buffer_page
        text = "\n".join(buffer).strip()
        if not text:
            buffer = []
            return
        if not force and len(text) < min_chars:
            return
        for part in _split_long_text(text, max_chars):
            idx = len(chunks) + 1
            chunks.append(
                TextChunk(
                    chunk_id=f"chunk_{idx:04d}",
                    page_hint=buffer_page or page or f"chunk_{idx:04d}",
                    title_path=buffer_title[:],
                    text=part,
                )
            )
        buffer = []
        buffer_title = []
        buffer_page = ""

    for line in markdown_text.splitlines():
        page_match = _page_hint(line)
        if page_match:
            page = page_match
            if len("\n".join(buffer)) >= min_chars:
                flush(force=True)

        heading = HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            title_stack = title_stack[: level - 1] + [title]
            if len("\n".join(buffer)) >= min_chars:
                flush(force=True)

        if not buffer:
            buffer_title = title_stack[:]
            buffer_page = page
        buffer.append(line)

        if len("\n".join(buffer)) >= max_chars:
            flush(force=True)

    flush(force=True)
    return chunks
