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
IMAGE_LINE_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")
URL_LINE_RE = re.compile(r"https?://cdn-mineru\.openxlab\.org\.cn/\S+")


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


def _clean_line(line: str) -> str | None:
    # 删除 MinerU 生成的图片 URL 行；图片下方的图注保留。
    if IMAGE_LINE_RE.match(line):
        return None
    line = URL_LINE_RE.sub("", line)
    return line.rstrip()


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
        # 极端长段落按句号/分号优先切，最后再硬切。
        sentence_parts = re.split(r"(?<=[。；;])", part)
        buf = ""
        for sent in sentence_parts:
            if len(buf) + len(sent) > max_chars and buf.strip():
                final.append(buf.strip())
                buf = sent
            else:
                buf += sent
        if buf.strip():
            final.append(buf.strip())
    return [item for item in final if item]


def split_markdown_chunks(
    markdown_text: str,
    min_chars: int = 1200,
    max_chars: int = 2600,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    title_stack: list[str] = []
    page = ""
    buffer: list[str] = []
    buffer_title: list[str] = []
    buffer_page = ""

    def buffer_text() -> str:
        return "\n".join(buffer).strip()

    def flush(force: bool = False) -> None:
        nonlocal buffer, buffer_title, buffer_page
        text = buffer_text()
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

    for raw_line in markdown_text.splitlines():
        cleaned = _clean_line(raw_line)
        if cleaned is None:
            continue
        line = cleaned

        page_match = _page_hint(line)
        if page_match:
            page = page_match
            if len(buffer_text()) >= min_chars:
                flush(force=True)

        heading = HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            # 一级/二级标题一定断开，避免“北风之神级”和“G级/629型”混到同一 chunk。
            if buffer and (level <= 2 or len(buffer_text()) >= min_chars):
                flush(force=True)
            title_stack = title_stack[: level - 1] + [title]

        if not buffer:
            buffer_title = title_stack[:]
            buffer_page = page
        buffer.append(line)

        if len(buffer_text()) >= max_chars:
            flush(force=True)

    flush(force=True)
    return chunks
