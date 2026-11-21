"""Structure-aware chunking over canonical extracted text.

Rule set:
  1. Split on structure first: headings start a new section, blank lines
     separate paragraphs.
  2. Pack paragraphs greedily to a target size. A chunk never ends inside a
     paragraph, and never straddles two sections.
  3. Only a single oversized paragraph gets broken up, and it breaks at
     sentence ends, then whitespace, then (last resort, never needed for
     clean prose) hard character count.

Chunk spans are [start, end) offsets into the canonical text and stay
contiguous: canonical[start:end] is the chunk, byte-for-byte, including the
whitespace inside it. No fixed-character windows anywhere.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

DEFAULT_TARGET = 1200   # chars, ~300 words
DEFAULT_MIN = 150       # a tail chunk smaller than this merges back

# Lines that read as headings in residential transaction docs. Named list so
# corpora can extend it, not a pile of inline regexes.
_HEADING_PATTERNS = [
    re.compile(r"^(ARTICLE|SECTION|EXHIBIT|SCHEDULE)\s+[A-Z0-9]+[.:]?\s*\S*", re.I),
    re.compile(r"^(?:\d+\.)*\d+\s+[A-Z][A-Za-z][^\n]*$"),          # 1.2 Contingencies
    re.compile(r"^(?=[A-Z][A-Z ,;&/'\-]{4,})[A-Z][A-Z ,;&/'\-]{4,}:?$"),  # ALL CAPS line
]

_SENT_BREAK = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Block:
    start: int
    end: int
    is_heading: bool


@dataclass
class Chunk:
    start: int
    end: int
    page: int
    section: str | None

    @property
    def length(self) -> int:
        return self.end - self.start


def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 90:
        return False
    return any(p.match(line) for p in _HEADING_PATTERNS)


def parse_blocks(text: str) -> list[Block]:
    """Group text into heading blocks and paragraph blocks.

    A paragraph is a run of consecutive non-blank lines. Blank lines belong
    to no block; chunk spans still cover them because spans are contiguous.
    """
    blocks: list[Block] = []
    lines = text.splitlines(keepends=True)
    pos = 0
    para_start = None
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            if para_start is not None:
                blocks.append(Block(para_start, pos, False))
                para_start = None
            pos += len(raw)
            continue
        if is_heading(stripped):
            if para_start is not None:
                blocks.append(Block(para_start, pos, False))
                para_start = None
            blocks.append(Block(pos, pos + len(raw.rstrip("\n")), True))
            pos += len(raw)
            continue
        if para_start is None:
            para_start = pos
        pos += len(raw)
    if para_start is not None:
        blocks.append(Block(para_start, pos, False))
    return blocks


def _split_oversized(text: str, block: Block, target: int) -> list[tuple[int, int]]:
    """Break one oversized paragraph into pieces near <=target, covering it
    contiguously. Cuts at sentence ends, then whitespace, then hard."""
    s, e = block.start, block.end
    body = text[s:e]

    # sentence edges relative to body, contiguous: [0, t1), [t1, t2), ...
    # the break whitespace stays with the sentence it ends
    edges = sorted({0, len(body)} | {m.end() for m in _SENT_BREAK.finditer(body)})

    pieces: list[tuple[int, int]] = []
    for a, b in zip(edges, edges[1:]):
        while b - a > target:
            cut = a + target
            ws = max(body.rfind(" ", a + 1, cut), body.rfind("\n", a + 1, cut))
            if ws > a:
                cut = ws + 1                     # keep whole words when possible
            pieces.append((a, cut))
            a = cut
        if pieces and (pieces[-1][1] - pieces[-1][0]) + (b - a) <= target:
            pieces[-1] = (pieces[-1][0], b)
        else:
            pieces.append((a, b))
    return [(s + a, s + b) for a, b in pieces]


def chunk_text(text: str, *, target: int = DEFAULT_TARGET,
               min_size: int = DEFAULT_MIN,
               page_spans: list[tuple[int, int]] | None = None) -> list[Chunk]:
    """Chunk canonical text into contiguous, structure-aware spans.

    page_spans: per-page [start, end) offsets from ingest, used only to label
    each chunk with its starting page.
    """
    page_starts = [p[0] for p in page_spans] if page_spans else []

    def page_of(offset: int) -> int:
        if not page_spans:
            return 0
        return max(bisect.bisect_right(page_starts, offset) - 1, 0)

    # expand oversized paragraphs into sentence-granular units first, so the
    # packing pass below sees one uniform list of splittable blocks
    units: list[Block] = []
    for b in parse_blocks(text):
        if not b.is_heading and b.end - b.start > target:
            for ps, pe in _split_oversized(text, b, target):
                units.append(Block(ps, pe, False))
        else:
            units.append(b)

    chunks: list[Chunk] = []
    section: str | None = None
    cur: list[Block] = []

    def emit() -> None:
        if cur:
            chunks.append(Chunk(cur[0].start, cur[-1].end,
                                page_of(cur[0].start), section))
            cur.clear()

    for u in units:
        if u.is_heading:
            emit()
            section = text[u.start:u.end].strip()
            cur.append(u)
            continue
        if cur and (u.end - cur[0].start) > target:
            emit()
        cur.append(u)
    emit()

    # merge runt chunks back into the previous chunk of the same section
    merged: list[Chunk] = []
    for c in chunks:
        if (merged and c.length < min_size and c.section == merged[-1].section
                and c.start >= merged[-1].end):
            merged[-1] = Chunk(merged[-1].start, c.end, merged[-1].page,
                               merged[-1].section)
        else:
            merged.append(c)
    return merged
