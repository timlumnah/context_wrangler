"""Table extraction: coordinate-aware reflow of aligned grids.

A PDF stores text in content-stream order, not visual order. Many table
writers emit cell by cell, column by column: all labels, then all
amounts. pypdf's default extraction hands that back as stream order, and
the row adjacency, the fact that $410 belongs to Title Insurance, is gone
from the bytes. No chunk then holds label and figure together, and an
answer that cites the right document can still quote a number from the
wrong row of its table.

With --tables, ingest re-extracts each page in pypdf's layout mode (it
uses the text coordinates to compose visual lines), looks for a grid:
at least MIN_ROWS consecutive lines that carry the same number of cells,
each cell column-aligned within COL_TOL characters, cells separated by
whitespace runs of 2+ (a single space is word space, inside a cell) and
reflows what it finds into pipe-joined rows:

    "    Title Insurance        $410.00"     layout line
    -> "Title Insurance | $410.00"           canonical row

The marker is part of the canonical text, not decoration: the byte-verify
contracts are unchanged, citations quote these bytes.

Pages with no grid keep the plain-mode text byte-for-byte, so enabling
tables cannot drift citations on the prose that makes up almost every
document here. Reflow is a targeted repair, not a corpus-wide rewrite.

OCR'd pages never come through here: tesseract at psm 6 already reads
tables line by line (measured, see core/ocr.py), and its output has no
column geometry left to align.
"""

from __future__ import annotations

import re

MIN_ROWS = 3
MIN_COLS = 2
COL_TOL = 2          # column start drift allowed, in layout-mode characters
SEP = " | "          # the row marker that lands in canonical text


def _cells(line: str) -> list[tuple[int, str]]:
    """Split one line into cells at whitespace runs of 2+. A cell may hold
    single spaces inside it ("Title Insurance"). Returns [(start, text)]."""
    cells: list[list] = []
    cur = None
    for m in re.finditer(r"\S+", line):
        if cur is not None and m.start() - cur[1] <= 1:
            cur[1] = m.end()
            cur[2] = f"{cur[2]} {m.group()}"
        else:
            if cur is not None:
                cells.append(cur)
            cur = [m.start(), m.end(), m.group()]
    if cur is not None:
        cells.append(cur)
    return [(c[0], c[2]) for c in cells]


def _columns(line: str) -> list[int] | None:
    """Column starts if the line reads as >=MIN_COLS aligned cells,
    else None."""
    cells = _cells(line.rstrip())
    if len(cells) < MIN_COLS:
        return None
    return [start for start, _text in cells]


def _region(lines: list[str], i: int) -> list[int] | None:
    """If lines starting at i form a grid run (>= MIN_ROWS lines, each
    with the same cell count and columns aligned to line i within
    COL_TOL), return the region length. Else None."""
    cols = _columns(lines[i])
    if cols is None:
        return None
    n = len(lines)
    j = i + 1
    while j < n:
        c = _columns(lines[j])
        if (c is None or len(c) != len(cols)
                or any(abs(a - b) > COL_TOL for a, b in zip(c, cols))):
            break
        j += 1
    length = j - i
    return length if length >= MIN_ROWS else None


def reflow(layout_text: str) -> str | None:
    """Reflow one layout-mode page. Returns None when the page holds no
    grid: the caller keeps the plain text then, byte-identical to a
    tables-off ingest."""
    lines = layout_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    i = 0
    found = False
    while i < len(lines):
        length = _region(lines, i)
        if length:
            found = True
            for row in range(i, i + length):
                out.append(SEP.join(text for _start, text in
                                    _cells(lines[row].rstrip())))
            i += length
            continue
        ln = lines[i].strip()
        out.append(" ".join(ln.split()) if ln else "")
        i += 1
    return "\n".join(out) if found else None


def page_text(page, *, tables: bool = False) -> str:
    """The extraction entry point ingest uses. tables=False is the v0.4
    path byte-for-byte. With tables on, layout mode is consulted and only
    grid pages get their text replaced; a layout-mode failure falls back
    to plain, loudly only if it changes nothing (it can't)."""
    plain = page.extract_text(extraction_mode="plain") or ""
    if not tables:
        return plain
    try:
        layout = page.extract_text(extraction_mode="layout") or ""
    except Exception:                       # noqa: BLE001 - layout is a bonus
        return plain
    reflowed = reflow(layout)
    return reflowed if reflowed is not None else plain
