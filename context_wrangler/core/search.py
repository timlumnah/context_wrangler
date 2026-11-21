"""BM25 keyword search over the FTS5 mirror, with cited spans.

Query handling: split the raw query on whitespace, tokenize each term the
same way FTS5 unicode61 does (runs of alphanumerics; "$12,500" becomes the
tokens 12 and 500). Multi-token terms become FTS5 phrases so adjacency
survives: "12,500" only matches "12,500", not a 12 and a 500 miles apart.
Terms combine with AND; --or gives the boolean OR.

Results carry [start, end) spans into the canonical text file plus a
snippet. The span is the citation; the snippet is decoration.
"""

from __future__ import annotations

import re
import sqlite3

from context_wrangler.core.index import read_canonical

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

_MARK_OPEN = "["    # snippet highlighting is decoration, never the quote
_MARK_CLOSE = "]"


def _tokens(seg: str) -> list[str]:
    return _TOKEN.findall(seg.lower())


def build_match(query: str, *, use_or: bool = False) -> str | None:
    """Translate a raw query into an FTS5 MATCH expression."""
    exprs = []
    for seg in query.split():
        toks = _tokens(seg)
        if not toks:
            continue
        exprs.append(f'"{" ".join(toks)}"' if len(toks) > 1 else toks[0])
    if not exprs:
        return None
    joiner = " OR " if use_or else " AND "
    return joiner.join(exprs)


def search(conn: sqlite3.Connection, index_dir: str, query: str, *,
           k: int = 8, use_or: bool = False) -> list[dict]:
    match = build_match(query, use_or=use_or)
    if match is None:
        return []
    rows = conn.execute(
        """
        SELECT chunks_fts.rowid              AS chunk_id,
               chunks.doc_id, docs.filename, chunks.page, chunks.section,
               chunks.start, chunks.end,
               bm25(chunks_fts, 1.0, 0.25, 0.0) AS score,
               snippet(chunks_fts, 0, ?, ?, ' ... ', 16) AS snip
        FROM chunks_fts
        JOIN chunks  ON chunks.chunk_id = chunks_fts.rowid
        JOIN docs    ON docs.doc_id = chunks.doc_id
        WHERE chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (_MARK_OPEN, _MARK_CLOSE, match, k)).fetchall()

    results = []
    for r in rows:
        canonical = read_canonical(index_dir, r["doc_id"])
        span_text = canonical[r["start"]:r["end"]]
        results.append({
            "doc_id": r["doc_id"],
            "filename": r["filename"],
            "chunk_id": r["chunk_id"],
            "page": r["page"] + 1,               # 1-based for humans
            "section": r["section"],
            "span": [r["start"], r["end"]],
            "score": r["score"],
            "snippet": r["snip"],
            # the verifiable quote: exactly canonical[start:end]
            "text": span_text,
        })
    return results


def chunk_record(conn: sqlite3.Connection, index_dir: str,
                 chunk_id: int) -> dict | None:
    """Build a citation record for one chunk by id (no FTS involvement).
    Snippet falls back to a plain window; the verifiable quote is `text`."""
    from context_wrangler.core.index import read_canonical
    r = conn.execute(
        """SELECT chunks.doc_id, docs.filename, chunks.page, chunks.section,
                  chunks.start, chunks.end
           FROM chunks JOIN docs ON docs.doc_id = chunks.doc_id
           WHERE chunks.chunk_id = ?""", (chunk_id,)).fetchone()
    if r is None:
        return None
    canonical = read_canonical(index_dir, r["doc_id"])
    body = canonical[r["start"]:r["end"]]
    return {
        "doc_id": r["doc_id"],
        "filename": r["filename"],
        "chunk_id": chunk_id,
        "page": r["page"] + 1,
        "section": r["section"],
        "span": [r["start"], r["end"]],
        "score": None,
        "snippet": body[:240] + (" ..." if len(body) > 240 else ""),
        "text": body,
    }


def verify_all(conn: sqlite3.Connection, index_dir: str) -> dict:
    """Audit every indexed chunk: span must reproduce the FTS body
    byte-for-byte from the canonical text file. One bad span fails."""
    bad: list[dict] = []
    count = 0
    cache: dict[str, str] = {}
    for r in conn.execute(
            """SELECT chunks_fts.rowid AS chunk_id, chunks.doc_id,
                      chunks.start, chunks.end, chunks_fts.body
               FROM chunks_fts
               JOIN chunks ON chunks.chunk_id = chunks_fts.rowid
               ORDER BY chunks.doc_id, chunks.start"""):
        count += 1
        canonical = cache.get(r["doc_id"])
        if canonical is None:
            canonical = read_canonical(index_dir, r["doc_id"])
            cache[r["doc_id"]] = canonical
        span_text = canonical[r["start"]:r["end"]]
        if span_text != r["body"]:
            bad.append({"chunk_id": r["chunk_id"], "doc_id": r["doc_id"],
                        "start": r["start"], "end": r["end"]})
    return {"checked": count, "bad": bad, "ok": not bad}
