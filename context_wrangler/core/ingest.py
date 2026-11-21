"""Ingest: PDF folder -> canonical text files + SQLite index.

Every document gets a content hash (sha256 of file bytes). Re-runs only
process files whose hash is new. The canonical extracted text is written to
text/<hash>.txt and is the byte-level ground truth for citations.

Extraction regimes, v0.5:
  plain text layer      pypdf default. The v0.1 path, byte-stable.
  --tables              reflow coordinate-aligned grids into pipe rows
                        (core/tables.py); fires only on pages with a grid,
                        everything else byte-identical to the plain path.
  --ocr TIER            pages whose text layer is thinner than
                        TEXT_LAYER_MIN_CHARS get rasterized and read by
                        the tier (core/ocr.py). Text layers always win.
Whichever regime an index starts with is stamped in meta and later runs
must agree; two extraction regimes in one index is two truths.
"""

from __future__ import annotations

import datetime
import hashlib
import os
from dataclasses import dataclass

from pypdf import PdfReader

from context_wrangler.core.chunk import Chunk, chunk_text
from context_wrangler.core.index import (add_chunk_vectors, delete_doc,
                                         embed_config, ensure_ocr_config,
                                         ensure_tables_config,
                                         ensure_vec_table, has_vectors,
                                         insert_doc, open_db, text_path)
from context_wrangler.core.ocr import TEXT_LAYER_MIN_CHARS
from context_wrangler.core.tables import page_text as extraction_page_text


def embed_text(docmeta: str, section: str | None, body: str) -> str:
    """What actually goes into the embedder: document identity, section
    heading when present, then the chunk body. Same reasoning as the FTS
    docmeta column: a clause three chunks from the parties line still needs
    to answer to that identity. Queries embed bare."""
    head = docmeta if not section else f"{docmeta} | {section}"
    return f"{head}\n{body}" if head else body


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _normalize_page(raw: str | None) -> str:
    """Deterministic page text. Newlines unified, trailing spaces stripped.

    Blank lines are preserved: the chunker reads them as paragraph breaks.
    """
    if raw is None:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    out = "\n".join(lines)
    return out.strip("\n")


def extract_pages(path: str, *, ocr=None, tables: bool = False
                  ) -> tuple[list[str], int]:
    """(Page texts in order, count of pages that came from OCR).

    Tables pages keep the plain-mode bytes whenever no grid is found,
    byte-identical to the v0.4 path. OCR fires only where the text layer
    is thinner than TEXT_LAYER_MIN_CHARS: a real text layer always wins,
    because the tool's whole claim is citing the extracted bytes.
    """
    reader = PdfReader(path)
    pages: list[str] = []
    ocr_pages = 0
    for i, page in enumerate(reader.pages):
        text = _normalize_page(extraction_page_text(page, tables=tables))
        if ocr is not None and len(text.strip()) < TEXT_LAYER_MIN_CHARS:
            ocrd = _normalize_page(ocr.page_text(path, i))
            if ocrd.strip():
                text = ocrd
                ocr_pages += 1
        pages.append(text)
    return pages, ocr_pages


def canonical_text(pages: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join pages with a single \\n. Returns text plus per-page [start, end).
    The join newline belongs to no page span."""
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for i, page in enumerate(pages):
        if i:
            pos += 1                      # the joining newline
        start = pos
        pos += len(page)
        spans.append((start, pos))
        parts.append(page)
    return "\n".join(parts), spans


@dataclass
class IngestReport:
    added: list[str]
    updated: list[str]
    moved: list[str]
    unchanged: list[str]
    removed: list[str]
    duplicates: list[str]
    errors: list[tuple[str, str]]
    backfilled: int = 0
    ocr_pages: int = 0            # pages read by the OCR tier this run

    @property
    def processed(self) -> int:
        return len(self.added) + len(self.updated)


def _backfill_vectors(conn, index_dir: str, embedder,
                      report: IngestReport) -> None:
    """Embed chunks that exist in the index but not in vec_chunks. Covers
    the case of adding --embed to an index that was built without it, and
    closes the hole where old chunks would be invisible to every KNN
    query. The embed text is built by the same make_docmeta/embed_text
    path ingest uses, so vectors stay comparable to each other."""
    from context_wrangler.core.index import read_canonical
    missing = conn.execute(
        """SELECT c.chunk_id, c.doc_id, c.start, c.end, c.section,
                  d.filename
           FROM chunks c JOIN docs d ON d.doc_id = c.doc_id
           WHERE c.chunk_id NOT IN (SELECT chunk_id FROM vec_chunks)
           ORDER BY c.chunk_id""").fetchall()
    if not missing:
        return
    cache: dict[str, str] = {}
    texts: list[tuple[int, str]] = []
    for r in missing:
        if r["doc_id"] not in cache:
            cache[r["doc_id"]] = read_canonical(index_dir, r["doc_id"])
        text = cache[r["doc_id"]]
        body = text[r["start"]:r["end"]]
        texts.append((r["chunk_id"],
                      embed_text(make_docmeta(text, r["filename"]),
                                 r["section"], body)))
    vecs = embedder.embed([t for _cid, t in texts])
    add_chunk_vectors(conn, [(cid, v) for (cid, _t), v in zip(texts, vecs)])
    report.backfilled = len(texts)


def make_docmeta(text: str, filename: str) -> str:
    return text[:300].replace("\n", " ") + " " + filename.replace("_", " ")


def _chunk_rows(text: str, page_spans) -> list:
    chunks: list[Chunk] = chunk_text(text, page_spans=page_spans)
    return [(c.start, c.end, c.page, c.section, text[c.start:c.end])
            for c in chunks]


def ingest_dir(pdf_dir: str, index_dir: str, embedder=None, *, ocr=None,
               tables: bool = False) -> IngestReport:
    """Index every *.pdf under pdf_dir into index_dir. Incremental by hash.

    embedder (optional, from core.embedder.make_embedder): embed each
    newly-processed doc's chunks into the sqlite-vec table. Only changed
    docs are re-embedded; unchanged docs keep their vectors.

    ocr (optional, from core.ocr.make_ocr) and tables (bool) are the two
    v0.5 extraction regimes; both are guarded against silently mixing
    regimes inside one index (index.ensure_ocr_config /
    ensure_tables_config).
    """
    conn = open_db(index_dir)
    ensure_ocr_config(conn, ocr.config if ocr is not None else None)
    ensure_tables_config(conn, tables)
    report = IngestReport([], [], [], [], [], [], [])
    if embedder is not None:
        ensure_vec_table(conn, embedder)
        _backfill_vectors(conn, index_dir, embedder, report)
    now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")

    on_disk: dict[str, str] = {}          # path -> hash
    for root, _dirs, files in os.walk(pdf_dir):
        for name in sorted(files):
            if not name.lower().endswith(".pdf"):
                continue
            full = os.path.abspath(os.path.join(root, name))
            try:
                on_disk[full] = sha256_file(full)
            except OSError as exc:
                report.errors.append((full, f"read: {exc}"))

    known = {r["source_path"]: r for r in
             conn.execute("SELECT * FROM docs")}
    hash_to_doc = {r["sha256"]: r["doc_id"] for r in
                   conn.execute("SELECT doc_id, sha256 FROM docs")}

    # removals: indexed paths that vanished. if the same content hash still
    # exists on disk, the move branch below re-points it instead.
    for path, row in known.items():
        if path not in on_disk:
            if any(h == row["sha256"] for h in on_disk.values()):
                continue                   # will be handled as a move
            delete_doc(conn, row["doc_id"])
            tp = text_path(index_dir, row["doc_id"])
            if os.path.exists(tp):
                os.remove(tp)
            report.removed.append(path)

    for path, digest in sorted(on_disk.items()):
        old = known.get(path)
        if old is not None and old["sha256"] == digest:
            report.unchanged.append(path)
            continue
        if digest in hash_to_doc:
            # same content already indexed (doc_id is the hash itself).
            # re-point only on a true move: the old path is gone from disk.
            # two live paths with identical bytes: report the duplicate and
            # keep the first-indexed path. no ping-pong.
            doc_id = hash_to_doc[digest]
            live_old = conn.execute(
                "SELECT source_path FROM docs WHERE doc_id = ?", (doc_id,))
            row = live_old.fetchone()
            if row and row["source_path"] not in on_disk:
                conn.execute("UPDATE docs SET source_path = ?, filename = ?,"
                             " ingested_at = ? WHERE doc_id = ?",
                             (path, os.path.basename(path), now, doc_id))
                report.moved.append(path)
            else:
                report.duplicates.append(path)
            continue

        if embedder is None and has_vectors(conn):
            # we only get here for added/updated files: an index that has
            # vectors must not gain chunks without vectors, that is an
            # invisible hole in every KNN query
            raise RuntimeError(
                f"file needs processing ({path}) but no --embed was given "
                "and this index has vectors; refusing. re-run with --embed "
                f"{embed_config(conn)[0]}")
        try:
            pages, n_ocr = extract_pages(path, ocr=ocr, tables=tables)
        except Exception as exc:           # noqa: BLE001 - one bad PDF is data
            report.errors.append((path, f"extract: {exc}"))
            continue

        text, page_spans = canonical_text(pages)
        doc_id = digest
        tpath = text_path(index_dir, doc_id)
        with open(tpath, "w", encoding="utf-8") as fh:
            fh.write(text)

        if old is not None:
            delete_doc(conn, old["doc_id"])
            stale = text_path(index_dir, old["doc_id"])
            if os.path.exists(stale):
                os.remove(stale)     # content-addressed: no other doc shares
            op = report.updated      # a hash, once the row is gone
        else:
            op = report.added
        rows = _chunk_rows(text, page_spans)
        docmeta = make_docmeta(text, os.path.basename(path))
        inserted = insert_doc(conn, doc_id=doc_id, filename=os.path.basename(path),
                              source_path=path, sha256=digest,
                              n_pages=len(pages), text_len=len(text),
                              ingested_at=now, page_spans=page_spans,
                              chunk_rows=rows, docmeta=docmeta,
                              ocr_pages=n_ocr)
        report.ocr_pages += n_ocr
        if embedder is not None and inserted:
            vecs = embedder.embed(
                [embed_text(docmeta, r[3], body)
                 for r, (_cid, body) in zip(rows, inserted)])
            add_chunk_vectors(conn, [(cid, v) for (cid, _b), v
                                     in zip(inserted, vecs)])
        hash_to_doc[digest] = doc_id
        op.append(path)

    conn.commit()
    if embedder is not None:
        # meter the ingest: OpenRouter reports exact dollars per embed
        # batch; the frontier table reads this back as the one-time
        # cost of having an embeddable index at all.
        meter = getattr(embedder, "usage", None)
        if meter is not None and meter.cost_usd is not None:
            prev = conn.execute(
                "SELECT value FROM meta WHERE key = 'embed_cost_usd'"
            ).fetchone()
            total = (float(prev["value"]) if prev else 0.0) + meter.cost_usd
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) "
                "VALUES ('embed_cost_usd', ?)", (repr(total),))
            conn.commit()
    conn.close()
    return report
