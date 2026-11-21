"""SQLite schema and storage for the one-file index.

Layout of an index dir (default ./cw_index):
    index.db          SQLite: docs, pages, chunks, FTS5 mirror
    text/<sha256>.txt canonical extracted text per document

Citation spans are character offsets into the canonical text file. The file is
the ground truth the grounding checks byte-verify against; raw PDF bytes are
not (extraction normalizes whitespace, so they never would match).

FTS5 is a mirror of chunk spans, rebuilt per ingest run. At the scale this
tool targets (10^4-10^5 docs) a rebuild of changed docs is cheap; full
incremental FTS maintenance is deliberately not v0.1.
"""

from __future__ import annotations

import os
import sqlite3

# v2 -> v3 (v0.5): docs gains ocr_pages, and meta gains the ocr_config /
# tables_config keys written by the extraction guards below. Old indexes
# refuse to open; re-ingest. Same rule as v1 -> v2: an index must not mix
# extraction regimes, because citations byte-verify against the canonical
# text AS EXTRACTED, and two regimes in one index is two truths.
SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS docs (
    doc_id TEXT PRIMARY KEY,          -- sha256 of source file bytes
    filename TEXT NOT NULL,
    source_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    n_pages INTEGER NOT NULL,
    text_len INTEGER NOT NULL,
    ocr_pages INTEGER NOT NULL DEFAULT 0,   -- pages whose text came from --ocr
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pages (
    doc_id TEXT NOT NULL REFERENCES docs(doc_id) ON DELETE CASCADE,
    page_no INTEGER NOT NULL,          -- 0-based
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    PRIMARY KEY (doc_id, page_no)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES docs(doc_id) ON DELETE CASCADE,
    start INTEGER NOT NULL,            -- inclusive char offset into canonical text
    end INTEGER NOT NULL,              -- exclusive
    page INTEGER NOT NULL,             -- page where the chunk starts
    section TEXT                        -- heading the chunk belongs to, if any
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    body,
    docmeta,
    chunk_id UNINDEXED,
    tokenize = 'unicode61'
);
"""


def _load_vec_extension(conn: sqlite3.Connection) -> None:
    """sqlite-vec is optional: the BM25-only index must work without it."""
    try:
        import sqlite_vec
    except ImportError as exc:               # pragma: no cover
        raise RuntimeError(
            "this index contains vectors but sqlite-vec is not installed; "
            "pip install sqlite-vec") from exc
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def has_vectors(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'vec_chunks'"
    ).fetchone() is not None


def db_path(index_dir: str) -> str:
    return os.path.join(index_dir, "index.db")


def text_path(index_dir: str, doc_id: str) -> str:
    return os.path.join(index_dir, "text", f"{doc_id}.txt")


def open_db(index_dir: str) -> sqlite3.Connection:
    os.makedirs(os.path.join(index_dir, "text"), exist_ok=True)
    conn = sqlite3.connect(db_path(index_dir))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                     (str(SCHEMA_VERSION),))
        conn.commit()
    elif int(row["value"]) != SCHEMA_VERSION:
        raise RuntimeError(
            f"index schema version {row['value']} != supported {SCHEMA_VERSION}; "
            "re-ingest with a fresh index dir")
    if has_vectors(conn):
        _load_vec_extension(conn)
    return conn


def embed_config(conn: sqlite3.Connection) -> tuple[str, int] | None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'embed_model'").fetchone()
    if row is None:
        return None
    dim = int(conn.execute(
        "SELECT value FROM meta WHERE key = 'embed_dim'").fetchone()["value"])
    return row["value"], dim


def _config_guard(conn: sqlite3.Connection, key: str, want: str | None,
                  human: str) -> None:
    """One extraction regime per index, the same way embedders get one
    vector space per index. `want` is the config this ingest run would
    write (None = the plain/default regime). Absent meta gets it stamped
    on first use; a disagreeing re-ingest raises instead of quietly
    producing canonical text two ways in one index."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    have = row["value"] if row else None
    if have is None:
        if want is not None:
            conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)",
                         (key, want))
            conn.commit()
        return
    if have != want:
        raise RuntimeError(
            f"index was ingested with {human} {have!r}, this run says "
            f"{want!r}; re-ingest the whole corpus with one regime or "
            "rebuild the index dir")


def ensure_ocr_config(conn: sqlite3.Connection, config: str | None) -> None:
    # "none" is a stamp, not an absence: the regime locks at FIRST ingest,
    # so a later run cannot silently flip pages of the same index between
    # extraction regimes.
    _config_guard(conn, "ocr_config", config or "none", "ocr tier")


def ensure_tables_config(conn: sqlite3.Connection, tables: bool) -> None:
    _config_guard(conn, "tables_config",
                  "layout-reflow" if tables else "plain", "table extraction")


def ensure_vec_table(conn: sqlite3.Connection, embedder) -> None:
    """One active embedder per index. Switching providers means re-embedding
    the whole corpus: that is enforced here rather than silently mixing
    vector spaces, which would score garbage distances."""
    cfg = embed_config(conn)
    if cfg is not None:
        if cfg != (embedder.name, embedder.dim):
            raise RuntimeError(
                f"index was embedded with {cfg[0]} (dim {cfg[1]}), not "
                f"{embedder.name} (dim {embedder.dim}); rebuild the index dir "
                "to switch embedders")
        return
    _load_vec_extension(conn)
    conn.execute(
        f"CREATE VIRTUAL TABLE vec_chunks USING vec0("
        f"chunk_id INTEGER PRIMARY KEY,"
        f"embedding FLOAT[{embedder.dim}] distance_metric = cosine)")
    conn.execute("INSERT INTO meta (key, value) VALUES ('embed_model', ?)",
                 (embedder.name,))
    conn.execute("INSERT INTO meta (key, value) VALUES ('embed_dim', ?)",
                 (str(embedder.dim),))
    conn.commit()


def add_chunk_vectors(conn: sqlite3.Connection,
                      rows: list[tuple[int, list[float]]]) -> None:
    import sqlite_vec
    conn.executemany(
        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
        [(cid, sqlite_vec.serialize_float32(vec)) for cid, vec in rows])


def read_canonical(index_dir: str, doc_id: str) -> str:
    with open(text_path(index_dir, doc_id), encoding="utf-8") as fh:
        return fh.read()


def delete_doc(conn: sqlite3.Connection, doc_id: str) -> None:
    """Remove a document and everything that points at it."""
    ids = [r["chunk_id"] for r in
           conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))]
    for cid in ids:
        conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
        if has_vectors(conn):
            conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
    # chunks and pages go via ON DELETE CASCADE
    conn.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))


def insert_doc(conn: sqlite3.Connection, *, doc_id: str, filename: str,
               source_path: str, sha256: str, n_pages: int, text_len: int,
               ingested_at: str, page_spans: list[tuple[int, int]],
               chunk_rows: list[tuple[int, int, int, str | None, str]],
               docmeta: str = "", ocr_pages: int = 0
               ) -> list[tuple[int, str]]:
    """Store one document. chunk_rows: (start, end, page, section, body).

    docmeta is indexed once per chunk row: document identity (parties,
    address, doc type) lives at the head of the file, but an AND query
    needs it to co-occur with a clause that sits three chunks later. The
    column carries that identity so cross-chunk joins work in BM25 without
    polluting the byte-verified body.

    Returns [(chunk_id, body)] so the caller can embed in the same order.
    Body is written to the FTS mirror; offsets are the truth, the mirror is
    checked against them by verify().
    """
    conn.execute(
        "INSERT INTO docs (doc_id, filename, source_path, sha256, n_pages,"
        " text_len, ocr_pages, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (doc_id, filename, source_path, sha256, n_pages, text_len,
         ocr_pages, ingested_at))
    conn.executemany(
        "INSERT INTO pages (doc_id, page_no, start, end) VALUES (?, ?, ?, ?)",
        [(doc_id, i, s, e) for i, (s, e) in enumerate(page_spans)])
    inserted: list[tuple[int, str]] = []
    for start, end, page, section, body in chunk_rows:
        cur = conn.execute(
            "INSERT INTO chunks (doc_id, start, end, page, section)"
            " VALUES (?, ?, ?, ?, ?)",
            (doc_id, start, end, page, section))
        conn.execute(
            "INSERT INTO chunks_fts (rowid, body, docmeta, chunk_id)"
            " VALUES (?, ?, ?, ?)",
            (cur.lastrowid, body, docmeta, cur.lastrowid))
        inserted.append((cur.lastrowid, body))
    return inserted
