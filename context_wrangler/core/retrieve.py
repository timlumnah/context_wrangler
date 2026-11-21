"""One retrieval path for everyone: search CLI, answer CLI, eval harness.

mode auto = hybrid when the index has vectors and sqlite-vec loads, bm25
otherwise. The embedder is built from the index's own embed_model meta and
cached, so a 162-question eval loop does not re-probe the model each call.
"""

from __future__ import annotations

import sqlite3

from context_wrangler.core.index import embed_config, has_vectors
from context_wrangler.core.search import chunk_record, search

_embedder_cache: dict[str, object] = {}


def get_embedder(conn: sqlite3.Connection):
    name = conn.execute(
        "SELECT value FROM meta WHERE key = 'embed_model'").fetchone()
    if name is None:
        raise RuntimeError("index was built without --embed")
    from context_wrangler.core.embedder import make_embedder
    key = name["value"]
    if key not in _embedder_cache:
        _embedder_cache[key] = make_embedder(key)
    return _embedder_cache[key]


def candidates(conn: sqlite3.Connection, index_dir: str, query: str, *,
               mode: str = "auto", k: int = 8) -> list[dict]:
    vec_ok = has_vectors(conn)
    if mode == "auto":
        mode = "hybrid" if vec_ok else "bm25"
    if mode == "bm25":
        return search(conn, index_dir, query, k=k)
    if not vec_ok:
        raise RuntimeError(f"mode {mode} needs vectors; index has none "
                           "(re-ingest with --embed)")
    if mode == "vec":
        from context_wrangler.core.hybrid import vector_search_ids
        emb = get_embedder(conn)
        hits = []
        for cid, dist in vector_search_ids(conn, emb, query, k):
            rec = chunk_record(conn, index_dir, cid)
            if rec:
                rec["vec_distance"] = dist
                hits.append(rec)
        return hits
    if mode == "hybrid":
        from context_wrangler.core.hybrid import hybrid_search
        return hybrid_search(conn, index_dir, get_embedder(conn), query, k=k)
    raise ValueError(f"unknown mode {mode!r}")
