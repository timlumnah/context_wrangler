"""Hybrid retrieval: BM25 and vector run side by side, fused with RRF.

Reciprocal rank fusion, k0 = 60: score(id) = sum over retrievers of
1/(60 + rank). Ranks are 1-based; an id absent from a list contributes 0.

Why RRF and not score blending: bm25() returns an unbounded negative scale
and cosine distance lives in [0, 2]. Normalizing those onto each other is
guesswork; order is the only signal both retrievers agree on. Ties break on
chunk id so the fused ranking is deterministic.
"""

from __future__ import annotations

import sqlite3

from context_wrangler.core.search import chunk_record, search

RRF_K0 = 60.0


def rrf(rank_lists: list[list[int]], k0: float = RRF_K0) -> list[tuple[int, float]]:
    """Fuse 1-based ranked id lists into [(id, score)] descending."""
    scores: dict[int, float] = {}
    for rl in rank_lists:
        for rank, cid in enumerate(rl, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k0 + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def vector_search_ids(conn: sqlite3.Connection, embedder, query: str,
                      n: int) -> list[tuple[int, float]]:
    """KNN over vec_chunks. Returns [(chunk_id, distance)] ascending."""
    import sqlite_vec
    qv = sqlite_vec.serialize_float32(embedder.embed_one(query))
    rows = conn.execute(
        "SELECT chunk_id, distance FROM vec_chunks"
        " WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (qv, n)).fetchall()
    return [(r["chunk_id"], r["distance"]) for r in rows]


def hybrid_search(conn: sqlite3.Connection, index_dir: str, embedder,
                  query: str, *, k: int = 8, k0: float = RRF_K0,
                  pool: int | None = None) -> list[dict]:
    """BM25 pool + vector pool fused to top k. Records carry both ranks and
    the fused score, so a reviewer can see which retriever pulled a hit."""
    pool = pool or max(k * 3, 20)
    bm_hits = search(conn, index_dir, query, k=pool)
    bm_ids = [h["chunk_id"] for h in bm_hits]
    vec_pairs = vector_search_ids(conn, embedder, query, pool)
    vec_ids = [cid for cid, _d in vec_pairs]
    vec_dist = dict(vec_pairs)

    fused = rrf([bm_ids, vec_ids], k0)[:k]
    bm_by = {h["chunk_id"]: h for h in bm_hits}
    out: list[dict] = []
    for cid, score in fused:
        rec = dict(bm_by[cid]) if cid in bm_by else chunk_record(conn, index_dir, cid)
        if rec is None:
            continue                        # chunk vanished mid-flight
        rec["rrf_score"] = score
        rec["bm25_rank"] = (bm_ids.index(cid) + 1) if cid in bm_ids else None
        rec["vec_rank"] = (vec_ids.index(cid) + 1) if cid in vec_ids else None
        rec["vec_distance"] = vec_dist.get(cid)
        out.append(rec)
    return out
