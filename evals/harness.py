"""Run the answer key against each retriever tier and publish the table.

Grounding gate first, numbers second: cw verify byte-checks every span, and
a failed span check aborts the whole run before any score is computed.
An answer key graded against a corrupt index is worse than no answer key.

Tiers available in v0.2:
    bm25   keyword only, no model at all
    vec    pure KNN over the embedded index
    hybrid BM25 + vec fused with RRF

The embedder is taken from the index's own metadata (set at ingest time).
test-hash is refused here on purpose: lexical hashing is plumbing proof,
not a quality tier, and it must never leak into a published table.
"""

from __future__ import annotations

import datetime
import json
import platform
import sqlite3

from context_wrangler.core.embedder import make_embedder
from context_wrangler.core.hybrid import hybrid_search, vector_search_ids
from context_wrangler.core.index import embed_config, has_vectors, open_db
from context_wrangler.core.search import search, verify_all
from evals.metrics import (recall_at_k, reciprocal_rank, selftest,
                           superseded_in_top_n, top1_correct)

K = 10


class CachedEmbedder:
    """Memoize query embeddings; evals ask the same queries repeatedly."""

    def __init__(self, inner):
        self.inner = inner
        self.cache: dict[str, list[float]] = {}
        self.name = inner.name
        self.dim = inner.dim
        self.publishable = inner.publishable

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in self.cache]
        if missing:
            for t, v in zip(missing, self.inner.embed(missing)):
                self.cache[t] = v
        return [self.cache[t] for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def _doc_ids_by_chunk(conn: sqlite3.Connection,
                      chunk_ids: list[int]) -> list[str]:
    if not chunk_ids:
        return []
    marks = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT chunk_id, doc_id FROM chunks WHERE chunk_id IN ({marks})",
        chunk_ids)
    mapping = {r["chunk_id"]: r["doc_id"] for r in rows}
    return [mapping[c] for c in chunk_ids if c in mapping]


def retrieve_ranked_docs(conn, index_dir, query: str, mode: str,
                         embedder, k: int = 30) -> list[str]:
    """Ranked document ids for one query under one retriever mode."""
    if mode == "bm25":
        hits = search(conn, index_dir, query, k=k)
        return [h["doc_id"] for h in hits]
    if mode == "vec":
        pairs = vector_search_ids(conn, embedder, query, k)
        return _doc_ids_by_chunk(conn, [cid for cid, _d in pairs])
    if mode == "hybrid":
        hits = hybrid_search(conn, index_dir, embedder, query, k=k)
        return [h["doc_id"] for h in hits]
    raise ValueError(f"unknown mode {mode!r}")


def retrieve_ranked_chunks(conn, index_dir, query: str, mode: str,
                           embedder, k: int = 30) -> list[int]:
    """Ranked CHUNK ids. Doc-level recall cannot see row adjacency: a
    table question hits the right document while every chunk keeps the
    label apart from its figure. Table questions are additionally graded
    at chunk level; this is the path that makes that a measurement."""
    if mode == "bm25":
        return [h["chunk_id"] for h in search(conn, index_dir, query, k=k)]
    if mode == "vec":
        return [cid for cid, _d in vector_search_ids(conn, embedder, query, k)]
    if mode == "hybrid":
        return [h["chunk_id"] for h in
                hybrid_search(conn, index_dir, embedder, query, k=k)]
    raise ValueError(f"unknown mode {mode!r}")


def _chunk_bodies(conn, index_dir: str, chunk_ids: list[int]) -> list[str]:
    """Verbatim chunk texts, same order, canonical file read once per doc."""
    if not chunk_ids:
        return []
    marks = ",".join("?" * len(chunk_ids))
    rows = {r["chunk_id"]: r for r in conn.execute(
        f"SELECT chunk_id, doc_id, start, end FROM chunks "
        f"WHERE chunk_id IN ({marks})", chunk_ids)}
    cache: dict[str, str] = {}
    out: list[str] = []
    from context_wrangler.core.index import read_canonical
    for cid in chunk_ids:
        r = rows.get(cid)
        if r is None:
            continue
        canonical = cache.get(r["doc_id"])
        if canonical is None:
            canonical = read_canonical(index_dir, r["doc_id"])
            cache[r["doc_id"]] = canonical
        out.append(canonical[r["start"]:r["end"]])
    return out


def _scan_translator(conn, key: dict) -> dict[str, str]:
    """key doc ids (text-PDF hashes) -> scanned-index doc ids, joined by
    filename. The scanned twin of a document is a different file with the
    same name and the same planted facts; scanify keeps names stable
    precisely so this mapping needs no extra artifact."""
    fname_of = {did: f for f, did in key.get("doc_ids", {}).items()}
    if not fname_of:
        raise RuntimeError("scan map needs a key with doc_ids")
    scan_ids = {r["filename"]: r["doc_id"] for r in
                conn.execute("SELECT filename, doc_id FROM docs")}
    trans: dict[str, str] = {}
    for tid, fname in fname_of.items():
        sid = scan_ids.get(fname)
        if sid is None:
            raise RuntimeError(
                f"scan map: index has no document named {fname!r}; is this "
                "the scanned twin of the same corpus?")
        trans[tid] = sid
    return trans


def run(index_dir: str, key_path: str, modes: tuple[str, ...] = ("bm25",),
        k: int = K, scan_map: bool = False) -> dict:
    conn = open_db(index_dir)

    # grounding gate: every span must byte-verify before any scoring
    vg = verify_all(conn, index_dir)
    if not vg["ok"]:
        conn.close()
        raise RuntimeError(
            f"grounding failed: {len(vg['bad'])} of {vg['checked']} spans "
            "do not reproduce; fix the index before trusting eval numbers")

    key = json.load(open(key_path, encoding="utf-8"))
    questions = [qq for qq in key["questions"]]
    trans = _scan_translator(conn, key) if scan_map else None

    embedder = None
    if any(m in ("vec", "hybrid") for m in modes):
        if not has_vectors(conn):
            conn.close()
            raise RuntimeError("index has no vectors; re-ingest with --embed")
        cfg = embed_config(conn)
        if cfg[0].startswith("test-hash"):
            conn.close()
            raise RuntimeError(
                f"refusing to publish a quality tier from {cfg[0]}: "
                "feature hashing is a plumbing test, not an embedder. "
                "re-ingest with a real model")
        embedder = CachedEmbedder(make_embedder(cfg[0]))

    rows: dict[str, dict] = {}
    per_question: dict[str, list[dict]] = {}
    for mode in modes:
        graded = []
        for qq in questions:
            docs = retrieve_ranked_docs(conn, index_dir, qq["query"], mode,
                                        embedder, k=max(k * 3, 30))
            rel = {(trans[d] if trans else d) for d in qq["expected_docs"]}
            rec = {"id": qq["id"], "kind": qq["kind"],
                   "top_doc": docs[0] if docs else None,
                   "n_hits_topk": len(set(docs[:k]))}
            if qq["kind"] == "refusal":
                rec["hits_topk"] = len(set(docs[:k]))
            else:
                rec["recall"] = recall_at_k(docs, rel, k)
                rec["mrr"] = reciprocal_rank(docs, rel)
                if qq["kind"] == "trap":
                    rec["top1"] = top1_correct(docs, rel)
                    rec["stale_in_top3"] = superseded_in_top_n(
                        docs, {(trans[d] if trans else d)
                               for d in qq["superseded_docs"]}, 3)
                if qq["kind"] == "table":
                    cids = retrieve_ranked_chunks(
                        conn, index_dir, qq["query"], mode, embedder, k=k)
                    bodies = _chunk_bodies(conn, index_dir, cids[:k])
                    rec["row_top10"] = int(any(
                        qq["expected_text"] in b for b in bodies))
            graded.append(rec)
        per_question[mode] = graded

        agg: dict[str, float] = {"grounded_spans": vg["checked"]}
        for kind in ("exact", "table", "trap", "join", "refusal"):
            subset = [g for g in graded if g["kind"] == kind]
            if not subset:
                continue
            if kind == "refusal":
                agg["refusal_mean_hits"] = (
                    sum(g["hits_topk"] for g in subset) / len(subset))
            else:
                agg[f"{kind}_recall"] = sum(g["recall"] for g in subset) / len(subset)
                agg[f"{kind}_mrr"] = sum(g["mrr"] for g in subset) / len(subset)
                if kind == "table":
                    agg["table_row_top10"] = (
                        sum(g["row_top10"] for g in subset) / len(subset))
                if kind == "join":
                    # join@10 counts only the queries where BOTH sides made
                    # the window; half a join is not a join
                    agg["join_recall"] = sum(
                        1 for g in subset if g["recall"] == 1.0) / len(subset)
                if kind == "trap":
                    agg["trap_top1"] = sum(g["top1"] for g in subset) / len(subset)
                    agg["trap_stale_top3"] = (
                        sum(g["stale_in_top3"] for g in subset) / len(subset))
        agg["n_questions"] = len(graded)
        rows[mode] = agg
    conn.close()

    return {
        "run_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "index": index_dir,
        "scan_map": bool(scan_map),
        "key": {"generated": key.get("generated_utc"),
                "deals": key.get("deals"),
                "questions": len(questions)},
        "env": {
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
            "embedder": embedder.name if embedder else None,
        },
        "k": k,
        "aggregate": rows,
        "per_question": per_question,
    }


def to_markdown(report: dict) -> str:
    scan_note = (" Scanned-corpus run: doc ids are mapped from the key by "
                 "filename, the indexed files are image-only twins."
                 if report.get("scan_map") else "")
    lines = [
        f"Retrieval eval, run {report['run_utc']}. k={report['k']}, "
        f"{report['key']['deals']} deals, "
        f"{report['key']['questions']} questions. "
        f"Python {report['env']['python']}, SQLite {report['env']['sqlite']}, "
        f"embedder {report['env']['embedder'] or 'none (BM25 only)'}.{scan_note}",
        "",
        "| Mode | exact recall@10 | exact MRR | table recall@10 | table row@10 "
        "| trap top-1 | trap stale-in-top-3 | join@10 | refusal mean hits |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for mode, agg in report["aggregate"].items():
        fmt = lambda key: (f"{agg[key]:.3f}" if key in agg else "-")
        lines.append(
            f"| {mode} | {fmt('exact_recall')} | {fmt('exact_mrr')} "
            f"| {fmt('table_recall')} | {fmt('table_row_top10')} "
            f"| {fmt('trap_top1')} | {fmt('trap_stale_top3')} "
            f"| {fmt('join_recall')} | {fmt('refusal_mean_hits')} |")
    return "\n".join(lines)


def run_selftest() -> int:
    results = selftest()
    failed = [label for label, ok in results if not ok]
    for label, ok in results:
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    print(f"selftest: {len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0
