"""sqlite-vec plumbing: embed at ingest, KNN search, no orphan rows,
one embedder per index enforced."""

import pytest

from context_wrangler.core.embedder import HashEmbedder
from context_wrangler.core.index import (embed_config, ensure_vec_table,
                                         has_vectors, open_db)
from context_wrangler.core.hybrid import vector_search_ids
from context_wrangler.core.ingest import ingest_dir


@pytest.fixture
def embedded(corpus):
    emb = HashEmbedder(dim=256)
    rep = ingest_dir(corpus["dir"], corpus["index"], embedder=emb)
    assert not rep.errors
    return corpus


def test_vectors_created(embedded):
    conn = open_db(embedded["index"])
    assert has_vectors(conn)
    assert embed_config(conn) == ("test-hash:256", 256)
    n_chunks = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
    n_vec = conn.execute("SELECT COUNT(*) c FROM vec_chunks").fetchone()["c"]
    assert n_vec == n_chunks
    conn.close()


def test_knn_finds_the_planted_clause(embedded):
    conn = open_db(embedded["index"])
    emb = HashEmbedder(dim=256)
    ids = vector_search_ids(conn, emb, "earnest money deposit escrow", 5)
    assert ids
    top_cid = ids[0][0]
    row = conn.execute("SELECT section FROM chunks WHERE chunk_id = ?",
                       (top_cid,)).fetchone()
    assert "EARNEST MONEY" in (row["section"] or "")
    conn.close()


def test_update_leaves_no_orphans(embedded):
    # touch one file so its hash changes: old vectors must go with it
    with open(embedded["paths"]["addendum_one.pdf"], "ab") as fh:
        fh.write(b"\n")
    ingest_dir(embedded["dir"], embedded["index"], embedder=HashEmbedder(256))
    conn = open_db(embedded["index"])
    n_chunks = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
    n_vec = conn.execute("SELECT COUNT(*) c FROM vec_chunks").fetchone()["c"]
    assert n_vec == n_chunks
    conn.close()


def test_delete_removes_vectors(embedded):
    import os
    os.remove(embedded["paths"]["closing_statement.pdf"])
    ingest_dir(embedded["dir"], embedded["index"], embedder=HashEmbedder(256))
    conn = open_db(embedded["index"])
    n_chunks = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
    n_vec = conn.execute("SELECT COUNT(*) c FROM vec_chunks").fetchone()["c"]
    assert n_vec == n_chunks
    conn.close()


def test_embedder_switch_is_refused(embedded):
    conn = open_db(embedded["index"])
    with pytest.raises(RuntimeError, match="rebuild the index"):
        ensure_vec_table(conn, HashEmbedder(dim=512))
    conn.close()


def test_reingest_without_embedder_keeps_vectors(embedded):
    # incremental by hash: unchanged docs keep vectors; no embedder on a
    # later run must not silently wipe the vec table
    rep = ingest_dir(embedded["dir"], embedded["index"], embedder=None)
    assert len(rep.unchanged) == 3
    conn = open_db(embedded["index"])
    assert has_vectors(conn)
    conn.close()


def test_changed_file_without_embedder_is_refused(embedded):
    # the guard: a vec index may never gain chunks without vectors
    with open(embedded["paths"]["addendum_one.pdf"], "ab") as fh:
        fh.write(b"\n")
    with pytest.raises(RuntimeError, match="refusing"):
        ingest_dir(embedded["dir"], embedded["index"], embedder=None)


def test_fresh_index_without_embedder_is_fine(corpus):
    rep = ingest_dir(corpus["dir"], corpus["index"] + "_plain",
                    embedder=None)
    assert len(rep.added) == 3


def test_backfill_when_embedding_an_existing_index(corpus):
    # v0.1-style bm25 index first, then --embed arrives: old chunks must
    # not stay invisible to KNN
    ingest_dir(corpus["dir"], corpus["index"])
    rep = ingest_dir(corpus["dir"], corpus["index"], embedder=HashEmbedder(256))
    assert rep.backfilled > 0
    conn = open_db(corpus["index"])
    n_chunks = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
    n_vec = conn.execute("SELECT COUNT(*) c FROM vec_chunks").fetchone()["c"]
    assert n_vec == n_chunks == rep.backfilled
    ids = vector_search_ids(conn, HashEmbedder(256), "earnest money", 3)
    assert ids
    conn.close()
