"""RRF math and the hybrid pipeline over a test-hash embedded index."""

import pytest

from context_wrangler.core.embedder import HashEmbedder
from context_wrangler.core.hybrid import hybrid_search, rrf
from context_wrangler.core.index import open_db, read_canonical
from context_wrangler.core.ingest import ingest_dir


def test_rrf_hand_computed():
    # a: rank1 in list1 only  -> 1/61
    # b: rank2 in list1, rank1 in list2 -> 1/62 + 1/61
    # c: rank2 in list2 only -> 1/62
    fused = rrf([[1, 2], [2, 3]], k0=60)
    ids = [cid for cid, _s in fused]
    assert ids == [2, 1, 3]
    assert abs(fused[0][1] - (1 / 62 + 1 / 61)) < 1e-12


def test_rrf_deterministic_tiebreak():
    assert rrf([[5], [7]]) == rrf([[5], [7]])
    fused = rrf([[5], [7]])
    assert [cid for cid, _ in fused] == [5, 7]   # tie -> lower id first


def test_rrf_empty():
    assert rrf([[], []]) == []


def test_hybrid_records_and_spans(embedded_corpus):
    conn, index_dir = embedded_corpus
    emb = HashEmbedder(256)
    hits = hybrid_search(conn, index_dir, emb, "earnest money deposit", k=3)
    assert hits
    for h in hits:
        canonical = read_canonical(index_dir, h["doc_id"])
        assert canonical[h["span"][0]:h["span"][1]] == h["text"]
        assert "rrf_score" in h
        assert h["bm25_rank"] is not None or h["vec_rank"] is not None
    conn.close()


def test_hybrid_rescues_a_phrase_bm25_misses(embedded_corpus):
    # paraphrased query: BM25 AND finds nothing (tokens absent together),
    # the lexical-hash vec side still returns neighbors, and hybrid surfaces
    # them where bm25 gives an empty list
    conn, index_dir = embedded_corpus
    emb = HashEmbedder(256)
    from context_wrangler.core.search import search
    assert search(conn, index_dir, "down payment held in escrow") == []
    hits = hybrid_search(conn, index_dir, emb,
                         "down payment held in escrow", k=3)
    assert hits, "hybrid should still return vector neighbors"
    conn.close()


@pytest.fixture
def embedded_corpus(corpus):
    rep = ingest_dir(corpus["dir"], corpus["index"],
                     embedder=HashEmbedder(256))
    assert not rep.errors
    conn = open_db(corpus["index"])
    return conn, corpus["index"]
