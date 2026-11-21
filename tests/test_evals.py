"""The harness end to end on a small seeded corpus, plus the gates that
refuse to publish garbage: corrupt spans abort, test-hash aborts."""

import json
import os

import pytest

from context_wrangler.core.embedder import HashEmbedder
from context_wrangler.core.index import open_db
from context_wrangler.core.ingest import ingest_dir
from evals import harness
from evals.metrics import selftest


def _seed_corpus(tmp_path, deals=6, seed=42, embedder=None):
    from context_wrangler.corpus import generate_corpus, write_corpus
    c = generate_corpus(deals=deals, seed=seed)
    out = str(tmp_path / "corpus")
    write_corpus(out, c)
    idx = str(tmp_path / "idx")
    rep = ingest_dir(os.path.join(out, "pdfs"), idx, embedder=embedder)
    assert not rep.errors
    return idx, os.path.join(out, "answer_key.json")


def test_selftest_passes():
    results = selftest()
    assert len(results) >= 12
    failed = [label for label, ok in results if not ok]
    assert not failed, failed


def test_harness_bm25_small_corpus(tmp_path):
    idx, key = _seed_corpus(tmp_path, deals=6, seed=42)
    report = harness.run(idx, key, modes=("bm25",))
    agg = report["aggregate"]["bm25"]
    assert agg["exact_recall"] > 0.9, agg
    assert agg["n_questions"] > 25
    assert agg["refusal_mean_hits"] == 0.0   # AND queries cannot match ghosts
    # traps are the honest gap: bm25 alone should NOT ace them
    assert 0.0 <= agg["trap_top1"] <= 1.0
    assert agg["trap_stale_top3"] > 0.0 or agg["trap_top1"] == 1.0
    md = harness.to_markdown(report)
    assert "| bm25 |" in md


def test_harness_grounding_gate_aborts(tmp_path):
    idx, key = _seed_corpus(tmp_path, deals=3, seed=7)
    conn = open_db(idx)
    conn.execute("UPDATE chunks_fts SET body = 'TAMPERED' WHERE rowid = 1")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="grounding failed"):
        harness.run(idx, key, modes=("bm25",))


def test_harness_refuses_publishing_test_hash(tmp_path):
    idx, key = _seed_corpus(tmp_path, deals=3, seed=7,
                            embedder=HashEmbedder(256))
    with pytest.raises(RuntimeError, match="not an embedder"):
        harness.run(idx, key, modes=("hybrid",))


def test_harness_hybrid_plumbing(tmp_path, monkeypatch):
    """Stand in a fake publishable embedder so the KNN side runs without a
    model: the point here is that hybrid produces a complete report."""
    idx, key = _seed_corpus(tmp_path, deals=3, seed=7,
                            embedder=HashEmbedder(256))
    conn = open_db(idx)
    conn.execute("UPDATE meta SET value = 'fake-embed:256' WHERE key = 'embed_model'")
    conn.commit()
    conn.close()

    class Fake:
        name = "fake-embed:256"
        dim = 256
        publishable = True

        def __init__(self):
            self.inner = HashEmbedder(256)

        def embed(self, texts):
            return self.inner.embed(texts)

        def embed_one(self, text):
            return self.inner.embed_one(text)

    monkeypatch.setattr(harness, "make_embedder", lambda spec: Fake())
    report = harness.run(idx, key, modes=("hybrid",))
    agg = report["aggregate"]["hybrid"]
    assert agg["n_questions"] > 5
    assert agg["refusal_mean_hits"] > 0.0   # KNN always returns neighbors


def test_json_roundtrip(tmp_path):
    idx, key = _seed_corpus(tmp_path, deals=3, seed=7)
    report = harness.run(idx, key, modes=("bm25",))
    blob = json.dumps(report)
    assert json.loads(blob)["aggregate"]["bm25"]["exact_recall"] > 0


def test_answer_sample_is_stratified_and_deterministic():
    from context_wrangler.corpus import generate_corpus
    from evals.answers import _sample
    c = generate_corpus(deals=5, seed=3)
    s1 = _sample(c.questions, 10)
    s2 = _sample(c.questions, 10)
    assert [q["id"] for q in s1] == [q["id"] for q in s2]
    kinds = {q["kind"] for q in s1}
    assert {"exact", "refusal"} <= kinds
    assert len(s1) <= 12            # near the requested size, deterministic
