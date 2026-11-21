"""The corpus generator: determinism, and an answer key that survives a
walk through the real extraction pipeline, not just the generator's
in-memory view."""

import hashlib
import os

from context_wrangler.corpus import (generate_corpus, write_corpus,
                                     _verify_corpus)
from context_wrangler.core.index import open_db, text_path
from context_wrangler.core.ingest import ingest_dir
from context_wrangler.core.search import search


def test_same_seed_same_bytes():
    a = generate_corpus(deals=4, seed=11)
    b = generate_corpus(deals=4, seed=11)
    assert sorted(a.pdfs) == sorted(b.pdfs)
    for fname in a.pdfs:
        assert hashlib.sha256(a.pdfs[fname]).hexdigest() == \
               hashlib.sha256(b.pdfs[fname]).hexdigest() == a.doc_ids[fname]


def test_different_seed_different_corpus():
    a = generate_corpus(deals=4, seed=11)
    b = generate_corpus(deals=4, seed=12)
    assert a.pdfs != b.pdfs


def test_key_survives_extraction(tmp_path):
    """The strongest claim the evals make: every planted fact is findable
    in the canonical text the retriever actually sees, under the doc id the
    index stores. If extraction mangles a line, this fails before any score
    is published."""
    c = generate_corpus(deals=5, seed=42)
    out = str(tmp_path / "corpus")
    write_corpus(out, c)
    rep = ingest_dir(os.path.join(out, "pdfs"), str(tmp_path / "idx"))
    assert not rep.errors and len(rep.added) == len(c.pdfs)
    conn = open_db(str(tmp_path / "idx"))
    ids = {r["doc_id"] for r in conn.execute("SELECT doc_id FROM docs")}
    conn.close()
    assert ids == set(c.doc_ids.values())
    for qq in c.questions:
        if qq["kind"] in ("exact", "trap"):
            doc_id = qq["expected_docs"][0]
            with open(text_path(str(tmp_path / "idx"), doc_id),
                      encoding="utf-8") as fh:
                assert qq["expected_text"] in fh.read(), qq["id"]
        elif qq["kind"] == "join":
            assert len(qq["expected_docs"]) == 2


def test_traps_and_refusals_exist_in_a_normal_run():
    c = generate_corpus(deals=10, seed=42)
    kinds = [q["kind"] for q in c.questions]
    assert kinds.count("trap") >= 1
    assert kinds.count("refusal") == 12
    # trap queries must not leak which document wins: no "current", no
    # "amended", just the bare fact phrase plus deal identity
    for q in c.questions:
        if q["kind"] == "trap":
            words = q["query"].split()
            assert "current" not in words and "amended" not in words
            head = " ".join(words[:3]).lower()
            assert head in ("earnest money deposit",
                            f"closing date {words[2]}".lower())


def test_refusal_tokens_absent_from_every_canonical_text(tmp_path):
    c = generate_corpus(deals=3, seed=5)
    out = str(tmp_path / "corpus")
    write_corpus(out, c)
    ingest_dir(os.path.join(out, "pdfs"), str(tmp_path / "idx"))
    conn = open_db(str(tmp_path / "idx"))
    blobs = []
    for r in conn.execute("SELECT doc_id FROM docs"):
        with open(text_path(str(tmp_path / "idx"), r["doc_id"]),
                  encoding="utf-8") as fh:
            blobs.append(fh.read())
    conn.close()
    joined = "\n".join(blobs)
    for q in c.questions:
        if q["kind"] == "refusal":
            for tok in q["absent_tokens"]:
                assert tok not in joined


def test_verify_catches_a_tampered_plant():
    c = generate_corpus(deals=2, seed=9)
    # break the key the way a labeling bug would: claim a fact nobody planted
    bad = dict(c.questions[0])
    bad["expected_text"] = "Earnest Money Deposit is $999,999,999"
    c.questions.append(bad)
    try:
        _verify_corpus(c)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "a planted fact missing from its PDF must fail loudly"


def test_bm25_recovers_planted_facts(tmp_path):
    c = generate_corpus(deals=5, seed=42)
    out = str(tmp_path / "corpus")
    write_corpus(out, c)
    idx = str(tmp_path / "idx")
    ingest_dir(os.path.join(out, "pdfs"), idx)
    conn = open_db(idx)
    exact = [q for q in c.questions if q["kind"] == "exact"]
    for qq in exact:
        hits = search(conn, idx, qq["query"], k=10)
        assert hits, qq["query"]
        assert any(h["doc_id"] in qq["expected_docs"] for h in hits), qq["id"]
        assert any(qq["expected_text"] in h["text"] for h in hits), qq["id"]
    conn.close()


def test_write_corpus_reports_orphans(tmp_path):
    """Regenerating after an rng-stream change renames files; leftovers
    must be reported loudly, never indexed silently into a new key."""
    c = generate_corpus(deals=2, seed=3)
    out = str(tmp_path / "corpus")
    assert write_corpus(out, c) == []
    stale = os.path.join(out, "pdfs", "d999_ghost_purchase_agreement.pdf")
    with open(stale, "wb") as fh:
        fh.write(b"%PDF-1.4 fake\n")
    orphans = write_corpus(out, c)
    assert orphans == ["d999_ghost_purchase_agreement.pdf"]


def test_corpus_table_questions_present():
    c = generate_corpus(deals=2, seed=3)
    tables = [q for q in c.questions if q["kind"] == "table"]
    assert len(tables) == 4                     # 2 per deal
    for qq in tables:
        assert " | " in qq["expected_text"]     # the reflowed row shape
        assert qq["table_cells"][0] in qq["expected_text"]
        assert qq["table_cells"][1] in qq["expected_text"]
