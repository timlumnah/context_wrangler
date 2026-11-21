"""Ingest: hash-based incremental, canonical text files, page spans."""

import os

from context_wrangler.core.index import open_db, text_path
from context_wrangler.core.ingest import ingest_dir, sha256_file


def _doc_count(index_dir: str) -> int:
    conn = open_db(index_dir)
    n = conn.execute("SELECT COUNT(*) c FROM docs").fetchone()["c"]
    conn.close()
    return n


def test_fresh_ingest_adds_all(corpus):
    rep = ingest_dir(corpus["dir"], corpus["index"])
    assert len(rep.added) == 3
    assert not rep.errors
    assert _doc_count(corpus["index"]) == 3


def test_reingest_is_a_no_op(corpus):
    ingest_dir(corpus["dir"], corpus["index"])
    rep = ingest_dir(corpus["dir"], corpus["index"])
    assert not rep.added and not rep.updated and not rep.errors
    assert len(rep.unchanged) == 3


def test_canonical_text_file_matches_hash(corpus):
    rep = ingest_dir(corpus["dir"], corpus["index"])
    conn = open_db(corpus["index"])
    for r in conn.execute("SELECT doc_id, source_path FROM docs"):
        digest = sha256_file(r["source_path"])
        assert r["doc_id"] == digest
        with open(text_path(corpus["index"], digest), encoding="utf-8") as fh:
            body = fh.read()
        assert ("ARTICLE" in body or "SECTION" in body
                or "CLOSING STATEMENT" in body), digest
    conn.close()


def test_page_spans_tile_the_text(corpus):
    ingest_dir(corpus["dir"], corpus["index"])
    conn = open_db(corpus["index"])
    pa = conn.execute(
        "SELECT doc_id, n_pages, text_len FROM docs WHERE filename = ?",
        ("purchase_agreement.pdf",)).fetchone()
    assert pa["n_pages"] == 2
    pages = conn.execute(
        "SELECT start, end FROM pages WHERE doc_id = ? ORDER BY page_no",
        (pa["doc_id"],)).fetchall()
    text = open(text_path(corpus["index"], pa["doc_id"]),
                encoding="utf-8").read()
    assert len(text) == pa["text_len"]
    # consecutive pages separated by exactly one join newline
    assert pages[0][0] == 0
    assert pages[1][0] == pages[0][1] + 1
    assert pages[1][1] == len(text)
    conn.close()


def test_modified_file_is_updated(corpus):
    ingest_dir(corpus["dir"], corpus["index"])
    old_ids = _doc_ids(corpus["index"])
    with open(corpus["paths"]["addendum_one.pdf"], "ab") as fh:
        fh.write(b"\n")                        # byte change, new hash
    rep = ingest_dir(corpus["dir"], corpus["index"])
    assert len(rep.updated) == 1
    assert not rep.added
    new_ids = _doc_ids(corpus["index"])
    assert len(new_ids - old_ids) == 1        # replaced doc
    assert len(_doc_ids(corpus["index"])) == 3
    # orphaned canonical file for the old hash is gone
    conn = open_db(corpus["index"])
    for stale in old_ids - new_ids:
        assert not os.path.exists(text_path(corpus["index"], stale))
    conn.close()


def test_moved_file_repoints_not_reextracted(corpus):
    ingest_dir(corpus["dir"], corpus["index"])
    before = corpus["paths"]["closing_statement.pdf"]
    after = os.path.join(corpus["dir"], "renamed_closing.pdf")
    os.rename(before, after)
    rep = ingest_dir(corpus["dir"], corpus["index"])
    assert len(rep.moved) == 1
    assert not rep.added and not rep.removed
    conn = open_db(corpus["index"])
    n = conn.execute("SELECT COUNT(*) c FROM docs WHERE filename LIKE ?",
                     ("renamed_closing%",)).fetchone()["c"]
    conn.close()
    assert n == 1


def test_duplicate_content_is_reported(corpus):
    ingest_dir(corpus["dir"], corpus["index"])
    import shutil
    shutil.copy(corpus["paths"]["addendum_one.pdf"],
                os.path.join(corpus["dir"], "copy_of_addendum.pdf"))
    rep = ingest_dir(corpus["dir"], corpus["index"])
    assert len(rep.duplicates) == 1
    assert not rep.added


def test_deleted_file_is_removed(corpus):
    ingest_dir(corpus["dir"], corpus["index"])
    os.remove(corpus["paths"]["closing_statement.pdf"])
    rep = ingest_dir(corpus["dir"], corpus["index"])
    assert len(rep.removed) == 1
    assert _doc_count(corpus["index"]) == 2


def test_non_pdf_files_ignored(corpus):
    with open(os.path.join(corpus["dir"], "notes.txt"), "w") as fh:
        fh.write("not a pdf")
    rep = ingest_dir(corpus["dir"], corpus["index"])
    assert len(rep.added) == 3


def _doc_ids(index_dir: str) -> set:
    conn = open_db(index_dir)
    ids = {r["doc_id"] for r in conn.execute("SELECT doc_id FROM docs")}
    conn.close()
    return ids
