"""Grounding: spans are the contract. If the index and the canonical text
file ever disagree, verify() must fail loudly, not quietly."""

from context_wrangler.core.index import open_db, text_path
from context_wrangler.core.search import verify_all


def _db(index_dir):
    return open_db(index_dir)


def test_verify_passes_after_clean_ingest(indexed):
    conn = _db(indexed["index"])
    res = verify_all(conn, indexed["index"])
    n_before = res["checked"]
    conn.close()
    assert res["ok"]
    assert n_before > 0
    assert res["bad"] == []


def test_verify_catches_corrupted_mirror(indexed):
    conn = _db(indexed["index"])
    victim = conn.execute("SELECT rowid FROM chunks_fts LIMIT 1").fetchone()
    conn.execute("UPDATE chunks_fts SET body = body || ' INVENTED' WHERE rowid = ?",
                 (victim["rowid"],))
    conn.commit()
    res = verify_all(conn, indexed["index"])
    conn.close()
    assert not res["ok"]
    assert len(res["bad"]) == 1
    assert res["bad"][0]["chunk_id"] == victim["rowid"]


def test_verify_catches_drifted_span(indexed):
    conn = _db(indexed["index"])
    victim = conn.execute("SELECT chunk_id FROM chunks LIMIT 1").fetchone()
    # span shifted by 5 chars without touching the mirror text
    conn.execute(
        "UPDATE chunks SET start = start + 5 WHERE chunk_id = ?",
        (victim["chunk_id"],))
    conn.commit()
    res = verify_all(conn, indexed["index"])
    conn.close()
    assert not res["ok"]


def test_missing_canonical_file_fails_loud(indexed):
    conn = _db(indexed["index"])
    doc = conn.execute("SELECT doc_id FROM docs LIMIT 1").fetchone()
    import os
    os.remove(text_path(indexed["index"], doc["doc_id"]))
    try:
        verify_all(conn, indexed["index"])
        raised = False
    except FileNotFoundError:
        raised = True
    conn.close()
    assert raised, "a missing source file must not verify silently"
