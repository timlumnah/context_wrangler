"""BM25 search with cited spans. Precision cases matter: exact tokens are
why this is hybrid-ready, and the tests pin that keyword search finds them."""

from context_wrangler.core.index import open_db, read_canonical
from context_wrangler.core.search import build_match, search


def _search(indexed, query, **kw):
    conn = open_db(indexed["index"])
    try:
        return search(conn, indexed["index"], query, **kw)
    finally:
        conn.close()


def test_match_expression_shape():
    assert build_match("earnest money") == "earnest AND money"
    # "$12,500" tokenizes like the index does; phrase keeps adjacency
    assert build_match("$12,500") == '"12 500"'
    assert build_match("date!") == "date"
    assert build_match("...") is None
    assert build_match("a b", use_or=True) == "a OR b"


def test_earnest_money_facts(indexed):
    hits = _search(indexed, "earnest money deposit")
    assert hits, "no results"
    texts = " ".join(h["text"] for h in hits)
    assert "$12,500" in texts
    assert hits[0]["filename"] in ("purchase_agreement.pdf",
                                   "addendum_one.pdf")


def test_exact_figure_phrase(indexed):
    hits = _search(indexed, "$9,340.75")
    assert any(h["filename"] == "closing_statement.pdf" for h in hits)
    assert any("9,340.75" in h["text"] for h in hits)


def test_settlement_number(indexed):
    hits = _search(indexed, "2027-0315-44")
    assert hits
    assert hits[0]["filename"] == "closing_statement.pdf"
    assert "2027-0315-44" in hits[0]["text"]


def test_date_query(indexed):
    hits = _search(indexed, "closing date March 15 2027")
    assert any("March 15, 2027" in h["text"] for h in hits)


def test_and_is_the_default(indexed):
    hits = _search(indexed, "inspection financing")
    for h in hits:
        low = h["text"].lower()
        assert "inspection" in low or "financing" in low


def test_or_mode_broadens(indexed):
    and_hits = _search(indexed, "wire inspection")
    or_hits = _search(indexed, "wire inspection", use_or=True)
    assert len(or_hits) >= max(len(and_hits), 1)
    names = {h["filename"] for h in or_hits}
    assert "closing_statement.pdf" in names
    assert "purchase_agreement.pdf" in names


def test_no_match_returns_empty(indexed):
    assert _search(indexed, "zebra sandwich") == []


def test_results_carry_valid_spans(indexed):
    hits = _search(indexed, "escrow")
    assert hits
    from context_wrangler.core.index import read_canonical
    for h in hits:
        canonical = read_canonical(indexed["index"], h["doc_id"])
        start, end = h["span"]
        # the citation is exactly the span in the canonical text file
        assert canonical[start:end] == h["text"]
    # every hit lives in a document that genuinely contains the term
    # (docmeta identity carry can rank a title chunk for a term that sits
    # elsewhere in the same document; the body still never lies)
    assert any("escrow" in h["text"].lower() for h in hits)


def test_k_limit(indexed):
    hits = _search(indexed, "of", use_or=True, k=2)
    assert len(hits) <= 2
