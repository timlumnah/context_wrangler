"""v0.5 table extraction: column-major damage, layout reflow, regime guard."""

import json

import pytest

from context_wrangler.corpus import Table, build_pdf_bytes, make_pdf
from context_wrangler.core.index import open_db
from context_wrangler.core.ingest import extract_pages, ingest_dir
from context_wrangler.core.search import search, verify_all
from context_wrangler.core.tables import _cells, page_text, reflow

ROWS = [
    ["Item", "Amount"],
    ["Title Insurance", "$410.00"],
    ["Transfer Tax", "$1,285.00"],
    ["Recording Fee", "$155.00"],
    ["Escrow Fee", "$350.00"],
    ["Reconveyance Fee", "$85.00"],
]


def _table_pdf(tmp_path, name="closing.pdf"):
    p = tmp_path / name
    p.write_bytes(build_pdf_bytes([["CLOSING STATEMENT", "",
                                    "SECTION 3 SETTLEMENT TABLE",
                                    Table(ROWS)]]))
    return str(p)


# ---------------------------------------------------------------- units

def test_cells_splits_on_double_space_keeps_word_space():
    line = "    Title Insurance        $410.00"
    cells = _cells(line)
    assert [t for _s, t in cells] == ["Title Insurance", "$410.00"]
    assert cells[1][0] - cells[0][0] > 5


def test_reflow_unit_on_hand_layout_text():
    layout = "\n".join([
        "CLOSING STATEMENT",
        "",
        "    Item                     Amount",
        "    Title Insurance          $410.00",
        "    Transfer Tax             $1,285.00",
        "    Recording Fee            $155.00",
        "",
        "Prose that carries on after the grid.",
    ])
    out = reflow(layout)
    assert out is not None
    assert "Title Insurance | $410.00" in out
    assert "Item | Amount" in out
    assert "Prose that carries on after the grid." in out
    assert "CLOSING STATEMENT" in out


def test_reflow_none_without_grid():
    prose = ("PURCHASE AGREEMENT\n\n"
             "Article one says the buyer pays the price and\n"
             "the seller delivers the deed at closing day.\n")
    assert reflow(prose) is None


def test_table_width_validation():
    with pytest.raises(ValueError):
        Table([["a", "b"], ["c"]])


def test_single_space_lines_are_not_grids():
    # one cell per line: never a region, so no false reflow
    assert reflow("Alpha\nBeta\nGamma\nDelta") is None


# ---------------------------------------------------------------- damage

def test_naive_extraction_loses_row_adjacency(tmp_path):
    from pypdf import PdfReader
    path = _table_pdf(tmp_path)
    plain = page_text(PdfReader(path).pages[0], tables=False)
    for label, amount in ROWS[1:]:
        same_line = [ln for ln in plain.splitlines()
                     if label in ln and amount in ln]
        assert not same_line, f"naive extraction still paired {label}"


def test_tables_flag_restores_rows(tmp_path):
    from pypdf import PdfReader
    path = _table_pdf(tmp_path)
    page = PdfReader(path).pages[0]
    reflowed = page_text(page, tables=True)
    for label, amount in ROWS:
        assert f"{label} | {amount}" in reflowed


def test_prose_pages_byte_identical_under_tables(tmp_path):
    from fixtures_pdf import PURCHASE_AGREEMENT
    path = tmp_path / "pa.pdf"
    make_pdf(str(path), PURCHASE_AGREEMENT)
    plain_pages, _ = extract_pages(str(path))
    reflowed_pages, _ = extract_pages(str(path), tables=True)
    assert plain_pages == reflowed_pages


# ---------------------------------------------------------------- ingest

def test_ingest_tables_search_and_verify(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    _table_pdf(tmp_path, name="closing.pdf")
    (pdf_dir / "closing.pdf").write_bytes(
        (tmp_path / "closing.pdf").read_bytes())
    idx = str(tmp_path / "idx")
    rep = ingest_dir(str(pdf_dir), idx, tables=True)
    assert not rep.errors and rep.added
    conn = open_db(idx)
    hits = search(conn, idx, "title insurance amount", k=5)
    assert any("| $410.00" in h["text"] for h in hits)
    v = verify_all(conn, idx)
    assert v["ok"], v["bad"]
    conn.close()


def test_ingest_tables_off_shows_the_damage(tmp_path):
    """Same file, tables off: the doc is findable, and label and figure
    often even land in the SAME chunk (contiguous lines), but the row
    pairing is gone at line level: no line and no reflowed row string
    binds them. That granularity gap is why table questions are graded
    on the pipe-row string, not on document hits."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    p = _table_pdf(tmp_path)
    (pdf_dir / "closing.pdf").write_bytes(open(p, "rb").read())
    idx = str(tmp_path / "idx_off")
    ingest_dir(str(pdf_dir), idx)
    conn = open_db(idx)
    hits = search(conn, idx, "title insurance amount", k=10)
    assert hits                                   # the doc is found...
    for h in hits:
        assert "Title Insurance | $410.00" not in h["text"]
        paired = [ln for ln in h["text"].splitlines()
                  if "Title Insurance" in ln and "$410.00" in ln]
        assert not paired
    conn.close()


def test_regime_stamps_and_guard(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "closing.pdf").write_bytes(
        build_pdf_bytes([["HDR", "", Table(ROWS)]]))
    idx = str(tmp_path / "idx_guard")
    ingest_dir(str(pdf_dir), idx, tables=True)
    conn = open_db(idx)
    stamps = dict(conn.execute("SELECT key, value FROM meta")).copy()
    conn.close()
    assert stamps["tables_config"] == "layout-reflow"
    # switching regimes on the same index is refused
    (pdf_dir / "second.pdf").write_bytes(
        build_pdf_bytes([["SECOND DOC", "", "Some body text here."]]))
    with pytest.raises(RuntimeError, match="table extraction"):
        ingest_dir(str(pdf_dir), idx, tables=False)
    # same regime again is fine
    rep = ingest_dir(str(pdf_dir), idx, tables=True)
    assert not rep.errors
    # and default (plain) is stamped on first use, refusing a later flip
    idx2 = str(tmp_path / "idx_plain")
    ingest_dir(str(pdf_dir), idx2)
    conn = open_db(idx2)
    assert dict(conn.execute("SELECT key, value FROM meta"))["tables_config"] \
        == "plain"
    conn.close()
    with pytest.raises(RuntimeError, match="table extraction"):
        ingest_dir(str(pdf_dir), idx2, tables=True)
