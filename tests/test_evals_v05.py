"""v0.5 eval surface: table kind grading and the scanned-corpus map."""

import json
import os

import pytest

from context_wrangler.corpus import generate_corpus, write_corpus
from context_wrangler.core.index import open_db
from context_wrangler.core.ingest import ingest_dir
from context_wrangler.core.ocr import SystemOcr, scanify_pdf, tools_available
from evals.answers import SAMPLE_BY_KIND, _grade
from evals.harness import run, to_markdown

needs_tools = pytest.mark.skipif(
    not tools_available(),
    reason="needs pdftoppm (poppler-utils) and tesseract on PATH")


def _tiny_corpus(tmp_path, deals=2, seed=7):
    c = generate_corpus(deals=deals, seed=seed)
    out = str(tmp_path / "corpus")
    write_corpus(out, c)
    return c, out


def test_answer_grading_table_kind():
    qq = {"kind": "table", "expected_text": "Escrow Fee | $350"}
    ok = {"status": "ok", "citations":
          [{"quote": "Escrow Fee | $350", "doc_id": "x"}]}
    wrong = {"status": "ok", "citations": [{"quote": "Escrow Fee", "doc_id": "x"}]}
    refused = {"status": "refused", "citations": []}
    assert _grade(qq, ok)["correct"]
    assert not _grade(qq, wrong)["correct"]
    assert not _grade(qq, refused)["correct"]
    assert "table" in SAMPLE_BY_KIND


def test_scan_map_refuses_wrong_corpus(tmp_path):
    """A scanned index of unrelated files must not silently score zero;
    the translator raises when a key document has no namesake."""
    from pypdf import PdfReader
    _c, out = _tiny_corpus(tmp_path)
    pdf_dir = tmp_path / "other_pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "unrelated_deal.pdf").write_bytes(
        __import__("context_wrangler.corpus", fromlist=["build_pdf_bytes"])
        .build_pdf_bytes([["UNRELATED", "", "Nothing here matters."]]))
    idx = str(tmp_path / "idx_other")
    ingest_dir(str(pdf_dir), idx)
    with pytest.raises(RuntimeError, match="scan map"):
        run(idx, os.path.join(out, "answer_key.json"), modes=("bm25",),
            scan_map=True)


@needs_tools
def test_scanned_twin_scores_under_the_map(tmp_path):
    c, out = _tiny_corpus(tmp_path)
    text_idx = str(tmp_path / "idx_text")
    rep = ingest_dir(os.path.join(out, "pdfs"), text_idx, tables=True)
    assert rep.added
    scan_dir = tmp_path / "scanned_pdfs"
    scan_dir.mkdir()
    for name in os.listdir(os.path.join(out, "pdfs")):
        scanify_pdf(os.path.join(out, "pdfs", name),
                    str(scan_dir / name))
    scan_idx = str(tmp_path / "idx_scan")
    rep2 = ingest_dir(str(scan_dir), scan_idx, ocr=SystemOcr())
    assert rep2.ocr_pages > 0
    report = run(scan_idx, os.path.join(out, "answer_key.json"),
                 modes=("bm25",), scan_map=True)
    assert report["scan_map"] is True
    agg = report["aggregate"]["bm25"]
    assert "exact_recall" in agg and "table_recall" in agg
    md = to_markdown(report)
    assert "table row@10" in md
    assert "image-only twins" in md
    # the text index under the same map is a no-op translation:
    report2 = run(text_idx, os.path.join(out, "answer_key.json"),
                  modes=("bm25",), scan_map=True)
    a2 = report2["aggregate"]["bm25"]
    assert a2["table_row_top10"] == pytest.approx(1.0, abs=1e-9)
