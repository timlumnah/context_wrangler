"""v0.5 OCR: tier plumbing, threshold, regime guard, and the real
pdftoppm+tesseract roundtrip (skipped when the binaries are absent)."""

import os
import shutil
import subprocess

import pytest

from context_wrangler.corpus import build_pdf_bytes, make_pdf
from context_wrangler.core.index import open_db
from context_wrangler.core.ingest import extract_pages, ingest_dir
from context_wrangler.core.ocr import (OcrError, StubOcr, SystemOcr,
                                       TEXT_LAYER_MIN_CHARS, make_ocr,
                                       scanify_pdf, tools_available)
from context_wrangler.core.search import search, verify_all

PROSE = ["PURCHASE AND SALE AGREEMENT", "",
         "The Earnest Money Deposit is $12,500, payable to escrow."]
BLANK = [""]


def _pdf(tmp_path, name, pages):
    p = tmp_path / name
    p.write_bytes(build_pdf_bytes(pages))
    return str(p)


# ---------------------------------------------------------------- tier

def test_make_ocr_stub():
    o = make_ocr("stub:MK")
    assert isinstance(o, StubOcr) and not o.publishable
    assert o.page_text("whatever", 0) == "[MK page 1]"
    assert o.config == "stub:MK"


def test_make_ocr_system_requires_binaries(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _n: None)
    with pytest.raises(OcrError):
        make_ocr("system")
    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/thing")
    o = make_ocr("system:300:eng:6")
    assert isinstance(o, SystemOcr)
    assert o.config == "system:dpi=300,lang=eng,psm=6"


def test_make_ocr_rejects_junk():
    with pytest.raises(ValueError):
        make_ocr("")
    with pytest.raises(ValueError):
        make_ocr("google:vision")


# ------------------------------------------------- threshold + plumbing

def test_ocr_fires_only_on_thin_text_layers(tmp_path):
    path = _pdf(tmp_path, "mix.pdf", [PROSE, BLANK, ["$0"]])
    pages, n_ocr = extract_pages(path, ocr=StubOcr(marker="QX"))
    assert n_ocr == 2                       # blank + near-blank, prose kept
    assert "QX" not in pages[0] and "Earnest" in pages[0]
    assert "[QX page 2]" in pages[1]
    assert "[QX page 3]" in pages[2]
    # without the tier the thin pages stay thin
    pages0, n0 = extract_pages(path)
    assert n0 == 0 and pages0[0] == pages[0]


def test_ingest_records_ocr_pages(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    src = _pdf(tmp_path, "mix.pdf", [PROSE, BLANK])
    shutil.copy(src, pdf_dir / "mix.pdf")
    idx = str(tmp_path / "idx")
    rep = ingest_dir(str(pdf_dir), idx, ocr=StubOcr())
    assert rep.ocr_pages == 1
    conn = open_db(idx)
    row = conn.execute("SELECT ocr_pages FROM docs").fetchone()
    assert row["ocr_pages"] == 1
    stamps = dict(conn.execute("SELECT key, value FROM meta"))
    conn.close()
    assert stamps["ocr_config"] == "stub:ocr-stub"


def test_ocr_regime_guard(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "a.pdf").write_bytes(build_pdf_bytes([PROSE]))
    idx = str(tmp_path / "idx")
    ingest_dir(str(pdf_dir), idx, ocr=StubOcr(marker="ONE"))
    (pdf_dir / "b.pdf").write_bytes(build_pdf_bytes([PROSE]))
    with pytest.raises(RuntimeError, match="ocr tier"):
        ingest_dir(str(pdf_dir), idx)                    # regime dropped
    with pytest.raises(RuntimeError, match="ocr tier"):
        ingest_dir(str(pdf_dir), idx, ocr=StubOcr(marker="TWO"))
    rep = ingest_dir(str(pdf_dir), idx, ocr=StubOcr(marker="ONE"))
    assert not rep.errors


def test_plumbing_tier_never_publishable():
    assert StubOcr().publishable is False
    assert SystemOcr.__init__ is not None  # class-level claim:
    assert getattr(SystemOcr, "publishable", False) is True


# ------------------------------------------------------- real roundtrip

needs_tools = pytest.mark.skipif(
    not tools_available(),
    reason="needs pdftoppm (poppler-utils) and tesseract on PATH")


@needs_tools
def test_scanify_strips_the_text_layer(tmp_path):
    from pypdf import PdfReader
    src = _pdf(tmp_path, "t.pdf", [PROSE, BLANK])
    dst = str(tmp_path / "scanned.pdf")
    info = scanify_pdf(src, dst)
    assert info["pages"] == 2
    r = PdfReader(dst)
    assert all((pg.extract_text() or "").strip() == "" for pg in r.pages)
    # deterministic: the same input bytes rebuild the same file bytes
    dst2 = str(tmp_path / "scanned2.pdf")
    scanify_pdf(src, dst2)
    assert open(dst, "rb").read() == open(dst2, "rb").read()


@needs_tools
def test_ocr_roundtrip_reads_back_the_facts(tmp_path):
    src = _pdf(tmp_path, "t.pdf", [PROSE])
    dst = str(tmp_path / "scanned.pdf")
    scanify_pdf(src, dst)
    ocr = SystemOcr()
    pages, n = extract_pages(dst, ocr=ocr)
    assert n == 1
    text = pages[0]
    for needle in ["PURCHASE AND SALE AGREEMENT",
                   "Earnest Money Deposit is $12,500"]:
        assert needle in text, f"OCR roundtrip lost {needle!r}"


@needs_tools
def test_ingest_scanned_search_and_verify(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    src = _pdf(tmp_path, "t.pdf", [PROSE])
    scanify_pdf(src, str(pdf_dir / "t.pdf"))
    idx = str(tmp_path / "idx_scan")
    rep = ingest_dir(str(pdf_dir), idx, ocr=SystemOcr())
    assert rep.ocr_pages == 1
    conn = open_db(idx)
    hits = search(conn, idx, "$12,500", k=5)
    assert hits and "Earnest" in hits[0]["text"]
    assert verify_all(conn, idx)["ok"]
    stamps = dict(conn.execute("SELECT key, value FROM meta"))
    conn.close()
    assert stamps["ocr_config"].startswith("system:dpi=")
