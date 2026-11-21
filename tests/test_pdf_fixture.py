"""PDF writer sanity: the hand-built bytes must genuinely read back."""

from fixtures_pdf import make_pdf


def test_minimal_pdf_roundtrips(tmp_path):
    from pypdf import PdfReader
    p = str(tmp_path / "x.pdf")
    make_pdf(p, [["Hello world", "", "Second paragraph"]])
    text = PdfReader(p).pages[0].extract_text()
    assert "Hello world" in text
    assert "Second paragraph" in text


def test_pages_in_order(tmp_path):
    from pypdf import PdfReader
    p = str(tmp_path / "y.pdf")
    make_pdf(p, [["PAGE ONE MARKER"], ["PAGE TWO MARKER"]])
    reader = PdfReader(p)
    assert len(reader.pages) == 2
    assert "PAGE ONE MARKER" in reader.pages[0].extract_text()
    assert "PAGE TWO MARKER" in reader.pages[1].extract_text()


def test_special_chars_escaped(tmp_path):
    from pypdf import PdfReader
    p = str(tmp_path / "z.pdf")
    make_pdf(p, [["Deposit (see Exhibit A) $12,500 \\ done"]])
    text = PdfReader(p).pages[0].extract_text()
    assert "(see Exhibit A)" in text
    assert "\\ done" in text
