"""OCR stage: pages without a text layer -> text, via system binaries.

Subprocess calls to pdftoppm (poppler) and tesseract. No new python
dependencies, no models in process, no network. The stage is consulted by
ingest.py per page, and only when the page's own text layer is thinner
than TEXT_LAYER_MIN_CHARS: real text layers always win, OCR is what fires
on the scanned page that has none.

Truth contract: the canonical text of an OCR'd page IS the OCR output.
Citations byte-verify against the canonical file exactly as they always
do, so a span into a scanned doc is true to the bytes the tool indexed
(what tesseract read), not a promise about the glyphs on the page. That
scope is stated in the README, not glossed.

Config is measured, not guessed: dpi 250, lang eng, psm 6. On the demo
corpus rasterized at 250 dpi, prose OCR was byte-exact including figures
("$485,000", dates); tesseract's default psm 3 read a settlement table
column by column and split every label from its amount, while --psm 6
read it line by line and kept them paired. The config string is written
into index meta at first use and a re-ingest under a different tier is
refused, the same one-consistent-extraction guarantee the embedder guard
gives the vector space.

scanify_pdf lives here too: it turns a text PDF into a scanned-looking
one (rasterize each page to a gray bitmap, rebuild the PDF with nothing
but page images and no text layer). The v0.5 eval uses it to build the
scanned twin of the synthetic corpus: same documents, same planted facts
at the same page geometry, no text layer anywhere. That makes the cost
of OCR measurable end to end instead of describable.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import zlib

TEXT_LAYER_MIN_CHARS = 24   # a page text layer thinner than this is a scan
DEFAULT_DPI = 250
DEFAULT_LANG = "eng"
DEFAULT_PSM = "6"           # assume a uniform block: measures paired tables


class OcrError(RuntimeError):
    pass


def tools_available() -> bool:
    return bool(shutil.which("pdftoppm")) and bool(shutil.which("tesseract"))


class SystemOcr:
    name = "system"
    publishable = True

    def __init__(self, dpi: int = DEFAULT_DPI, lang: str = DEFAULT_LANG,
                 psm: str = DEFAULT_PSM):
        if not tools_available():
            raise OcrError(
                "the system OCR tier needs pdftoppm (poppler-utils) and "
                "tesseract on PATH; install them or drop --ocr")
        self.dpi, self.lang, self.psm = int(dpi), str(lang), str(psm)

    @property
    def config(self) -> str:
        return f"system:dpi={self.dpi},lang={self.lang},psm={self.psm}"

    def raster(self, pdf_path: str, page_no: int, out_base: str) -> str:
        """One page -> gray PGM (P5). Returns the written path."""
        subprocess.run(
            ["pdftoppm", "-gray", "-r", str(self.dpi),
             "-f", str(page_no + 1), "-l", str(page_no + 1),
             "-singlefile", pdf_path, out_base],
            check=True, capture_output=True)
        return out_base + ".pgm"

    def page_text(self, pdf_path: str, page_no: int) -> str:
        with tempfile.TemporaryDirectory(prefix="cw-ocr-") as tmp:
            pgm = self.raster(pdf_path, page_no, os.path.join(tmp, "p"))
            proc = subprocess.run(
                ["tesseract", pgm, "stdout", "-l", self.lang,
                 "--psm", self.psm],
                check=True, capture_output=True, text=True)
            return proc.stdout


class StubOcr:
    """Plumbing tier for pipeline tests: emits marker text so tests can
    prove the OCR path lands in canonical files, the index, and the byte
    checks. Publishable = False, same rule as stub:extract and test-hash:
    plumbing proves the loop, never a quality claim."""
    name = "stub"
    publishable = False

    def __init__(self, marker: str = "ocr-stub"):
        self.marker = marker

    @property
    def config(self) -> str:
        return f"stub:{self.marker}"

    def page_text(self, pdf_path: str, page_no: int) -> str:
        return f"[{self.marker} page {page_no + 1}]"


def make_ocr(spec: str):
    """'system' | 'system:300' | 'system:300:eng:6' | 'stub' | 'stub:MARK'.
    An empty spec is the caller's problem, not this function's."""
    if not spec:
        raise ValueError("make_ocr('') needs an actual spec")
    parts = spec.split(":")
    kind = parts[0]
    if kind == "system":
        dpi = int(parts[1]) if len(parts) > 1 else DEFAULT_DPI
        lang = parts[2] if len(parts) > 2 else DEFAULT_LANG
        psm = parts[3] if len(parts) > 3 else DEFAULT_PSM
        return SystemOcr(dpi=dpi, lang=lang, psm=psm)
    if kind == "stub":
        return StubOcr(marker=parts[1] if len(parts) > 1 else "ocr-stub")
    raise ValueError(f"unknown ocr spec {spec!r}; use "
                     "system[:dpi[:lang[:psm]]] or stub[:marker]")


# ---------------------------------------------------------------- scanifier

_P5_HEADER = re.compile(rb"P5\s+(?:#[^\n]*\n\s+)*(\d+)\s+(\d+)\s+(\d+)\s")


def _pgm_raw(data: bytes) -> tuple[int, int, bytes]:
    """Parse a P5 graymap. Anything else (P2 ascii, P5 with a maxval the
    embed path can't promise) refuses loudly rather than guessing."""
    m = _P5_HEADER.match(data)
    if not m:
        raise OcrError("expected a P5 graymap from pdftoppm -gray")
    w, h, maxval = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if maxval != 255:
        raise OcrError(f"only 8-bit graymaps embed cleanly, maxval={maxval}")
    return w, h, data[m.end():]


def _pdf_wrap(objects: dict[int, bytes], n_pages: int) -> bytes:
    """Standard one-catalog PDF skeleton around pre-built page objects."""
    objs = {1: b"<< /Type /Catalog /Pages 2 0 R >>",
            2: f"<< /Type /Pages /Kids [{' '.join(f'{3 + i} 0 R' for i in range(n_pages))}] "
               f"/Count {n_pages} >>".encode()}
    objs.update(objects)
    out = bytearray(b"%PDF-1.4\n")
    offs: dict[int, int] = {}
    for num in sorted(objs):
        offs[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objs[num] + b"\nendobj\n"
    xref = len(out)
    total = max(objs) + 1
    out += f"xref\n0 {total}\n".encode() + b"0000000000 65535 f \n"
    for num in sorted(objs):
        out += f"{offs[num]:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {total} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)


def scanify_pdf(src: str, dst: str, dpi: int = DEFAULT_DPI) -> dict:
    """Rasterize every page, rebuild the file as image-only PDFs: the
    simulated scan. Deterministic in the strong sense the corpus generator
    is judged by: same source bytes and same poppler build produce the
    same output bytes, so scanned doc hashes are stable across runs."""
    from pypdf import PdfReader              # lazy: BM25 path never imports
    ocr = SystemOcr(dpi=dpi)
    n = len(PdfReader(src).pages)
    objs: dict[int, bytes] = {}
    for i in range(n):
        page_obj, cont_obj, img_obj = 3 + i, 3 + n + i, 3 + 2 * n + i
        with tempfile.TemporaryDirectory(prefix="cw-scan-") as tmp:
            pgm = ocr.raster(src, i, os.path.join(tmp, "p"))
            with open(pgm, "rb") as fh:
                raw = fh.read()
        w, h, samples = _pgm_raw(raw)
        comp = zlib.compress(samples, 6)
        content = (f"q {w} 0 0 {h} 0 0 cm /Im0 Do Q").encode()
        objs[page_obj] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] "
            f"/Resources << /XObject << /Im0 {img_obj} 0 R >> >> "
            f"/Contents {cont_obj} 0 R >>").encode()
        objs[cont_obj] = (b"<< /Length " + str(len(content)).encode() +
                          b" >>\nstream\n" + content + b"\nendstream")
        objs[img_obj] = (
            f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
            f"/ColorSpace /DeviceGray /BitsPerComponent 8 "
            f"/Filter /FlateDecode /Length {len(comp)} >>\nstream\n").encode() \
            + comp + b"\nendstream"
    data = _pdf_wrap(objs, n)
    with open(dst, "wb") as fh:
        fh.write(data)
    return {"pages": n, "dpi": dpi, "bytes": len(data)}


def scanify_dir(src_dir: str, dst_dir: str,
                dpi: int = DEFAULT_DPI) -> list[tuple[str, dict]]:
    """Every *.pdf under src_dir -> scanned twin under dst_dir, same
    relative name. Returns [(filename, info)] sorted."""
    out = []
    for root, _dirs, files in os.walk(src_dir):
        for name in sorted(files):
            if not name.lower().endswith(".pdf"):
                continue
            srcp = os.path.join(root, name)
            rel = os.path.relpath(srcp, src_dir)
            dstp = os.path.join(dst_dir, rel)
            os.makedirs(os.path.dirname(dstp), exist_ok=True)
            out.append((rel, scanify_pdf(srcp, dstp, dpi=dpi)))
    return out
