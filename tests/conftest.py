import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import pytest


@pytest.fixture
def corpus(tmp_path):
    """Fixture PDF corpus + a fresh index dir. Returns dict of paths."""
    from fixtures_pdf import write_corpus
    pdf_dir = tmp_path / "pdfs"
    paths = write_corpus(str(pdf_dir))
    return {"dir": str(pdf_dir), "index": str(tmp_path / "cw_index"),
            "paths": paths}


@pytest.fixture
def indexed(corpus):
    from context_wrangler.core.ingest import ingest_dir
    rep = ingest_dir(corpus["dir"], corpus["index"])
    assert not rep.errors, rep.errors
    corpus["report"] = rep
    return corpus
