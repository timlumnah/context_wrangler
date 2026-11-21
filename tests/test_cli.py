"""End-to-end CLI: the exact commands the README will promise."""

import json
import subprocess
import sys


def run_cw(args, cwd=None):
    proc = subprocess.run(
        [sys.executable, "-m", "context_wrangler"] + args,
        capture_output=True, text=True, cwd=cwd)
    return proc


def test_full_flow(corpus):
    idx = ["--index", corpus["index"]]
    ing = run_cw(["ingest", corpus["dir"], *idx])
    assert ing.returncode == 0, ing.stderr
    assert "added 3" in ing.stdout

    st = run_cw(["stats", *idx])
    assert st.returncode == 0
    assert "docs 3" in st.stdout
    assert "purchase_agreement.pdf" in st.stdout

    se = run_cw(["search", "earnest", "money", "deposit", *idx])
    assert se.returncode == 0
    assert "purchase_agreement.pdf" in se.stdout
    assert "span " in se.stdout

    js = run_cw(["search", "escrow", "--json", *idx])
    payload = json.loads(js.stdout)
    assert payload["results"], "json search returned nothing"
    first = payload["results"][0]
    for key in ("doc_id", "filename", "page", "span", "text", "score"):
        assert key in first

    # verify is the proof bar: byte-for-byte over every chunk
    vf = run_cw(["verify", *idx])
    assert vf.returncode == 0
    assert "PASS" in vf.stdout

    # empty index dir must still work; no results, no crash
    empty = run_cw(["search", "anything", "--index",
                    corpus["index"] + "_none"])
    assert empty.returncode == 0
    assert "no matches" in empty.stdout
