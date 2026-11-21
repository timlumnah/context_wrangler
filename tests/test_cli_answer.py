"""CLI surface of v0.3: assemble and answer, run as subprocesses with the
plumbing model so nothing here touches a network."""

import json
import subprocess
import sys


def run_cw(args):
    return subprocess.run([sys.executable, "-m", "context_wrangler"] + args,
                          capture_output=True, text=True)


def test_assemble_cli_reports_drops(indexed):
    idx = ["--index", indexed["index"]]
    p = run_cw(["assemble", "earnest", "money", "deposit", *idx,
                "--mode", "bm25"])
    assert p.returncode == 0, p.stderr
    assert "capacity" in p.stdout
    assert "[c1]" in p.stdout

    tight = run_cw(["assemble", "earnest", "money", "deposit", *idx,
                    "--mode", "bm25", "--budget", "820", "--json"])
    assert tight.returncode == 0, tight.stderr
    payload = json.loads(tight.stdout)
    # capacity is 820 - 800 scaffold = 20 tokens: at least one drop, honest
    assert payload["dropped"], payload
    assert all(d["reason"] in ("budget", "per_doc_cap", "too_small")
               for d in payload["dropped"])
    assert payload["used"] <= payload["capacity"]


def test_answer_cli_stub_and_no_model(indexed):
    idx = ["--index", indexed["index"]]
    p = run_cw(["answer", "earnest", "money", "deposit", *idx,
                "--model", "stub:extract", "--mode", "bm25", "--json"])
    assert p.returncode == 0, p.stderr
    res = json.loads(p.stdout)
    assert res["status"] == "ok"
    assert res["citations"]
    for c in res["citations"]:
        assert {"label", "span", "quote", "doc_id"} <= set(c)

    no_model = run_cw(["answer", "anything", *idx])
    assert no_model.returncode == 2
    assert "CW_MODEL" in no_model.stderr

    ghost = run_cw(["answer", "earnest", "Zadvorsky", "Wollaston", *idx,
                    "--model", "stub:extract", "--mode", "bm25"])
    assert ghost.returncode == 0
    assert "not in these documents" in ghost.stdout
    assert "no_keyword_support" in ghost.stdout
