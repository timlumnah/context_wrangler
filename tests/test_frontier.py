"""Frontier table rules, offline and synthetic: the money column is
metered or it is a dash, tiers must describe the index they actually ran
on, and plumbing never publishes."""

import pytest

from evals import frontier as F


def _report(model="openrouter:openai/gpt-4o-mini", publishable=True,
            cost_usd=0.024, metered=True, n=3):
    rows = [{"kind": "exact", "correct": True, "unstable": False, "status": "ok",
             "gen_ms": 700.0 + i,
             "cost": {"calls": 1, "prompt_tokens": 100,
                      "completion_tokens": 10,
                      "cost_usd": None if cost_usd is None else cost_usd / n,
                      "metered": metered}} for i in range(n)]
    rows[1]["correct"] = False
    rows[2]["unstable"] = True
    return {"publishable": publishable, "sampled": n, "runs": 3, "seed": 42,
            "model": model, "rows": rows,
            "cost": {"metered_all_rows": metered,
                    "cost_usd_total": cost_usd,
                    "calls": n, "prompt_tokens": 300,
                    "completion_tokens": 30}}


def test_tier_row_majority_and_money():
    r = F._tier_row("t", "idx", "openrouter:openai/text-embedding-3-small",
                    "openrouter:openai/gpt-4o-mini",
                    _report(), ingest_cost=0.004, embed_delta=0.012)
    assert r["correct_majority"] == 2
    assert r["n_unstable"] == 1
    # (0.024 chat + 0.012 embed) / 3 questions * 1000 = $12.00
    assert r["dollars_per_1k_queries"] == pytest.approx(12.0)
    assert r["ingest_cost_usd"] == 0.004
    md = F.to_markdown({"run_utc": "x", "mode": "hybrid", "seed": 42,
                        "runs": 3, "limit": 3, "rerank": None,
                        "rows": [r]})
    assert "| $12.0000 |" in md
    assert "$0.0040" in md


def test_unmetered_money_shows_dash_never_a_guess():
    r = F._tier_row("t", "idx", "openai:some-embed", "openai:gpt-4o-mini",
                    _report(model="openai:gpt-4o-mini",
                            cost_usd=None, metered=False), None, None)
    assert r["dollars_per_1k_queries"] is None
    md = F.to_markdown({"run_utc": "x", "mode": "hybrid", "seed": 42,
                        "runs": 3, "limit": 3, "rerank": None, "rows": [r]})
    assert "| - |" in md


def test_plumbing_tier_refused():
    with pytest.raises(RuntimeError, match="plumbing"):
        F._tier_row("t", "idx", "ollama:nomic-embed-text", "stub:extract",
                    _report(model="stub:extract", publishable=False), None, 0.0)


def test_local_tier_dollars_are_zero_a_fact_not_a_gap():
    rep = _report(model="ollama:qwen2.5:7b", cost_usd=0.0, metered=True)
    for row in rep["rows"]:
        row["cost"]["cost_usd"] = 0.0
    r = F._tier_row("t", "idx", "ollama:nomic-embed-text", "ollama:qwen2.5:7b",
                    rep, 0.0, 0.0)
    assert r["dollars_per_1k_queries"] == pytest.approx(0.0)


def test_unpriced_query_embed_voids_the_dollar_column():
    """Rule under test: local chat is a metered $0, but if the query-time
    embedder does not report dollars, the total is unknowable, and the
    table shows a dash instead of a confident wrong number."""
    rep = _report(model="ollama:qwen2.5:7b", cost_usd=0.0, metered=True)
    for row in rep["rows"]:
        row["cost"]["cost_usd"] = 0.0
    r = F._tier_row("t", "idx", "openai:text-embedding-3-small",
                    "ollama:qwen2.5:7b", rep, None, None)
    assert r["dollars_per_1k_queries"] is None


def test_local_embed_delta_gap_stays_zero():
    """ollama embed with a missing snapshot delta is still $0: local
    compute is a fact, not a measurement gap."""
    rep = _report(model="ollama:qwen2.5:7b", cost_usd=0.0, metered=True)
    for row in rep["rows"]:
        row["cost"]["cost_usd"] = 0.0
    r = F._tier_row("t", "idx", "ollama:nomic-embed-text", "ollama:qwen2.5:7b",
                    rep, 0.0, None)
    assert r["dollars_per_1k_queries"] == pytest.approx(0.0)


def test_frontier_requires_matching_stack(tmp_path):
    """run_frontier must refuse a tier whose index was not embedded with
    the embedder the table claims: a mismatch is a fabricated row. The
    refusal happens BEFORE any model is constructed."""
    from context_wrangler.core.embedder import HashEmbedder
    from context_wrangler.core.ingest import ingest_dir
    import context_wrangler.corpus as CG
    out = str(tmp_path / "corpus")
    idx = str(tmp_path / "idx")
    c = CG.generate_corpus(deals=2, seed=3)
    CG.write_corpus(out, c)
    ingest_dir(f"{out}/pdfs", idx, embedder=HashEmbedder(64))
    key = f"{out}/answer_key.json"
    tiers = ({"id": "fake", "label": "fake", "index": idx,
              "embed": "openrouter:some/model",
              "chat": "ollama:qwen2.5:7b"},)
    with pytest.raises(RuntimeError, match="real stacks"):
        F.run_frontier(key, tiers, limit=1, runs=1)


def test_resolve_tiers_filters_and_refuses_unknown():
    allt = F.resolve_tiers(None)
    assert [t["id"] for t in allt] == ["local", "api-embed", "api-cheap"]
    sub = F.resolve_tiers(["api-embed", "api-cheap"])   # the CI subset
    assert [t["id"] for t in sub] == ["api-embed", "api-cheap"]
    with pytest.raises(ValueError, match="unknown tier"):
        F.resolve_tiers(["gpu-rental"])
