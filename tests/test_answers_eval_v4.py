"""v0.4 answer eval semantics, offline: runs, majority, instability,
metered cost passthrough, and rerank wiring. No model, no network: the
answer_query call is faked with scripted verdicts per question kind."""

import json

import pytest

import evals.answers as EA
from context_wrangler.core.embedder import HashEmbedder
from context_wrangler.core.ingest import ingest_dir
import context_wrangler.corpus as CG


@pytest.fixture
def corpus(tmp_path):
    out = str(tmp_path / "corpus")
    c = CG.generate_corpus(deals=3, seed=11)
    CG.write_corpus(out, c)
    idx = str(tmp_path / "idx")
    rep = ingest_dir(f"{out}/pdfs", idx, embedder=HashEmbedder(64))
    assert not rep.errors
    return idx, f"{out}/answer_key.json"


def _res(verdict, qq, cands):
    if verdict == "refused":
        return {"status": "refused", "gate": "model", "answer": "x",
                "citations": [], "attempts": 1, "gen_ms": 12.0,
                "context": {"used": 0, "capacity": 0, "blocks": 0,
                            "dropped": []},
                "usage": {"calls": 1, "prompt_tokens": 5,
                          "completion_tokens": 1, "cost_usd": 0.0}}
    if verdict == "bad":
        return {"status": "rejected", "answer": "", "citations": [],
                "reasons": ["grounding fail"], "attempts": 2, "gen_ms": 40.0,
                "context": {"used": 0, "capacity": 0, "blocks": 0,
                            "dropped": []},
                "usage": {"calls": 2, "prompt_tokens": 8,
                          "completion_tokens": 2, "cost_usd": None}}
    quote = qq.get("expected_text", "anything")
    return {"status": "ok", "gate": None, "answer": "a",
            "citations": [{"label": "c1", "quote": quote,
                           "doc_id": cands[0]["doc_id"], "span": [0, 5],
                           "filename": cands[0]["filename"]}],
            "attempts": 1, "gen_ms": 12.0,
            "context": {"used": 1, "capacity": 2, "blocks": 1, "dropped": []},
            "usage": {"calls": 1, "prompt_tokens": 5, "completion_tokens": 1,
                      "cost_usd": 0.0005}}


SCRIPT = {"exact": ["ok", "bad", "ok"], "table": ["ok", "ok", "ok"],
          "trap": ["ok", "ok", "ok"], "join": ["refused"] * 3,
          "refusal": ["refused"] * 3}


def test_majority_instability_and_metering(corpus, monkeypatch):
    idx, key = corpus
    real = json.load(open(key, encoding="utf-8"))
    chosen = EA._sample(real["questions"], 3)
    assert {q["kind"] for q in chosen} == set(SCRIPT), \
        "fixture changed: script assumes one question per kind"
    by_query = {q["query"]: q for q in chosen}

    def fake(question, cands, **kw):
        qq = by_query[question]
        return _res(SCRIPT[qq["kind"]].pop(0), qq, cands)

    monkeypatch.setattr(EA, "answer_query", fake)
    rep = EA.run_answers(idx, key, "ollama:qwen2.5:7b", limit=3,
                         mode="bm25", seed=42, runs=3)
    assert rep["runs"] == 3 and rep["seed"] == 42
    assert rep["publishable"]
    rows = {r["kind"]: r for r in rep["rows"]}
    assert [len(r["per_run"]) for r in rep["rows"]] == [3] * len(rep["rows"])
    ex = rows["exact"]
    assert ex["unstable"] is True and ex["correct"] is True   # 2/3 majority
    assert ex["correct_runs"] == "2/3"
    assert rows["trap"]["correct"] is True
    assert rows["trap"]["unstable"] is False
    assert rows["trap"].get("stale_cited") is not True
    assert rows["join"]["correct"] is False
    assert rows["refusal"]["correct"] is True     # refused == correct
    # one row metered cost None (rejected retries) -> no fake totals
    assert rep["cost"]["metered_all_rows"] is False
    md = EA.to_markdown(rep)
    assert "seed 42" in md and "3 run(s) per question" in md
    assert "unstable" in md


def test_all_metered_rows_publish_a_total(corpus, monkeypatch):
    idx, key = corpus
    real = json.load(open(key, encoding="utf-8"))
    chosen = EA._sample(real["questions"], 3)
    by_query = {q["query"]: q for q in chosen}

    def fake(question, cands, **kw):
        return _res("refused", by_query[question], cands)

    monkeypatch.setattr(EA, "answer_query", fake)
    rep = EA.run_answers(idx, key, "ollama:qwen2.5:7b", limit=3,
                         mode="bm25", seed=1, runs=2)
    assert rep["cost"]["metered_all_rows"] is True
    assert rep["cost"]["cost_usd_total"] == pytest.approx(0.0)
    assert "Metered cost" in EA.to_markdown(rep)


def test_seed_and_spec_reach_every_run(corpus, monkeypatch):
    idx, key = corpus
    calls = []

    def spy(question, cands, *, llm_spec, budget, canonical,
           keyword_cands, seed=None):
        calls.append((llm_spec, seed))
        return _res("refused", {"kind": "refusal"}, cands)

    monkeypatch.setattr(EA, "answer_query", spy)
    rep = EA.run_answers(idx, key, "openrouter:openai/gpt-4o-mini",
                         limit=2, mode="bm25", seed=123, runs=2)
    assert calls and all(c == ("openrouter:openai/gpt-4o-mini", 123)
                         for c in calls)
    assert len(calls) == len(rep["rows"]) * 2
    assert all(r["status"] == "refused" for r in rep["rows"])


def test_rerank_stub_marks_plumbing_and_records_info(corpus, monkeypatch):
    idx, key = corpus

    def fake(question, cands, **kw):
        return _res("refused", {"kind": "refusal"}, cands)

    monkeypatch.setattr(EA, "answer_query", fake)
    rep = EA.run_answers(idx, key, "ollama:qwen2.5:7b", limit=2,
                         mode="bm25", seed=7, runs=1,
                         rerank_spec="stub:lexical")
    assert rep["publishable"] is False        # plumbing reranker poisons tier
    assert rep["rerank"] == "stub:lexical"
    row = rep["rows"][0]
    assert row["rerank"]["enabled"] is True
    assert row["rerank"]["name"] == "stub:lexical"
    md = EA.to_markdown(rep)
    assert "plumbing tier" in md and "rerank stub:lexical" in md


def test_stub_model_never_publishable(corpus, monkeypatch):
    idx, key = corpus

    def fake(question, cands, **kw):
        return _res("refused", {"kind": "refusal"}, cands)

    monkeypatch.setattr(EA, "answer_query", fake)
    rep = EA.run_answers(idx, key, "stub:extract", limit=2, mode="bm25")
    assert rep["publishable"] is False
