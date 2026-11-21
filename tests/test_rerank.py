"""Rerank stage: contract, fail-open, plumbing tier, and metering."""

import json

import pytest

from context_wrangler.core.rerank import (StubLexicalReranker,
                                          build_rerank_prompt, make_reranker,
                                          rerank, sanitize_order)


def _c(i, text):
    return {"doc_id": f"d{i}", "filename": f"f{i}.pdf", "chunk_id": i,
            "page": 1, "section": "S", "span": [0, len(text)],
            "snippet": text[:40], "text": text, "score": -i}


CANDS = [_c(0, "The earnest money deposit is $12,500 payable to escrow."),
         _c(1, "Lead paint disclosure required by federal law."),
         _c(2, "Roof shingles were replaced in spring by vendor nine.")]


# ------------------------------------------------------------------ the code rules

def test_sanitize_order_keeps_only_valid_slots():
    assert sanitize_order([2, 0], 3) == [2, 0, 1]
    assert sanitize_order([5, -1, "x", 1, 1, None], 4) == [1, 0, 2, 3]
    assert sanitize_order([], 2) == [0, 1]
    # a permutation in: identity out. vacuous input earns nothing.
    assert sanitize_order([0, 1, 2], 3) == [0, 1, 2]


class ScriptedLLM:
    name = "scripted"
    publishable = True
    is_model = True

    def __init__(self, replies):
        self.replies = list(replies)
        from context_wrangler.core.embedder import UsageMeter
        self.usage = UsageMeter()

    def generate(self, system, user):
        # meter per call like a real client whose object is reused: a
        # row that reported the cumulative total would double-count
        # every earlier row.
        self.usage.record(prompt=10, completion=2, cost=0.0002)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def _model(replies):
    from context_wrangler.core.rerank import _ModelReranker
    return _ModelReranker(ScriptedLLM(replies))


def test_model_rerank_reorders_and_tags_ranks():
    rr = _model([json.dumps({"order": [2, 0, 1]})])
    out, info = rerank(CANDS, "earnest money deposit", reranker=rr, n=3)
    assert [c["chunk_id"] for c in out] == [2, 0, 1]
    assert [c["rerank_rank"] for c in out] == [0, 1, 2]
    assert info["ok"] is True and info["moved"] == 3
    assert info["window"] == 3
    assert info["usage"]["cost_usd"] == pytest.approx(0.0002)
    assert "ms" in info


def test_model_rerank_fails_open_on_garbage():
    rr = _model(["not json at all", "still not json"])
    out, info = rerank(CANDS, "q", reranker=rr, n=3)
    assert [c["chunk_id"] for c in out] == [0, 1, 2]     # order untouched
    assert info["ok"] is False
    assert info["attempts"] == 2                          # it got the retry
    assert "invalid_json" in info["reasons"]


def test_model_rerank_partial_order_is_completed_not_invented():
    rr = _model([json.dumps({"order": [1]})])
    out, info = rerank(CANDS, "q", reranker=rr, n=3)
    assert [c["chunk_id"] for c in out] == [1, 0, 2]
    assert info["ok"] is True          # the omitted tail keeps its slots


def test_window_tail_boundary():
    many = CANDS * 2
    rr = _model([json.dumps({"order": [1, 0]})])
    out, info = rerank(many, "q", reranker=rr, n=2)
    assert info["window"] == 2
    assert [c["chunk_id"] for c in out[2:]] == [c["chunk_id"] for c in many[2:]]
    # tail records are untouched copies, no rerank_rank on them
    assert "rerank_rank" not in out[2]


# ------------------------------------------------------------------ plumbing tier

def test_stub_lexical_is_deterministic_and_plumbing():
    rr = StubLexicalReranker()
    order, info = rr.order("earnest money deposit escrow", CANDS)
    assert order[0] == 0                     # the deposit chunk wins
    assert info["ok"] and info["attempts"] == 0
    again, _ = rr.order("earnest money deposit escrow", CANDS)
    assert again == order
    assert not rr.publishable
    made = make_reranker("stub:lexical")
    assert made is not None and not made.publishable


def test_make_reranker_empty_is_none():
    assert make_reranker("") is None
    assert make_reranker("none") is None


def test_rerank_none_reranker_is_identity():
    out, info = rerank(CANDS, "q", reranker=None)
    assert out == CANDS and info == {"enabled": False}


def test_shared_reranker_client_meters_per_question_not_cumulative():
    rr = _model([json.dumps({"order": [1, 0, 2]})])   # one client, many rows
    _out1, i1 = rerank(CANDS, "q", reranker=rr, n=3)
    _out2, i2 = rerank(CANDS, "q", reranker=rr, n=3)
    for info in (i1, i2):
        assert info["usage"]["calls"] == 1
        assert info["usage"]["cost_usd"] == pytest.approx(0.0002)


def test_prompt_carries_indices_and_truncated_excerpts():
    long = _c(9, "word " * 400)
    prompt = build_rerank_prompt("what deposit?", [CANDS[0], long])
    assert "0: f0.pdf p1 [S]" in prompt
    assert "1: f9.pdf p1 [S]" in prompt
    assert "word word word" in prompt
    assert prompt.count("word " * 300) == 0   # excerpt is capped, not whole


def test_rerank_llm_spec_wired_through_make_reranker(monkeypatch):
    from context_wrangler.core import answer as A
    seen = {}

    def fake_make_llm(spec, *, seed=None, schema=None):
        seen["spec"], seen["seed"], seen["schema"] = spec, seed, schema
        return ScriptedLLM([json.dumps({"order": [1, 0, 2]})])

    monkeypatch.setattr(A, "make_llm", fake_make_llm)
    import context_wrangler.core.rerank as R
    monkeypatch.setattr(R, "make_llm", fake_make_llm)
    rr = make_reranker("openrouter:openai/gpt-4o-mini", seed=42)
    out, info = rerank(CANDS, "q", reranker=rr, n=3)
    assert seen["spec"] == "openrouter:openai/gpt-4o-mini"
    assert seen["seed"] == 42
    assert seen["schema"]["properties"]["order"]["items"]["type"] == "integer"
    assert info["ok"]
    assert [c["chunk_id"] for c in out] == [1, 0, 2]
