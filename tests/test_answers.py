"""The refusal gate. These tests are the README claim: an answer quoting
a span that was not in the assembled context fails, a refusal is a
first-class output, and a model that cannot cite is rejected, not printed."""

import json

from context_wrangler.core.answer import (REFUSAL, StubExtractiveLLM,
                                          answer_query, verify_grounding)
from context_wrangler.core.assemble import assemble

CANON = {"d1": "ARTICLE 2 EARNEST MONEY\nThe Earnest Money Deposit is "
               "$12,500, payable to the escrow agent within 72 hours."}
CANDS = [{"doc_id": "d1", "filename": "d1.pdf", "section": "ARTICLE 2",
          "span": [0, len(CANON["d1"])], "text": CANON["d1"]}]

getter = lambda doc: CANON.get(doc)


def _assembled():
    return assemble(CANDS, budget=99999, canonical=CANON)


def _payload(quotes_and_labels):
    return {"answer": "The deposit is $12,500.", "refused": False,
            "citations": quotes_and_labels}


def test_valid_answer_passes():
    res = _assembled()
    b = res.blocks[0]
    cit = [{"label": "c1", "quote": b.text[26:76]}]
    assert verify_grounding(_payload(cit), res, getter) == []


def test_invented_label_fails():
    res = _assembled()
    b = res.blocks[0]
    cit = [{"label": "c7", "quote": b.text[:50]}]
    reasons = verify_grounding(_payload(cit), res, getter)
    assert reasons and "invented source" in reasons[0]


def test_quote_content_drift_fails_whitespace_drift_passes():
    res = _assembled()
    b = res.blocks[0]
    cit = [{"label": "c1",
            "quote": b.text[26:76].replace("12,500", "25,000")}]
    reasons = verify_grounding(_payload(cit), res, getter)
    assert any("not verbatim text" in r for r in reasons)

    mangled = b.text[0:60].replace("\n", " ")   # whitespace only
    assert verify_grounding(_payload([{"label": "c1", "quote": mangled}]),
                            res, getter) == []


def test_quote_from_nothing_in_context_fails():
    res = _assembled()
    cit = [{"label": "c1", "quote": "The deposit is a million dollars."}]
    reasons = verify_grounding(_payload(cit), res, getter)
    assert any("not verbatim text" in r for r in reasons)
    # empty quote: cannot resolve a span from nothing
    assert verify_grounding(_payload([{"label": "c1", "quote": "  "}]),
                            res, getter)


def test_whitespace_drift_is_flagged_and_server_bytes_ship():
    import context_wrangler.core.answer as A

    class DriftModel(A.LLM):
        name = "test:drift"

        def generate(self, system, user):
            text = CANDS[0]["text"][0:60].replace("\n", " ")
            return json.dumps({"answer": "The deposit is $12,500.",
                               "refused": False,
                               "citations": [{"label": "c1",
                                              "quote": text}]})

    orig = A.make_llm
    A.make_llm = lambda spec, **kw: DriftModel()
    try:
        res = answer_query("earnest money deposit", CANDS,
                           llm_spec="ollama:whatever", canonical=CANON)
    finally:
        A.make_llm = orig
    assert res["status"] == "ok"
    cit = res["citations"][0]
    assert cit["quote"] == CANON["d1"][0:60]       # the file's bytes ship
    assert cit["whitespace_drift"] is True         # drift is told, not hidden
    assert CANON["d1"][cit["span"][0]:cit["span"][1]] == cit["quote"]


def test_canonical_mismatch_fails_even_if_block_slice_matches():
    # an index bug could make block.text and the file disagree; the
    # byte-check catches it because it re-reads the file
    res = _assembled()
    b = res.blocks[0]
    tampered = dict(CANON)
    tampered["d1"] = CANON["d1"].replace("12,500", "13,500")
    cit = [{"label": "c1", "quote": b.text[26:76]}]
    reasons = verify_grounding(_payload(cit), res, lambda d: tampered[d])
    assert any("byte-check" in r for r in reasons)


def test_no_citation_no_answer():
    res = _assembled()
    reasons = verify_grounding({"answer": "confident nonsense",
                                "refused": False, "citations": []},
                               res, getter)
    assert any("cites nothing" in r for r in reasons)


def test_refusal_form_is_exact():
    res = _assembled()
    ok = {"answer": REFUSAL, "refused": True, "citations": []}
    assert verify_grounding(ok, res, getter) == []
    sloppy = {"answer": "I think it's not here", "refused": True,
              "citations": []}
    assert verify_grounding(sloppy, res, getter)
    padded = {"answer": REFUSAL, "refused": True,
              "citations": [{"label": "c1", "span": [0, 4], "quote": "ARTI"}]}
    assert verify_grounding(padded, res, getter)


def test_missing_keys_rejected():
    res = _assembled()
    assert verify_grounding({"answer": "x"}, res, getter)
    assert verify_grounding([1, 2], res, getter)


# ----------------------------------------------------- full loop, no network

def test_stub_answers_planted_fact(indexed):
    from context_wrangler.core.index import open_db, read_canonical
    from context_wrangler.core.retrieve import candidates
    conn = open_db(indexed["index"])
    cands = candidates(conn, indexed["index"], "earnest money deposit",
                       mode="bm25", k=8)
    canon = {c["doc_id"]: read_canonical(indexed["index"], c["doc_id"])
             for c in cands}
    conn.close()
    res = answer_query("earnest money deposit", cands,
                       llm_spec="stub:extract", canonical=canon)
    assert res["status"] == "ok", res
    assert any("$12,500" in c["quote"] or "$18,000" in c["quote"]
               for c in res["citations"])
    # every emitted citation carries a doc id and byte-checks
    for c in res["citations"]:
        assert c["doc_id"] in canon
        assert canon[c["doc_id"]][c["span"][0]:c["span"][1]] == c["quote"]


def test_ghost_question_refused_before_the_model(indexed):
    from context_wrangler.core.index import open_db
    from context_wrangler.core.retrieve import candidates
    conn = open_db(indexed["index"])
    cands = candidates(conn, indexed["index"],
                       "earnest money Zadvorsky 999 Wollaston",
                       mode="bm25", k=8)
    conn.close()
    assert cands == []
    res = answer_query("q", cands, llm_spec="stub:extract", canonical={})
    assert res["status"] == "refused" and res["gate"] == "no_context"
    assert res["answer"] == REFUSAL


def test_keyword_gate_refuses_even_when_vectors_answer():
    """The dangerous shape: hybrid hands back plausible chunks for a ghost
    identity. Zero BM25 support must refuse BEFORE any model sees the
    context."""
    fake_vec_cands = CANDS                      # looks fine, wrong deal
    res = answer_query("earnest money deposit Gronberg 639 Zircon",
                       fake_vec_cands, llm_spec="stub:extract",
                       canonical=CANON, keyword_cands=[])
    assert res["status"] == "refused"
    assert res["gate"] == "no_keyword_support"
    assert res["citations"] == []
    # and with support present, the same question is admissible
    res2 = answer_query("earnest money deposit", fake_vec_cands,
                        llm_spec="stub:extract", canonical=CANON,
                        keyword_cands=CANDS)
    assert res2["status"] == "ok"


def test_bad_model_output_is_rejected_not_printed():
    import context_wrangler.core.answer as A

    class BadModel(A.LLM):
        name = "test:bad"

        def generate(self, system, user):
            return json.dumps({"answer": "The deposit is $9,999,999.",
                               "refused": False,
                               "citations": [{"label": "c1",
                                              "quote": "TOTAL LIE"}]})

    orig = A.make_llm
    A.make_llm = lambda spec, **kw: BadModel()
    try:
        res = answer_query("earnest money deposit", CANDS,
                           llm_spec="ollama:whatever", canonical=CANON)
    finally:
        A.make_llm = orig
    assert res["status"] == "rejected"
    assert res["citations"] == []
    assert any("not verbatim text" in r for r in res["reasons"])
    assert res["attempts"] == 2       # it got the corrective retry


def test_unreachable_model_reports_error_status():
    res = answer_query("q", CANDS, llm_spec="ollama:nonexistent-model-xyz",
                       canonical=CANON)
    assert res["status"] == "rejected"
    assert any("model unreachable" in r for r in res["reasons"])
