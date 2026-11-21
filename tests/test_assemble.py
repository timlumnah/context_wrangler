"""Budget allocator: merge rules, caps, drop reporting, label stability."""

from context_wrangler.core.assemble import (LABEL_OVERHEAD_TOKENS, Block,
                                             SCAFFOLD_TOKENS, assemble,
                                             render_context, tokens)


def cand(doc, s, e, text, section=None, fname=None):
    return {"doc_id": doc, "filename": fname or f"{doc}.pdf", "span": [s, e],
            "section": section, "text": text}


T1 = "The Earnest Money Deposit is $12,500, payable to the escrow agent. " * 4
T2 = "The Closing Date shall be March 15, 2027, unless extended. " * 4
CANON = {"d1": T1 + "\n" + T2}


def test_blocks_fit_and_report_costs():
    cands = [cand("d1", 0, len(T1), T1), cand("d1", len(T1) + 1,
                                              len(T1) + 1 + len(T2), T2)]
    res = assemble(cands, budget=SCAFFOLD_TOKENS + 99999, canonical=CANON)
    assert len(res.blocks) == 1          # touching spans merged via canonical
    b = res.blocks[0]
    assert b.span == (0, len(T1) + 1 + len(T2))
    assert b.text == CANON["d1"][b.span[0]:b.span[1]]   # merge kept bytes true
    assert res.used == sum(x.cost for x in res.blocks)


def test_budget_drops_reported_with_reason():
    cost = tokens(T1) + LABEL_OVERHEAD_TOKENS
    cands = [cand("d1", 0, len(T1), T1, fname="a.pdf"),
             cand("d2", 0, len(T2), T2, fname="b.pdf"),
             cand("d3", 0, len(T1), T1, fname="c.pdf"),
             cand("d4", 0, len(T1), T1, fname="d.pdf")]
    res = assemble(cands, budget=SCAFFOLD_TOKENS + 3 * cost + cost // 2)
    reasons = [r for _b, r in res.dropped]
    assert reasons.count("budget") == 1
    assert len(res.blocks) == 3
    cost_t2 = tokens(T2) + LABEL_OVERHEAD_TOKENS
    assert res.used == 2 * cost + cost_t2


def test_per_doc_cap():
    gap = "\n" * 50
    cands = [cand("d1", 0, len(T1), T1),
             cand("d1", len(T1) + 50, len(T1) + 50 + len(T2), T2)]
    res = assemble(cands, per_doc_cap=1, canonical=None)
    assert len(res.blocks) == 1
    assert res.dropped and res.dropped[0][1] == "per_doc_cap"


def test_too_small_dropped():
    res = assemble([cand("d1", 0, 8, "tiny")], budget=9999)
    assert not res.blocks
    assert res.dropped[0][1] == "too_small"


def test_merge_without_canonical_only_zero_gap():
    a = cand("d1", 0, 100, "x" * 100)
    touching = cand("d1", 100, 200, "y" * 100)
    res = assemble([a, touching], canonical=None)
    assert len(res.blocks) == 1 and res.blocks[0].span == (0, 200)
    assert res.blocks[0].text == "x" * 100 + "y" * 100

    gapped = cand("d1", 101, 201, "y" * 100)      # 1 char gap
    res2 = assemble([a, gapped], canonical=None)
    assert len(res2.blocks) == 2, "must not invent the gap byte"


def test_labels_and_render_stable():
    cands = [cand("d1", 0, len(T1), T1, section="ARTICLE 2"),
             cand("d2", 0, len(T2), T2, section="SECTION 1")]
    res = assemble(cands, budget=99999, canonical={})
    assert res.labels == ["c1", "c2"]
    assert res.block_by_label("c1") is res.blocks[0]
    assert res.block_by_label("c9") is None
    assert res.block_by_label("junk") is None
    rendered = render_context(res)
    assert "[c1] d1.pdf" in rendered and "ARTICLE 2" in rendered


def test_token_math_is_the_char_heuristic():
    assert tokens("x" * 8) == 2
    assert tokens("x" * 9) == 3
    assert tokens("") == 0
