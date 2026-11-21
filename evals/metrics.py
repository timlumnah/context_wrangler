"""Metric functions plus selftest.

The checks are plain code, not an LLM judge: repeatable and free to rerun,
same discipline as open_otto's evals. --selftest feeds each metric known
lists with known answers, including the vacuous ones. A metric that hands
out credit for an empty result set is a broken metric, and selftest says so
before any baseline number is trusted.
"""

from __future__ import annotations


def recall_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs present in the first k retrieval slots."""
    if not relevant:
        raise ValueError("recall on an empty relevant set is undefined; "
                         "refusal questions are scored elsewhere")
    seen = set(ranked_doc_ids[:k]) & relevant
    return len(seen) / len(relevant)


def reciprocal_rank(ranked_doc_ids: list[str], relevant: set[str]) -> float:
    for i, did in enumerate(ranked_doc_ids, start=1):
        if did in relevant:
            return 1.0 / i
    return 0.0


def top1_correct(ranked_doc_ids: list[str], relevant: set[str]) -> bool:
    return bool(ranked_doc_ids) and ranked_doc_ids[0] in relevant


def superseded_in_top_n(ranked_doc_ids: list[str], superseded: set[str],
                        n: int = 3) -> bool:
    """Did the stale document show up inside the top n? For trap questions
    this is the naive-citation rate the README promises to make visible."""
    return any(did in superseded for did in ranked_doc_ids[:n])


def selftest() -> list[tuple[str, bool]]:
    """Return [(check, passed)]; every check has a hand-computed answer."""
    out: list[tuple[str, bool]] = []

    def name(label: str, cond: bool) -> None:
        out.append((label, cond))

    name("recall perfect",
         recall_at_k(["a", "b"], {"a", "b"}, 10) == 1.0)
    name("recall partial",
         recall_at_k(["a", "x", "b"], {"a", "b"}, 10) == 1.0)
    name("recall windowed",
         recall_at_k(["x", "b", "a"], {"a", "b"}, 1) == 0.0)
    name("recall half",
         recall_at_k(["x", "a"], {"a", "b"}, 10) == 0.5)
    name("mrr first hit",
         reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5)
    name("mrr perfect",
         reciprocal_rank(["a"], {"a"}) == 1.0)
    name("mrr miss is zero",
         reciprocal_rank(["x", "y"], {"a"}) == 0.0)
    name("top1 yes", top1_correct(["a", "b"], {"a"}))
    name("top1 no", not top1_correct(["b", "a"], {"a"}))
    name("top1 empty is False, not an error", not top1_correct([], {"a"}))
    name("vacuous retrieval earns no recall",
         recall_at_k([], {"a", "b"}, 10) == 0.0)
    name("empty relevant set raises instead of defaulting to 1.0",
         _raises(lambda: recall_at_k(["a"], set(), 10)))
    name("superseded detected",
         superseded_in_top_n(["pa", "x"], {"pa"}, 3))
    name("superseded outside window ignored",
         not superseded_in_top_n(["x", "y", "z", "pa"], {"pa"}, 3))
    return out


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except ValueError:
        return True
