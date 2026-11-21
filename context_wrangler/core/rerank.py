"""Optional rerank stage behind --rerank (v0.4). The pipeline step the
README spec'd and v0.3 left unbuilt: after fusion, before the budget
allocator, a model reorders the top-N candidates by answer-likelihood.

The house contract holds: the model emits DATA (an order list), the code
decides. Order entries are validated (integers inside the window,
deduplicated), anything the model omits keeps its original slot at the
end of the window, and a malformed reply after one retry leaves the
ranking UNTOUCHED with the failure recorded. A reranker can reorder the
shortlist; it can never invent a candidate, and it can never be the
reason a bad chunk enters the context.

Latency and cost are measured per call and published, never hidden:
the frontier table's rerank row carries rerank_ms and metered dollars.

Tiers:
    <any llm spec>        ollama:*, openai:*, openrouter:* rerankers.
                          The cheap API tier (openrouter:openai/gpt-4o-mini)
                          is what the published numbers use.
    stub:lexical          deterministic token-overlap sort. No model, no
                          network. PLUMBING ONLY, same discipline as the
                          extractive stub and the hash embedder: it proves
                          the wiring and the harness refuses to publish it.
"""

from __future__ import annotations

import json
import re
import time

from context_wrangler.core.answer import LLM, _tokens, make_llm
from context_wrangler.core.embedder import UsageMeter

EXCERPT_CHARS = 240

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "order": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["order"],
    "additionalProperties": False,
}

RERANK_SYSTEM = (
    "You are ranking retrieval candidates for a document question. You "
    "output strict JSON only: {\"order\": [ints]}, a permutation of every "
    "index from 0 to N-1, best first. Best means: the candidate whose text "
    "is most likely to contain the exact fact that answers the question. "
    "Judge only the excerpt you were given; excerpt truncation is not a "
    "reason to demote a candidate that already shows the answer.")


def _excerpt(text: str) -> str:
    return re.sub(r"\s+", " ", text[:EXCERPT_CHARS]).strip()


def build_rerank_prompt(question: str, window: list[dict]) -> str:
    lines = [f"Question: {question}", "", f"Candidates (N={len(window)}):"]
    for i, c in enumerate(window):
        section = f" [{c['section']}]" if c.get("section") else ""
        lines.append(f"{i}: {c['filename']} p{c['page']}{section} "
                     f"| {_excerpt(c['text'])}")
    lines.append("")
    lines.append('Reply JSON: {"order": [i, j, ...]} using every index '
                 f"exactly once, 0..{len(window) - 1}.")
    return "\n".join(lines)


def sanitize_order(raw: list, n: int) -> list[int]:
    """Keep valid in-range ints, dedupe, append the omitted ones in
    original order. Returns a full permutation of 0..n-1."""
    seen: set[int] = set()
    order: list[int] = []
    for v in raw:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in seen:
            seen.add(i)
            order.append(i)
    order.extend(i for i in range(n) if i not in seen)
    return order


class StubLexicalReranker:
    """Plumbing tier: stable sort by content-token overlap with the
    question (4+ letter tokens only, same rule as the extractive stub)."""

    name = "stub:lexical"
    publishable = False
    is_model = False

    def __init__(self) -> None:
        self.usage = UsageMeter()
        self.usage.cost_usd = 0.0

    def order(self, question: str, window: list[dict]) -> tuple[list[int], dict]:
        qtoks = {t for t in _tokens(question) if len(t) >= 4}
        def score(c: dict) -> tuple[int, int]:
            ctoks = set(_tokens(_excerpt(c["text"])))
            return (-len(ctoks & qtoks), c["chunk_id"])
        t0 = time.perf_counter()
        order = sorted(range(len(window)), key=lambda i: score(window[i]))
        zero = UsageMeter()
        zero.cost_usd = 0.0
        return order, {"ok": True, "attempts": 0, "reasons": [],
                      "usage": zero.as_dict(),
                      "ms": round((time.perf_counter() - t0) * 1000.0, 1)}


def make_reranker(spec: str, *, seed: int | None = None):
    """Returns an object with .order(question, window) plus usage metering,
    or None for no rerank."""
    if spec in ("", "none"):
        return None
    if spec == "stub:lexical":
        return StubLexicalReranker()
    llm = make_llm(spec, seed=seed, schema=RERANK_SCHEMA)
    return _ModelReranker(llm)


class _ModelReranker:
    is_model = True
    publishable = True

    def __init__(self, llm: LLM):
        self.llm = llm
        self.name = llm.name

    @property
    def usage(self) -> UsageMeter:
        return self.llm.usage

    def order(self, question: str, window: list[dict]) -> tuple[list[int], dict]:
        user = build_rerank_prompt(question, window)
        info: dict = {"attempts": 0, "ok": False, "reasons": []}
        prev = self.llm.usage.snapshot()
        t0 = time.perf_counter()
        for attempt in (1, 2):
            info["attempts"] = attempt
            try:
                raw = self.llm.generate(RERANK_SYSTEM, user)
            except Exception as exc:                        # noqa: BLE001
                info["reasons"] = [f"model unreachable: {exc}"]
                info["billed_unknown"] = True   # request may have been
                break                           # charged before dying
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                info["reasons"] = ["invalid_json"]
                continue
            if not isinstance(payload, dict) or not isinstance(
                    payload.get("order"), list):
                info["reasons"] = ["reply has no integer 'order' list"]
                continue
            info["ok"] = True
            info["reasons"] = []
            order = sanitize_order(payload["order"], len(window))
            info["ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            info["usage"] = UsageMeter.diff(prev, self.llm.usage.snapshot())
            return order, info
        info["ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        info["usage"] = UsageMeter.diff(prev, self.llm.usage.snapshot())
        if info.get("billed_unknown"):
            info["usage"]["cost_usd"] = None
        return list(range(len(window))), info      # fail open, untouched


def rerank(cands: list[dict], question: str, *, reranker,
           n: int = 12) -> tuple[list[dict], dict]:
    """Reorder the top-n window with the reranker; the tail keeps its
    place after the window. Returns (new_cands, info) where info carries
    ms, metered usage, and what actually moved."""
    if reranker is None or not cands:
        return cands, {"enabled": False}
    info: dict = {"enabled": True, "name": reranker.name,
                  "publishable": reranker.publishable, "window": min(n, len(cands))}
    window, tail = cands[:info["window"]], cands[info["window"]:]
    order, minfo = reranker.order(question, window)
    info.update(minfo)     # includes per-call ms/usage; cumulative totals
    # belong to the client object, never to this row
    new_window = [dict(window[i], rerank_rank=r) for r, i in enumerate(order)]
    moved = sum(1 for r, i in enumerate(order) if r != i)
    info["moved"] = moved
    return new_window + tail, info
