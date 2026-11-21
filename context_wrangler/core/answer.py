"""Grounded answering with a refusal gate. The v0.3 product step.

Contract, enforced in code (verify_grounding), never by asking the model
politely:
  - answer is strict JSON: {answer, refused, citations[]}
  - every citation points at a context block by label, copies the quote
    verbatim from that block's slice, and the span lands inside the block
  - a citation's quote must ALSO byte-match the canonical text file at
    those offsets, checked after the model stops answering
  - a non-refusal answer with zero citations fails: no citation, no answer
  - the refusal form is exact: answer == "not in these documents",
    citations == []
Failure twice in a row means the CLI reports the answer REJECTED with
reasons and a nonzero exit. It does not print fluent garbage.

Providers follow the open_otto pattern (provider:model):
  ollama:<model>   chat with the schema passed as `format` (constrained
                   output; open_otto measured this cuts JSON rejections
                   from most drafts to about 1 in 27)
  openai:<model>   any OpenAI-style server, OPENAI_API_KEY/OPENAI_BASE_URL
  openrouter:<model>
                   OpenRouter (OPENROUTER_API_KEY), model ids carry the
                   vendor prefix (openrouter:openai/gpt-4o-mini). Requests
                   ask for usage.include, so every call meters exact
                   dollars for the cost/quality frontier table.
  stub:extract     deterministic extractive answerer, NO model, no network.
                   Plumbing tier: it proves the pipeline and the gate work.
                   The eval harness refuses to publish it as answer quality.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request

from context_wrangler.core.assemble import Assembled, render_context
from context_wrangler.core.embedder import UsageMeter

REFUSAL = "not in these documents"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "refused": {"type": "boolean"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["label", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "refused", "citations"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You answer strictly from the numbered context blocks. Nothing else "
    "exists. Every factual claim in your answer must be backed by a "
    "citation. A citation names its block label (like c1) and copies a "
    "span of text verbatim from that block as the quote. Do not compute "
    "character offsets and do not paraphrase inside a quote: the tool "
    "derives the exact span from your quote. If the context cannot answer "
    f"the question, output refused true, answer exactly {REFUSAL!r}, and "
    "an empty citations list. Output JSON only.")


def _tokens(s: str) -> list[str]:
    return re.findall(r"[^\W_]+", s.lower(), re.UNICODE)


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _find_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Locate the model's quote inside the block text. Exact match first;
    then a whitespace-flexible match (LLMs flatten newlines into spaces
    constantly, and the bytes that SHIP are re-read from the file, so this
    is tolerance of transcription, never of content)."""
    pos = text.find(quote)
    if pos >= 0:
        return pos, pos + len(quote)
    pattern = r"\s+".join(re.escape(p) for p in quote.split())
    m = re.search(pattern, text) if pattern else None
    if m and m.group(0).strip():
        return m.start(), m.end()
    return None


# ------------------------------------------------------------------ providers

class LLM:
    name: str
    publishable: bool = True
    usage: UsageMeter

    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError


class StubExtractiveLLM(LLM):
    """No model. Picks the sentence with the most query-token overlap (only
    tokens of 4+ letters count, so stop words cannot fake a match) from the
    best-scoring block; quotes its exact span. If nothing shares at least 2
    content tokens with the question, it refuses. Deterministic, offline,
    and honest about being plumbing."""

    publishable = False

    def __init__(self, blocks: Assembled, question: str):
        # the stub needs the context; it is constructed per call by
        # answer_query, which knows the assembled blocks
        self._blocks = blocks
        self._question = question
        self.usage = UsageMeter()
        self.usage.cost_usd = 0.0

    def generate(self, system: str, user: str) -> str:
        qtoks = {t for t in _tokens(self._question) if len(t) >= 4}
        best = None
        for label, b in zip(self._blocks.labels, self._blocks.blocks):
            for sent in re.split(r"(?<=[.!?])\s+", b.text):
                if not sent.strip():
                    continue
                st = set(_tokens(sent)) & qtoks
                score = (len(st), -b.span[0])
                if len(st) >= 2 and (best is None or score > best[0]):
                    pos = b.text.find(sent)
                    best = (score, label, b, pos, sent)
        if best is None:
            return json.dumps({"answer": REFUSAL, "refused": True,
                               "citations": []})
        _score, label, b, pos, sent = best
        payload = {
            "answer": f"The records state: {sent}",
            "refused": False,
            "citations": [{"label": label, "quote": sent}],
        }
        return json.dumps(payload)


class OllamaLLM(LLM):
    publishable = True

    def __init__(self, model: str, url: str = OLLAMA_URL,
                 seed: int | None = None, schema: dict | None = None):
        self.model = model
        self.url = url.rstrip("/")
        self.seed = seed
        self.schema = schema or ANSWER_SCHEMA
        self.name = f"ollama:{model}"
        self.usage = UsageMeter()
        self.usage.cost_usd = 0.0      # local compute

    def generate(self, system: str, user: str) -> str:
        options: dict = {"temperature": 0}
        if self.seed is not None:
            # temperature 0 alone is not deterministic across runs; open_otto
            # and the v0.3 eval both hit this. An explicit seed pins it.
            options["seed"] = self.seed
        payload = {
            "model": self.model,
            "stream": False,
            "format": self.schema,
            "options": options,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        req = urllib.request.Request(
            f"{self.url}/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
        self.usage.record(prompt=body.get("prompt_eval_count") or 0,
                          completion=body.get("eval_count") or 0)
        return body["message"]["content"]


class OpenAIChatLLM(LLM):
    publishable = True
    BASE_ENV = "OPENAI_BASE_URL"
    BASE_DEFAULT = "https://api.openai.com/v1"
    KEY_ENV = "OPENAI_API_KEY"

    def __init__(self, model: str, seed: int | None = None):
        key = os.environ.get(self.KEY_ENV)
        if not key:
            raise RuntimeError(f"{self.KEY_ENV} is not set")
        self.model = model
        self.key = key
        self.seed = seed
        self.base = os.environ.get(self.BASE_ENV, self.BASE_DEFAULT).rstrip("/")
        self.name = f"{self._prefix()}:{model}"
        self.usage = UsageMeter()

    def _prefix(self) -> str:
        return "openai"

    def _payload(self, system: str, user: str) -> dict:
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content":
                          system + "\nReply with a single JSON object."},
                         {"role": "user", "content": user}],
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload

    def generate(self, system: str, user: str) -> str:
        req = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=json.dumps(self._payload(system, user)).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
        u = body.get("usage") or {}
        self.usage.record(prompt=u.get("prompt_tokens", 0),
                          completion=u.get("completion_tokens", 0),
                          cost=u.get("cost"))
        return body["choices"][0]["message"]["content"]


class OpenRouterChatLLM(OpenAIChatLLM):
    """OpenAI-compatible /chat/completions through OpenRouter. OpenRouter
    meters exact per-call dollars in usage.cost, which is what makes the
    cost/quality frontier a published measurement instead of an estimate.
    Seed support is upstream-dependent: the seed is sent, and the eval
    publishes measured instability (n_unstable), not a reproducibility
    claim."""

    BASE_ENV = "OPENROUTER_BASE_URL"
    BASE_DEFAULT = "https://openrouter.ai/api/v1"
    KEY_ENV = "OPENROUTER_API_KEY"

    def _prefix(self) -> str:
        return "openrouter"

    def _payload(self, system: str, user: str) -> dict:
        p = super()._payload(system, user)
        p["usage"] = {"include": True}   # exact dollars in usage.cost
        return p


def make_llm(spec: str, *, seed: int | None = None,
             schema: dict | None = None) -> LLM:
    if spec == "stub:extract":
        raise ValueError("stub:extract is constructed by answer_query with "
                         "the assembled context; do not build it here")
    provider, _, model = spec.partition(":")
    if provider == "ollama":
        return OllamaLLM(model, seed=seed, schema=schema)
    if provider == "openai":
        return OpenAIChatLLM(model, seed=seed)
    if provider == "openrouter":
        return OpenRouterChatLLM(model, seed=seed)
    raise ValueError(f"unknown llm provider in {spec!r} "
                     "(use ollama:<model>, openai:<model>, "
                     "openrouter:<model>, or stub:extract)")


# ------------------------------------------------------------------ the gate

def verify_grounding(payload, assembled: Assembled,
                     canonical_getter) -> list[str]:
    """Returns [] when the answer is admissible, else the reasons. Every
    check here is a hard rule from the README."""
    reasons: list[str] = []
    if not isinstance(payload, dict):
        return ["payload is not a JSON object"]
    for key in ("answer", "refused", "citations"):
        if key not in payload:
            reasons.append(f"missing key {key!r}")
    if reasons:
        return reasons
    answer, refused, citations = (payload["answer"], payload["refused"],
                                  payload["citations"])
    if not isinstance(answer, str) or not isinstance(refused, bool) \
            or not isinstance(citations, list):
        return ["wrong types: answer/refused/citations must be str/bool/list"]
    if refused:
        if citations:
            reasons.append("refusal must carry zero citations")
        if answer.strip() != REFUSAL:
            reasons.append(f"refusal answer must be exactly {REFUSAL!r}")
        return reasons

    # non-refusal: no citation, no answer
    if not citations:
        return ["answer cites nothing: refusal was the only admissible output"]

    for i, cit in enumerate(citations):
        if not isinstance(cit, dict) or not {"label", "quote"} <= set(cit):
            reasons.append(f"citation {i}: needs label and quote")
            continue
        block = assembled.block_by_label(str(cit["label"]))
        if block is None:
            reasons.append(f"citation {i}: label {cit['label']!r} is not a "
                           "context block (invented source)")
            continue
        if not str(cit["quote"]).strip():
            reasons.append(f"citation {i}: empty quote")
            continue
        found = _find_quote(block.text, str(cit["quote"]))
        if found is None:
            reasons.append(
                f"citation {i}: the quote is not verbatim text of block "
                f"{cit['label']}; the answer quotes a span that is not in "
                "the assembled context")
            continue
        s, e = found
        canon_text = canonical_getter(block.doc_id)
        if canon_text is None:
            reasons.append(f"citation {i}: canonical text for "
                           f"{block.doc_id[:12]} is missing; cannot "
                           "byte-check")
            continue
        if canon_text[block.span[0] + s: block.span[0] + e] != block.text[s:e]:
            reasons.append(f"citation {i}: block text disagrees with the "
                           "canonical file; byte-check failed (index bug)")
    return reasons


# ------------------------------------------------------------------ pipeline

def build_user_prompt(question: str, assembled: Assembled) -> str:
    return (f"Context blocks:\n\n{render_context(assembled)}\n\n"
            f"Question: {question}\n\n"
            'Answer with JSON: {"answer": str, "refused": bool, '
            '"citations": [{"label": str, "quote": str}]}, where each '
            "quote is copied verbatim from that block.")


def answer_query(question: str, candidates: list[dict], *,
                 llm_spec: str, budget: int = 2400,
                 per_doc_cap: int = 2, canonical: dict[str, str] | None = None,
                 attempts: int = 2,
                 keyword_cands: list[dict] | None = None,
                 seed: int | None = None) -> dict:
    """Full v0.3 loop: gate, assemble, prompt, verify, retry once, else
    reject. v0.4 adds seed (pinned sampling for published evals) and
    reports the llm's metered usage + wall time in the result.

    keyword_cands: the BM25 AND result for the same question, from
    retrieve(mode="bm25"). Refusal rule (hard, in code, applies BEFORE the
    model and no matter what it says): if the keyword path finds no chunk
    containing every content term, the question's identity does not occur
    in the corpus, and KNN neighbors are just the least-wrong documents.
    Hybrid always returns something; this is why refusal-over-fluency needs
    a keyword anchor, not model politeness. The cost is false refusals on
    heavy paraphrase: documented, and a smarter threshold is v0.4 work.

    Returns {status: ok|refused|rejected, answer, citations, reasons?,
    gate?, attempts, context: {used, capacity, dropped_n}}.
    """
    from context_wrangler.core.assemble import assemble

    canon = canonical or {}
    if keyword_cands is not None and not keyword_cands:
        return {"status": "refused", "gate": "no_keyword_support",
                "answer": REFUSAL, "citations": [],
                "context": {"used": 0, "capacity": 0, "blocks": 0,
                            "dropped": []},
                "usage": None, "gen_ms": 0.0}

    assembled = assemble(candidates, budget=budget, per_doc_cap=per_doc_cap,
                         canonical=canon)
    summary = {"used": assembled.used, "capacity": assembled.capacity,
               "blocks": len(assembled.blocks),
               "dropped": [(b.doc_id, list(b.span), r)
                           for b, r in assembled.dropped]}

    if not assembled.blocks:
        return {"status": "refused", "gate": "no_context",
                "answer": REFUSAL, "citations": [], "context": summary,
                "usage": None, "gen_ms": 0.0}

    if llm_spec == "stub:extract":
        llm = StubExtractiveLLM(assembled, question)
    else:
        llm = make_llm(llm_spec, seed=seed)
    meter = getattr(llm, "usage", None) or UsageMeter()

    getter = lambda doc_id: canon.get(doc_id) if doc_id in canon else None
    user = build_user_prompt(question, assembled)
    last_reasons: list[str] = []
    gen_ms = 0.0
    for attempt in range(1, attempts + 1):
        t0 = time.perf_counter()
        try:
            raw = llm.generate(SYSTEM_PROMPT, user)
        except Exception as exc:                       # noqa: BLE001
            return {"status": "rejected", "answer": "", "citations": [],
                    "reasons": [f"model unreachable: {exc}"],
                    "attempts": attempt, "context": summary,
                    "usage": meter.as_dict(), "gen_ms": round(gen_ms, 1)}
        gen_ms += (time.perf_counter() - t0) * 1000.0
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            last_reasons = ["invalid_json"]
            user = user + f"\n\nRETRY {attempt}: your previous reply was " \
                "not a valid JSON object. Reply with JSON only."
            continue
        reasons = verify_grounding(payload, assembled, getter)
        if not reasons:
            status = "refused" if payload["refused"] else "ok"
            enriched = []
            for cit in payload["citations"]:
                block = assembled.block_by_label(str(cit["label"]))
                s, e = _find_quote(block.text, str(cit["quote"]))
                enriched.append({"label": cit["label"],
                                 # server-derived span, server-shipped bytes
                                 "span": [block.span[0] + s,
                                          block.span[0] + e],
                                 "quote": block.text[s:e],
                                 "whitespace_drift":
                                     block.text[s:e] != cit["quote"],
                                 "doc_id": block.doc_id,
                                 "filename": block.filename})
            return {"status": status, "gate": "model" if status == "refused" else None,
                    "answer": payload["answer"],
                    "citations": enriched,
                    "attempts": attempt, "context": summary,
                    "usage": meter.as_dict(), "gen_ms": round(gen_ms, 1)}
        last_reasons = reasons
        user = user + ("\n\nRETRY: your previous answer failed grounding "
                       "checks: " + "; ".join(reasons) +
                       ". Cite verbatim from the blocks only, or refuse.")

    return {"status": "rejected", "answer": "", "citations": [],
            "reasons": last_reasons, "attempts": attempts,
            "context": summary,
            "usage": meter.as_dict(), "gen_ms": round(gen_ms, 1)}
