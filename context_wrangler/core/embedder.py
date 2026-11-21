"""Bring-your-own embedders. Spec strings share the open_otto pattern:
provider:model.

    ollama:nomic-embed-text        local, free, needs `ollama pull`
    openai:text-embedding-3-small  API; OPENAI_API_KEY (+ OPENAI_BASE_URL
                                   for any OpenAI-style server)
    openrouter:openai/text-embedding-3-small
                                   API through OpenRouter; OPENROUTER_API_KEY.
                                   OpenRouter meters exact per-call cost in
                                   usage.cost, so the frontier table can
                                   publish dollars instead of estimates.
    test-hash                      deterministic feature hashing. No model,
                                   no network. PLUMBING ONLY: it measures
                                   vocabulary overlap, so it is valid for
                                       * proving the vector pipeline works
                                       * proving RRF math is right
                                   and NEVER valid as a quality tier in a
                                   published table. The eval harness refuses
                                   to emit it.

Everything here is stdlib urllib; requests is not a dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL", "https://api.openai.com/v1")

BATCH = 16
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


class UsageMeter:
    """Per-object spend accounting for the cost/quality frontier table.

    OpenRouter meters exact dollars in usage.cost, so API tiers publish
    measured money. Ollama publishes zero. A provider that returns tokens
    but no cost (raw OpenAI) publishes tokens with cost = None, and the
    frontier table refuses to guess the dollars: a missing number is
    honest, an invented one poisons the table."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd: float | None = None

    def record(self, *, prompt: int = 0, completion: int = 0,
               cost: float | None = None) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        if cost is not None:
            self.cost_usd = (self.cost_usd or 0.0) + cost

    def add(self, other: "UsageMeter") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        if other.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + other.cost_usd

    def snapshot(self) -> dict:
        return self.as_dict()

    @staticmethod
    def diff(prev: dict, now: dict) -> dict:
        """What THIS call spent, on an object whose meter accumulates
        across calls. Reporting the running total per row would
        double-count every earlier row; this is the only honest way to
        meter a shared client."""
        cost = None
        if now["cost_usd"] is not None:
            base = prev["cost_usd"] or 0.0
            cost = now["cost_usd"] - base
        return {"calls": now["calls"] - prev["calls"],
                "prompt_tokens": now["prompt_tokens"] - prev["prompt_tokens"],
                "completion_tokens": now["completion_tokens"]
                - prev["completion_tokens"],
                "cost_usd": cost}

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": self.cost_usd}


class Embedder:
    name: str
    dim: int
    publishable: bool = True      # may appear in a published results table
    usage: UsageMeter

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashEmbedder(Embedder):
    """Two md5-derived buckets per token, signed, L2-normalized. Deterministic
    across machines and runs."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"test-hash:{dim}"
        self.publishable = False
        self.usage = UsageMeter()
        self.usage.cost_usd = 0.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            self.usage.record()
            v = [0.0] * self.dim
            for tok in _TOKEN.findall(t.lower()):
                d = hashlib.md5(tok.encode()).digest()
                h1 = int.from_bytes(d[:4], "big")
                h2 = int.from_bytes(d[4:8], "big")
                v[h1 % self.dim] += 1.0 if h1 & 1 else -1.0
                v[h2 % self.dim] += 1.0 if h2 & 1 else -1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class OllamaEmbedder(Embedder):
    publishable = True

    def __init__(self, model: str, url: str = OLLAMA_URL):
        self.model = model
        self.url = url.rstrip("/")
        self.name = f"ollama:{model}"
        self.usage = UsageMeter()
        self.usage.cost_usd = 0.0      # local compute: the frontier tier is $0
        self.dim = len(self._post("/api/embed",
                                  {"model": model, "input": ["probe"]})[0])

    def _post(self, path: str, payload: dict) -> list[list[float]]:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        self.usage.record(prompt=body.get("prompt_eval_count") or 0)
        if "embeddings" in body:
            return body["embeddings"]
        if "embedding" in body:                   # legacy single endpoint
            return [body["embedding"]]
        raise RuntimeError(f"unexpected ollama reply: {list(body)[:4]}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            vecs = self._post("/api/embed",
                              {"model": self.model, "input": batch})
            if len(vecs) != len(batch):
                raise RuntimeError("ollama returned a short batch")
            out.extend(vecs)
        return out


class OpenAIEmbedder(Embedder):
    publishable = True
    BASE_ENV = "OPENAI_BASE_URL"
    BASE_DEFAULT = "https://api.openai.com/v1"

    def __init__(self, model: str):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = model
        self.key = key
        self.base = os.environ.get(self.BASE_ENV, self.BASE_DEFAULT).rstrip("/")
        self.name = f"openai:{model}"
        self.usage = UsageMeter()
        self.dim = len(self._post(["probe"])[0])

    def _payload(self, input_batch: list[str]) -> dict:
        return {"model": self.model, "input": input_batch}

    def _post(self, input_batch: list[str]) -> list[list[float]]:
        req = urllib.request.Request(
            f"{self.base}/embeddings",
            data=json.dumps(self._payload(input_batch)).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        u = data.get("usage") or {}
        self.usage.record(prompt=u.get("prompt_tokens", 0),
                          cost=u.get("cost"))
        rows = sorted(data["data"], key=lambda r: r["index"])
        return [r["embedding"] for r in rows]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            out.extend(self._post(texts[i:i + BATCH]))
        return out


class OpenRouterEmbedder(OpenAIEmbedder):
    """OpenAI-shaped /embeddings on OpenRouter, with the exact per-call
    dollars from usage.cost. Model ids are OpenRouter ids, which carry the
    vendor prefix: openrouter:openai/text-embedding-3-small."""

    BASE_ENV = "OPENROUTER_BASE_URL"
    BASE_DEFAULT = "https://openrouter.ai/api/v1"

    def __init__(self, model: str):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.model = model
        self.key = key
        self.base = os.environ.get(self.BASE_ENV, self.BASE_DEFAULT).rstrip("/")
        self.name = f"openrouter:{model}"
        self.usage = UsageMeter()
        self.dim = len(self._post(["probe"])[0])

    def _payload(self, input_batch: list[str]) -> dict:
        p = super()._payload(input_batch)
        p["usage"] = {"include": True}    # exact dollars in usage.cost
        return p


def make_embedder(spec: str) -> Embedder:
    """Parse provider:model. test-hash optionally takes a dim: test-hash:512.
    NOTE: ollama model names themselves contain no colon; openrouter model
    ids contain a slash, not a colon, so the split-on-first-colon rule holds
    for every provider here."""
    if spec == "test-hash" or spec.startswith("test-hash:"):
        dim = int(spec.split(":")[1]) if ":" in spec else 256
        return HashEmbedder(dim)
    if ":" not in spec:
        raise ValueError(
            f"embedder spec must be provider:model, got {spec!r} "
            "(e.g. ollama:nomic-embed-text, openai:text-embedding-3-small, "
            "test-hash)")
    provider, model = spec.split(":", 1)
    if provider == "ollama":
        return OllamaEmbedder(model)
    if provider == "openai":
        return OpenAIEmbedder(model)
    if provider == "openrouter":
        return OpenRouterEmbedder(model)
    raise ValueError(f"unknown embedder provider {provider!r}")
