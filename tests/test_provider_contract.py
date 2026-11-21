"""Provider contract tests against a local fake server, key-free.

Everything in v0.4 that talks to a paid API is pinned here against a
stdlib http.server that speaks the OpenAI /chat/completions,
OpenAI /embeddings, and Ollama /api/chat shapes. These tests prove the
wire format (bearer header, usage.include, seed, constrained-format
payload) and the metering (cost lands in the UsageMeter). They do NOT
prove live-account behavior: that is what the keyed frontier run in
instructions.txt is for, and the docstrings say which is which.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import context_wrangler.core.answer as A
import context_wrangler.core.embedder as E

REQUESTS = []          # what the fake server saw
REPLY_MODE = {"chat": "canned", "embed": "canned", "ollama": "canned"}


def _chat_reply(payload):
    content = json.dumps({"answer": "not in these documents",
                          "refused": True, "citations": []})
    if REPLY_MODE["chat"] == "garbage":
        return 200, {"garbage": True}
    if REPLY_MODE["chat"] == "notjson":
        return 200, {"choices": [{"message": {"content": "plain prose"}}],
                     "usage": {"prompt_tokens": 10,
                               "completion_tokens": 2}}
    return 200, {"choices": [{"message": {"content": content}}],
                 "usage": {"prompt_tokens": 1234, "completion_tokens": 56,
                           "cost": 0.00042}}


def _embed_reply(payload):
    if REPLY_MODE["embed"] == "garbage":
        return 200, {"nope": 1}
    n = len(payload["input"])
    return 200, {"data": [{"index": i, "embedding": [0.1, 0.2, 0.3]}
                          for i in range(n)],
                 "usage": {"prompt_tokens": 7 * n, "cost": 1.4e-6 * n}}


def _ollama_reply(payload):
    if REPLY_MODE["ollama"] == "garbage":
        return 200, {"nope": 1}
    return 200, {"message": {"content": json.dumps(
        {"answer": "x", "refused": True, "citations": []})},
        "prompt_eval_count": 900, "eval_count": 21}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        REQUESTS.append({
            "path": self.path,
            "auth": self.headers.get("Authorization"),
            "body": body,
        })
        if self.path.endswith("/chat/completions"):
            code, reply = _chat_reply(body)
        elif self.path.endswith("/embeddings"):
            code, reply = _embed_reply(body)
        elif self.path.endswith("/api/chat"):
            code, reply = _ollama_reply(body)
        elif self.path.endswith("/api/embed"):
            code, reply = 200, {"embeddings": [[0.5] * 4
                                               for _ in body["input"]]}
        else:
            code, reply = 404, {"error": "no route"}
        data = json.dumps(reply).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):        # silence
        pass


@pytest.fixture(scope="module")
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture
def routed(fake_server, monkeypatch):
    REQUESTS.clear()
    monkeypatch.setenv("OPENROUTER_BASE_URL", fake_server + "/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    monkeypatch.setenv("OPENAI_BASE_URL", fake_server + "/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-oai-key")
    for mode in REPLY_MODE:
        REPLY_MODE[mode] = "canned"
    return fake_server


# ---------------------------------------------------------------- chat

def test_openrouter_chat_sends_key_usage_and_seed(routed):
    llm = A.OpenRouterChatLLM("openai/gpt-4o-mini", seed=42)
    out = llm.generate("sys", "user")
    assert json.loads(out)["refused"] is True
    req = REQUESTS[-1]
    assert req["auth"] == "Bearer test-or-key"
    assert req["body"]["usage"] == {"include": True}
    assert req["body"]["seed"] == 42
    assert req["body"]["response_format"] == {"type": "json_object"}
    assert llm.usage.cost_usd == pytest.approx(0.00042)
    assert llm.usage.prompt_tokens == 1234
    assert llm.usage.completion_tokens == 56
    assert llm.usage.calls == 1
    assert llm.name == "openrouter:openai/gpt-4o-mini"


def test_openrouter_chat_accumulates_across_calls(routed):
    llm = A.OpenRouterChatLLM("openai/gpt-4o-mini")
    llm.generate("s", "u")
    llm.generate("s", "u")
    assert llm.usage.cost_usd == pytest.approx(0.00084)
    assert llm.usage.calls == 2


def test_no_seed_no_seed_field(routed):
    llm = A.OpenRouterChatLLM("openai/gpt-4o-mini")
    llm.generate("s", "u")
    assert "seed" not in REQUESTS[-1]["body"]


def test_openai_chat_never_asks_or_priced(routed):
    llm = A.OpenAIChatLLM("gpt-4o-mini", seed=7)
    llm.generate("s", "u")
    body = REQUESTS[-1]["body"]
    assert "usage" not in body            # usage.include is OpenRouter-only
    assert body["seed"] == 7
    assert llm.usage.prompt_tokens == 1234


def test_malformed_reply_raises(routed):
    REPLY_MODE["chat"] = "garbage"
    llm = A.OpenRouterChatLLM("openai/gpt-4o-mini")
    with pytest.raises(KeyError):
        llm.generate("s", "u")


def test_prose_reply_survives_to_the_json_retry(routed):
    REPLY_MODE["chat"] = "notjson"
    llm = A.OpenRouterChatLLM("openai/gpt-4o-mini")
    out = llm.generate("s", "u")
    assert out == "plain prose"           # answer_query turns this into a retry


def test_missing_key_errors_are_specific(routed, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        A.OpenRouterChatLLM("x/y")
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        A.OpenAIChatLLM("gpt")


def test_make_llm_routes_openrouter(routed):
    llm = A.make_llm("openrouter:openai/gpt-4o-mini", seed=3)
    assert isinstance(llm, A.OpenRouterChatLLM)
    assert llm.seed == 3


# ---------------------------------------------------------------- embed

def test_openrouter_embed_probe_batch_and_meter(routed):
    emb = E.OpenRouterEmbedder("openai/text-embedding-3-small")
    assert emb.dim == 3
    vecs = emb.embed(["a", "b", "c", "d", "e"])      # 5 > nothing; one batch
    assert len(vecs) == 5 and vecs[0] == [0.1, 0.2, 0.3]
    assert emb.usage.cost_usd > 0.0
    assert emb.name == "openrouter:openai/text-embedding-3-small"
    assert REQUESTS[-1]["auth"] == "Bearer test-or-key"
    assert REQUESTS[-1]["body"]["usage"] == {"include": True}


def test_embed_batches_are_capped(routed):
    emb = E.OpenRouterEmbedder("openai/text-embedding-3-small")
    emb.embed([f"t{i}" for i in range(E.BATCH + 3)])
    assert len(REQUESTS) == 3              # probe + two batches


def test_embed_malformed_raises(routed):
    REPLY_MODE["embed"] = "garbage"
    with pytest.raises((KeyError, RuntimeError)):
        E.OpenRouterEmbedder("openai/text-embedding-3-small")


def test_make_embedder_routes_openrouter(routed):
    emb = E.make_embedder("openrouter:openai/text-embedding-3-small")
    assert isinstance(emb, E.OpenRouterEmbedder)


def test_missing_embed_key_errors(routed, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        E.OpenRouterEmbedder("x/y")


# ---------------------------------------------------------------- ollama wire

def test_ollama_chat_sends_schema_and_seed(routed):
    llm = A.OllamaLLM("qwen2.5:7b", url=routed, seed=17)
    out = llm.generate("s", "u")
    body = REQUESTS[-1]["body"]
    assert body["format"] == A.ANSWER_SCHEMA
    assert body["options"] == {"temperature": 0, "seed": 17}
    assert json.loads(out)["answer"] == "x"
    assert llm.usage.prompt_tokens == 900
    assert llm.usage.completion_tokens == 21
    assert llm.usage.cost_usd == 0.0


def test_ollama_rerank_schema_replacement(routed):
    from context_wrangler.core.rerank import RERANK_SCHEMA
    llm = A.OllamaLLM("qwen2.5:7b", url=routed, schema=RERANK_SCHEMA)
    llm.generate("s", "u")
    assert REQUESTS[-1]["body"]["format"] == RERANK_SCHEMA
