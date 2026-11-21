"""Embedder contract tests. Network-free: only the test-hash tier runs here.
The ollama/openai tiers get exercised in the live demo run, not CI-style."""

import math

from context_wrangler.core.embedder import HashEmbedder, make_embedder


def test_deterministic_and_normalized():
    e = HashEmbedder(dim=128)
    a = e.embed(["the earnest money deposit is $12,500"])[0]
    b = e.embed(["the earnest money deposit is $12,500"])[0]
    assert a == b
    assert abs(math.sqrt(sum(x * x for x in a)) - 1.0) < 1e-6
    assert len(a) == 128


def test_vocabulary_overlap_scores_closer_than_unrelated():
    e = HashEmbedder()
    q = e.embed_one("earnest money deposit escrow")
    near = e.embed_one("the earnest money deposit is payable to escrow")
    far = e.embed_one("lead paint disclosure required by federal law")
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))
    assert dot(q, near) > dot(q, far)


def test_order_independence_is_documented_behavior():
    # feature hashing is a bag of words: this is exactly why the harness
    # refuses to publish a quality tier built on it
    e = HashEmbedder()
    a = e.embed_one("price of four hundred thousand")
    b = e.embed_one("thousand four hundred of price")
    assert a == b


def test_spec_parsing():
    e = make_embedder("test-hash:64")
    assert e.dim == 64 and not e.publishable
    for bad in ("nonsense", ""):
        try:
            make_embedder(bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, bad


def test_unknown_provider():
    try:
        make_embedder("cloudbust:nova")
        raised = False
    except ValueError:
        raised = True
    assert raised
