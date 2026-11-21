"""Cost/quality frontier: the same answer eval run across model tiers,
published as dollars-per-1k-queries vs score. This is the v0.4 product.

Tiers (the README spec, now measured):
    local        ollama embedder + ollama chat model. $0, metered as 0.
    api-embed    OpenRouter embedder + ollama chat model: the price of
                 better vectors while answering stays on your laptop.
    full-api     OpenRouter embedder + OpenRouter chat model.

Money is METETERED, never modeled: OpenRouter returns exact usage.cost
per call; ingest wrote the corpus embed cost into index meta
(embed_cost_usd); local compute is $0 by definition. A tier whose
dollars cannot be metered shows "-" in the money column and stays in
the table: hiding the expensive row is how a README sells a lie.

Stub and non-publishable tiers raise: this module exists to publish
quality numbers, and the harness rules for plumbing tiers still apply.
"""

from __future__ import annotations

import datetime

from context_wrangler.core.index import open_db
from evals.answers import run_answers

# tier = id, label, index dir, embed spec the index must carry, chat spec
DEFAULT_TIERS: tuple[dict, ...] = (
    {"id": "local", "label": "local",
     "index": "cw_corpus_index",
     "embed": "ollama:nomic-embed-text", "chat": "ollama:qwen2.5:7b"},
    {"id": "api-embed", "label": "api-embed + local chat",
     "index": "cw_corpus_index_api",
     "embed": "openrouter:openai/text-embedding-3-small",
     "chat": "ollama:qwen2.5:7b"},
    {"id": "api-cheap", "label": "full API (cheap)",
     "index": "cw_corpus_index_api",
     "embed": "openrouter:openai/text-embedding-3-small",
     "chat": "openrouter:openai/gpt-4o-mini"},
)


def resolve_tiers(ids: list[str] | None) -> tuple[dict, ...]:
    """Filter DEFAULT_TIERS by id; CI runs the api tiers only."""
    if not ids:
        return DEFAULT_TIERS
    out = []
    for t in DEFAULT_TIERS:
        if t["id"] in ids:
            out.append(t)
    unknown = set(ids) - {t["id"] for t in out}
    if unknown:
        raise ValueError(f"unknown tier id(s) {sorted(unknown)}; known: "
                         f"{[t['id'] for t in DEFAULT_TIERS]}")
    return tuple(out)


def _embed_cost_meta(index_dir: str) -> float | None:
    conn = open_db(index_dir)
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'embed_cost_usd'").fetchone()
        return float(row["value"]) if row else None
    finally:
        conn.close()


def _embed_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def embed_usage_snapshot(index_dir: str) -> float | None:
    """Metered cost of the index's embedder object, if it is cached and
    priced. The frontier calls this before and after each tier run: the
    delta is what the query-time embeddings of exactly this eval cost."""
    from context_wrangler.core.embedder import make_embedder
    from context_wrangler.core.retrieve import _embedder_cache
    conn = open_db(index_dir)
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'embed_model'").fetchone()
        if row is None:
            return None
        name = row["value"]
        emb = _embedder_cache.get(name) or make_embedder(name)
        _embedder_cache[name] = emb               # keep it cached for the run
        return emb.usage.cost_usd
    finally:
        conn.close()


def _tier_row(name: str, index_dir: str, embed_spec: str, chat_spec: str,
              report: dict, ingest_cost: float | None,
              embed_delta: float | None) -> dict:
    if not report["publishable"]:
        raise RuntimeError(f"tier {name!r} ran a plumbing model; "
                           "frontier refuses non-publishable tiers")
    n = report["sampled"]
    rows = report["rows"]
    correct = sum(r["correct"] for r in rows)
    unstable = sum(r.get("unstable", False) for r in rows)
    cost = report["cost"]
    per_q: float | None = None
    if cost["metered_all_rows"]:
        per_q = (cost["cost_usd_total"] / n if n else 0.0)
        if embed_delta is not None:
            per_q += embed_delta / n
        elif embed_spec.startswith("ollama"):
            pass                      # local embed: $0 is a fact, not a gap
        else:
            per_q = None              # query embed unpriced: refuse to guess
    per_1k = None if per_q is None else per_q * 1000.0
    gen_ms = [r["gen_ms"] for r in rows if r.get("gen_ms")]
    return {
        "tier": name,
        "index": index_dir,
        "embedder": embed_spec,
        "chat": chat_spec,
        "sampled": n, "runs": report["runs"], "seed": report["seed"],
        "correct_majority": correct,
        "n_unstable": unstable,
        "dollars_per_1k_queries": per_1k,
        "ingest_cost_usd": ingest_cost,
        "query_embed_cost_usd": embed_delta,
        "median_gen_ms": (sorted(gen_ms)[len(gen_ms) // 2] if gen_ms else None),
        "report": report,
    }


def run_frontier(key_path: str, tiers: tuple[dict, ...] | None = None, *,
                 limit: int = 16, seed: int = 42, runs: int = 3,
                 mode: str = "hybrid", rerank_spec: str = "", k: int = 12,
                 index_map: dict[str, str] | None = None) -> dict:
    """index_map redirects a tier id at a different index dir (CI uses
    this to run the api tier against a freshly built corpus without
    inventing dev paths)."""
    from context_wrangler.core.index import embed_config
    tiers = tiers if tiers is not None else DEFAULT_TIERS
    index_map = index_map or {}
    out_rows = []
    for t in tiers:
        name = t["label"]
        index_dir = index_map.get(t["id"], t["index"])
        embed_spec, chat_spec = t["embed"], t["chat"]
        conn = open_db(index_dir)          # cheap: we do NOT want to start
        try:                               # 7B models on stale indexes
            cfg = embed_config(conn)
        finally:
            conn.close()
        if cfg is None or cfg[0] != embed_spec:
            raise RuntimeError(
                f"tier {name!r} expects index {index_dir!r} embedded with "
                f"{embed_spec}, found {cfg[0] if cfg else 'no vectors'}; "
                "the frontier table describes real stacks or it lies")
        ingest_cost = _embed_cost_meta(index_dir)
        before = embed_usage_snapshot(index_dir)
        report = run_answers(index_dir, key_path, chat_spec, limit=limit,
                             mode=mode, seed=seed, runs=runs, k=k,
                             rerank_spec=rerank_spec)
        after = embed_usage_snapshot(index_dir)
        out_rows.append(_tier_row(name, index_dir, embed_spec, chat_spec,
                                  report, ingest_cost,
                                  _embed_delta(before, after)))
    return {
        "run_utc": datetime.datetime.now(datetime.UTC).isoformat(
            timespec="seconds"),
        "key": key_path,
        "limit": limit, "seed": seed, "runs": runs, "mode": mode,
        "rerank": rerank_spec or None,
        "rows": out_rows,
    }


def _money(v: float | None) -> str:
    return f"${v:.4f}" if v is not None else "-"


def to_markdown(front: dict) -> str:
    # the header states what the sampler ACTUALLY drew (kind quotas round
    # up), not the --limit knob it was asked to size from
    n_sampled = front["rows"][0]["sampled"] if front["rows"] else front["limit"]
    lines = [
        f"Cost/quality frontier, run {front['run_utc']}. {front['mode']} "
        f"retrieval, seed {front['seed']}, {front['runs']} runs per "
        f"question, {n_sampled} stratified questions, rerank "
        f"{front.get('rerank') or 'off'}.",
        "",
        "| Tier | Embedder | Chat model | correct | unstable | "
        "$/1k queries | ingest $ | median gen ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in front["rows"]:
        lines.append(
            f"| {r['tier']} | {r['embedder']} | {r['chat']} "
            f"| {r['correct_majority']}/{r['sampled']} "
            f"| {r['n_unstable']} | {_money(r['dollars_per_1k_queries'])} "
            f"| {_money(r['ingest_cost_usd'])} "
            f"| {r['median_gen_ms'] if r['median_gen_ms'] is not None else '-'} |")
    lines.append("")
    lines.append("$/1k queries counts per-query spend only (chat + query "
                 "embedding) times 1000. The ingest column is what one-time "
                 "corpus embedding cost, metered into index meta at ingest. "
                 "Dashes mean the provider does not meter dollars; we do not "
                 "model them. Local tiers are $0 because local compute is.")
    return "\n".join(lines)
