"""Answer-level evals: the grounded answerer scored against the same key.

Kinds and what counts as correct:
  exact    some citation quote contains the planted fact text
  table    some citation quote contains the planted reflowed row
           (label | amount); naive extraction makes this structurally
           impossible, which is the point of having the kind
  trap     quote carries the amended text AND no quote carries the stale
           text (citing the superseded number is the named failure)
  join     citations reach both expected documents
  refusal  the answerer refused (model gate or retrieval gate; both
           reported, since both are honest refusals)

Rejected outputs (gate refused to print the model's answer) are counted,
never smoothed into 'wrong': the gate firing is the product working.

The plumbing tier is labeled as such everywhere and refuses to write a
publishable markdown file: stub scores prove the loop, not quality.
"""

from __future__ import annotations

import datetime
import json

from context_wrangler.core.answer import answer_query
from context_wrangler.core.index import open_db, read_canonical
from context_wrangler.core.retrieve import candidates
from context_wrangler.core.search import verify_all

KINDS = ("exact", "table", "trap", "join", "refusal")
SAMPLE_BY_KIND = {"exact": 10, "table": 6, "trap": 8, "join": 4,
                  "refusal": 8}


def _grade(qq: dict, res: dict) -> dict:
    """Score one answer_query result against one key question."""
    quotes = " ".join(c.get("quote", "") for c in res["citations"])
    cited_docs = {c["doc_id"] for c in res["citations"]}
    row: dict = {}
    if qq["kind"] == "refusal":
        row["correct"] = res["status"] == "refused"
    elif res["status"] == "ok":
        if qq["kind"] in ("exact", "trap", "table"):
            row["correct"] = qq["expected_text"] in quotes
            if qq["kind"] == "trap":
                row["stale_cited"] = qq["stale_text"] in quotes
        elif qq["kind"] == "join":
            row["correct"] = set(qq["expected_docs"]) <= cited_docs
        else:
            row["correct"] = False
    else:
        row["correct"] = False
    return row


def _sample(questions: list[dict], limit: int) -> list[dict]:
    """Deterministic stratified sample: evenly spaced picks per kind."""
    by: dict[str, list[dict]] = {}
    for q in questions:
        by.setdefault(q["kind"], []).append(q)
    picked: list[dict] = []
    scale = limit / sum(SAMPLE_BY_KIND.values())
    for kind, want in SAMPLE_BY_KIND.items():
        pool = by.get(kind, [])
        n = min(pool and max(1, round(want * scale)) or 0, len(pool))
        step = max(1, len(pool) // max(n, 1))
        picked.extend(pool[::step][:n])
    return picked


def run_answers(index_dir: str, key_path: str, model_spec: str, *,
                limit: int = 32, k: int = 12, budget: int = 2400,
                mode: str = "auto", seed: int | None = 42,
                runs: int = 1, rerank_spec: str = "",
                rerank_n: int = 12) -> dict:
    """v0.4: each question runs `runs` times at a fixed seed. Headline
    correct is the MAJORITY across runs; n_unstable counts questions
    whose outcome varied at all, because 'seeded' is a claim and
    instability is a measurement. Reproducibility support is
    upstream-dependent (ollama honors options.seed; OpenRouter passes
    seed through and the metered runs report what actually happened).
    Cost per row is metered, not modeled: usage.cost where the provider
    reports dollars, cost_usd None (never guessed) otherwise."""
    conn = open_db(index_dir)
    vg = verify_all(conn, index_dir)
    if not vg["ok"]:
        conn.close()
        raise RuntimeError(
            f"grounding failed: {len(vg['bad'])} of {vg['checked']} spans "
            "do not reproduce; answer evals abort before scoring")

    key = json.load(open(key_path, encoding="utf-8"))
    chosen = _sample(key["questions"], limit)

    reranker = None
    rerank_fn = None
    if rerank_spec:
        from context_wrangler.core.rerank import make_reranker, rerank
        reranker = make_reranker(rerank_spec, seed=seed)
        rerank_fn = rerank

    rows = []
    for qq in chosen:
        run_rows = []
        for run_i in range(1, max(1, runs) + 1):
            cands = candidates(conn, index_dir, qq["query"], mode=mode, k=k)
            kw = (cands if mode == "bm25" else
                  candidates(conn, index_dir, qq["query"], mode="bm25", k=k))
            rr_info = {"enabled": False}
            if reranker is not None:
                cands, rr_info = rerank_fn(
                    cands, qq["query"], reranker=reranker, n=rerank_n)
            canon = {}
            for c in cands:
                canon.setdefault(c["doc_id"], read_canonical(index_dir,
                                                             c["doc_id"]))
            res = answer_query(qq["query"], cands, llm_spec=model_spec,
                               budget=budget, canonical=canon,
                               keyword_cands=kw, seed=seed)
            row = {"id": qq["id"], "kind": qq["kind"], "run": run_i,
                   "status": res["status"], "gate": res.get("gate"),
                   "attempts": res.get("attempts"),
                   "gen_ms": res.get("gen_ms"),
                   "usage": res.get("usage"), "rerank": rr_info}
            row.update(_grade(qq, res))
            run_rows.append(row)

        verdicts = [r["correct"] for r in run_rows]
        yes = sum(verdicts)
        combined = run_rows[0].copy()
        combined["runs"] = len(run_rows)
        combined["correct"] = yes * 2 > len(verdicts)      # majority
        combined["correct_runs"] = f"{yes}/{len(verdicts)}"
        combined["unstable"] = 0 < yes < len(verdicts)
        combined["per_run"] = [{"run": r["run"], "status": r["status"],
                                "correct": r["correct"]} for r in run_rows]
        # metered cost per question = sum over its runs (rerank included).
        # A row with NO model calls (gate refused before the model) is $0,
        # a fact. A row with calls whose provider did not meter dollars is
        # unknown, and unknowns poison the whole tier total: dash, never a
        # guess. A rejected-after-"model unreachable" row is unknown too:
        # a request that died mid-flight may still have been billed.
        calls = tok_p = tok_c = 0
        tot = 0.0
        priced = True
        for r in run_rows:
            for meter in (r["usage"], (r.get("rerank") or {}).get("usage")):
                if not meter:
                    continue                    # never built: zero calls
                calls += meter["calls"]
                tok_p += meter["prompt_tokens"]
                tok_c += meter["completion_tokens"]
                if meter["cost_usd"] is not None:
                    tot += meter["cost_usd"]
                else:
                    priced = False
        combined["cost"] = {"calls": calls, "prompt_tokens": tok_p,
                            "completion_tokens": tok_c,
                            "cost_usd": tot if priced else None,
                            "metered": priced}
        if any(r.get("stale_cited") for r in run_rows):
            combined["stale_cited"] = True
        rows.append(combined)
    conn.close()

    agg = {}
    for kind in SAMPLE_BY_KIND:
        subset = [r for r in rows if r["kind"] == kind]
        if not subset:
            continue
        agg[kind] = {
            "n": len(subset),
            "correct": sum(r.get("correct", False) for r in subset),
            "rejected": sum(r["status"] == "rejected" for r in subset),
            "refused": sum(r["status"] == "refused" for r in subset),
            "unstable": sum(r.get("unstable", False) for r in subset),
        }
        if kind == "trap":
            agg[kind]["stale_cited"] = sum(
                r.get("stale_cited", False) for r in subset)

    plumbing = model_spec.startswith("stub:") or (
        reranker is not None and not getattr(reranker, "publishable", True))
    total_cost = [r["cost"]["cost_usd"] for r in rows
                  if r["cost"]["cost_usd"] is not None]
    cost_complete = len(total_cost) == len(rows) and bool(rows)
    return {
        "run_utc": datetime.datetime.now(
            datetime.UTC).isoformat(timespec="seconds"),
        "model": model_spec,
        "publishable": not plumbing,
        "index": index_dir,
        "k": k, "budget": budget, "mode": mode,
        "seed": seed, "runs": max(1, runs),
        "rerank": rerank_spec or None,
        "sampled": len(chosen),
        "cost": {
            "cost_usd_total": sum(total_cost) if cost_complete else None,
            "metered_all_rows": cost_complete,
            "calls": sum(r["cost"]["calls"] for r in rows),
            "prompt_tokens": sum(r["cost"]["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["cost"]["completion_tokens"]
                                     for r in rows),
        },
        "aggregate": agg,
        "rows": rows,
    }


def to_markdown(report: dict) -> str:
    tag = "" if report["publishable"] else \
        "  (plumbing tier, not publishable quality)"
    seed = report.get("seed")
    runs = report.get("runs", 1)
    lines = [
        f"Answer eval, model {report['model']}, run {report['run_utc']}. "
        f"{report['sampled']} sampled questions, mode {report['mode']}, "
        f"k={report['k']}, budget {report['budget']} tokens, "
        f"seed {seed if seed is not None else 'none'}, "
        f"{runs} run(s) per question, rerank "
        f"{report.get('rerank') or 'off'}.{tag}",
        "",
        "| Kind | n | correct | refused | rejected | stale cited | unstable |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind, a in report["aggregate"].items():
        lines.append(
            f"| {kind} | {a['n']} | {a['correct']} | {a['refused']} "
            f"| {a['rejected']} | {a.get('stale_cited', '-')} "
            f"| {a.get('unstable', 0)} |")
    cost = report.get("cost", {})
    if cost.get("metered_all_rows"):
        lines.append("")
        lines.append(f"Metered cost: ${cost['cost_usd_total']:.4f} total, "
                     f"{cost['calls']} model calls, "
                     f"{cost['prompt_tokens']} prompt + "
                     f"{cost['completion_tokens']} completion tokens.")
    return "\n".join(lines)
