"""python -m evals

  .venv/bin/python -m evals --selftest
  .venv/bin/python -m evals --index cw_corpus_index --key corpus_out/answer_key.json
  .venv/bin/python -m evals --index ... --key ... --modes bm25 vec hybrid \
      --json evals/results/latest.json --md evals/results/baseline.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from evals.harness import run, run_selftest, to_markdown


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evals")
    ap.add_argument("--selftest", action="store_true",
                    help="prove the metric functions work; no index, no model")
    ap.add_argument("--index", default="cw_index")
    ap.add_argument("--key", default="corpus_out/answer_key.json")
    ap.add_argument("--modes", nargs="+", default=["bm25"],
                    choices=["bm25", "vec", "hybrid"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--scan-map", action="store_true",
                    help="index holds the scanned image-only twins of the "
                         "key's documents; map key doc ids to indexed doc "
                         "ids by filename (retrieval eval only)")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write the full report (incl. per-question rows)")
    ap.add_argument("--md", dest="md_out", default=None,
                    help="write the published markdown table")
    ap.add_argument("--answer", action="store_true",
                    help="answer-level eval instead of retrieval-level")
    ap.add_argument("--model", default="",
                    help="ollama:<model>, openai:<model>, "
                         "openrouter:<model>, or stub:extract "
                         "(plumbing tier, never publishable)")
    ap.add_argument("--limit", type=int, default=32,
                    help="max sampled questions for --answer")
    ap.add_argument("--budget", type=int, default=2400)
    ap.add_argument("--retrieve-mode", default="auto",
                    choices=["auto", "bm25", "vec", "hybrid"])
    ap.add_argument("--seed", type=int, default=42,
                    help="fixed seed handed to the model sampler for "
                         "published evals (pass -1 for no seed)")
    ap.add_argument("--runs", type=int, default=1,
                    help="runs per question in --answer; the table then "
                         "scores the majority and counts instability")
    ap.add_argument("--rerank", default="", metavar="LLM_SPEC",
                    help="optional rerank stage for --answer "
                         "(openrouter:openai/gpt-4o-mini, ollama:<model>, "
                         "stub:lexical). Latency and cost are metered into "
                         "the report either way.")
    ap.add_argument("--frontier", action="store_true",
                    help="cost/quality frontier across model tiers "
                         "instead of a single-tier answer eval")
    ap.add_argument("--tiers", nargs="+", default=None, metavar="ID",
                    help="frontier tier ids to run (default all: "
                         "local api-embed api-cheap; CI uses api tiers)")
    ap.add_argument("--tier-index", nargs="*", default=[], metavar="ID=DIR",
                    help="frontier: redirect a tier id to a different "
                         "index dir (CI: --tier-index api-cheap=cw_index_ci_api)")
    args = ap.parse_args(argv)
    seed = None if args.seed < 0 else args.seed

    if args.selftest:
        return run_selftest()

    if args.frontier:
        from evals.frontier import resolve_tiers, run_frontier
        from evals.frontier import to_markdown as md_frontier
        try:
            tiers = resolve_tiers(args.tiers)
            imap = {}
            for spec in args.tier_index:
                tid, _, dir_ = spec.partition("=")
                if not dir_:
                    print(f"bad --tier-index spec {spec!r}", file=sys.stderr)
                    return 2
                imap[tid] = dir_
            front = run_frontier(args.key, tiers, limit=args.limit,
                                 seed=seed, runs=max(1, args.runs), k=args.k,
                                 mode=args.retrieve_mode if args.retrieve_mode != "auto" else "hybrid",
                                 rerank_spec=args.rerank, index_map=imap)
        except (RuntimeError, ValueError) as exc:
            print(f"frontier refused: {exc}", file=sys.stderr)
            return 1
        md = md_frontier(front)
        print(md)
        if args.json_out:
            os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(front, fh, indent=2, default=str)
            print(f"\nfrontier raw -> {args.json_out}", file=sys.stderr)
        if args.md_out:
            os.makedirs(os.path.dirname(args.md_out) or ".", exist_ok=True)
            with open(args.md_out, "w", encoding="utf-8") as fh:
                fh.write(md + "\n")
            print(f"table -> {args.md_out}", file=sys.stderr)
        return 0

    if args.answer:
        from evals.answers import run_answers, to_markdown as md_answers
        if not args.model:
            print("--answer needs --model", file=sys.stderr)
            return 2
        report = run_answers(args.index, args.key, args.model,
                             limit=args.limit, k=args.k,
                             budget=args.budget, mode=args.retrieve_mode,
                             seed=seed, runs=max(1, args.runs),
                             rerank_spec=args.rerank)
        md = md_answers(report)
        print(md)
        if args.json_out:
            os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
        if args.md_out:
            if not report["publishable"]:
                print("refusing to write a publishable table from the "
                      "plumbing tier", file=sys.stderr)
                return 0
            os.makedirs(os.path.dirname(args.md_out) or ".", exist_ok=True)
            with open(args.md_out, "w", encoding="utf-8") as fh:
                fh.write(md + "\n")
        return 0

    if args.scan_map and (args.answer or args.frontier):
        print("--scan-map applies to the retrieval eval only",
              file=sys.stderr)
        return 2
    report = run(args.index, args.key, modes=tuple(args.modes), k=args.k,
                 scan_map=args.scan_map)
    md = to_markdown(report)
    print(md)
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nraw results -> {args.json_out}", file=sys.stderr)
    if args.md_out:
        os.makedirs(os.path.dirname(args.md_out) or ".", exist_ok=True)
        with open(args.md_out, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        print(f"table -> {args.md_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
