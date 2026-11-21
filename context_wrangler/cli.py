"""cw: the context_wrangler CLI. argparse only, no framework.

    cw ingest <pdf_dir> [--index ./cw_index] [--embed provider:model]
                         [--ocr system|stub] [--tables]
    cw scanify <src_pdfs> <dst_pdfs> [--dpi 250]   # make scanned twins
    cw generate-corpus <out_dir> [--deals 25] [--seed 42]
    cw search <query...> [-k 8] [--mode auto|bm25|vec|hybrid] [--json]
    cw assemble <question...> [--budget 2400] [--rerank SPEC]  # no model
    cw answer <question...> --model ollama:qwen2.5:7b [--budget 2400]
                              [--rerank SPEC] [--seed 42]
    cw verify [--index ./cw_index]
    cw stats [--index ./cw_index]

Answers come out of the grounding gate in core/answer.py or not at all:
every citation is byte-checked against the canonical text before the CLI
will print a non-refusal answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from context_wrangler.core.index import open_db
from context_wrangler.core.ingest import ingest_dir
from context_wrangler.core.search import search, verify_all

DEFAULT_INDEX = "cw_index"


def _rerank_line(info: dict) -> str:
    cost = (info.get("usage") or {}).get("cost_usd")
    money = "" if cost is None else f", cost ${cost:.6f}"
    state = "ok" if info.get("ok", True) else \
        f"FAILED ({'; '.join(info.get('reasons', []))}; original order kept)"
    return (f"rerank: {info['name']} {state}, window {info['window']}, "
            f"{info.get('moved', 0)} moved, {info.get('ms', 0)} ms"
            f"{money}")


def _meter_line(res: dict, rr_info: dict) -> str:
    parts = []
    usage = res.get("usage")
    if usage and usage["calls"]:
        parts.append(f"{usage['prompt_tokens']} prompt + "
                     f"{usage['completion_tokens']} completion tokens")
        if usage["cost_usd"] is not None:
            parts.append(f"answer cost ${usage['cost_usd']:.6f}")
        parts.append(f"gen {res.get('gen_ms', 0)} ms")
    return "metered: " + ", ".join(parts) if parts else ""


def _p_ingest(sub):
    p = sub.add_parser("ingest", help="index a folder of PDFs")
    p.add_argument("pdf_dir")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--embed", default=None, metavar="PROVIDER:MODEL",
                   help="also embed chunks (ollama:nomic-embed-text, "
                        "openai:text-embedding-3-small, test-hash)")
    p.add_argument("--ocr", default="", metavar="TIER",
                   help="OCR tier for pages with no text layer "
                        "(system[:dpi[:lang[:psm]]] or stub[:marker])")
    p.add_argument("--tables", action="store_true",
                   help="reflow coordinate-aligned tables into pipe rows "
                        "(pages without a grid are extracted byte-identical "
                        "to the plain path)")


def _p_scan(sub):
    p = sub.add_parser("scanify",
                       help="turn a folder of text PDFs into image-only "
                            "scanned-looking twins (v0.5 OCR eval)")
    p.add_argument("src_dir")
    p.add_argument("dst_dir")
    p.add_argument("--dpi", type=int, default=250)


def _p_gen(sub):
    p = sub.add_parser("generate-corpus",
                       help="write synthetic transaction docs + answer key")
    p.add_argument("out_dir")
    p.add_argument("--deals", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)


def _p_search(sub):
    p = sub.add_parser("search", help="search, results carry cited spans")
    p.add_argument("query", nargs="+")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--mode", default="auto",
                   choices=["auto", "bm25", "vec", "hybrid"])
    p.add_argument("--or", action="store_true", dest="use_or",
                   help="boolean OR instead of AND across terms "
                        "(bm25 mode only)")
    p.add_argument("--json", action="store_true",
                   help="machine-readable results")
    p.add_argument("--full", action="store_true",
                   help="print the whole chunk, not the snippet")


def _p_simple(sub, name, help_text):
    p = sub.add_parser(name, help=help_text)
    p.add_argument("--index", default=DEFAULT_INDEX)


def _p_assemble(sub):
    p = sub.add_parser("assemble",
                       help="show the token-budget allocation, no model")
    p.add_argument("query", nargs="+")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--budget", type=int, default=2400)
    p.add_argument("--mode", default="auto",
                   choices=["auto", "bm25", "vec", "hybrid"])
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--rerank", default="", metavar="LLM_SPEC",
                   help="optional rerank stage before the allocator "
                        "(ollama:<model>, openrouter:<model>, stub:lexical)")
    p.add_argument("--seed", type=int, default=-1,
                   help="pin the sampler seed for --rerank (-1 = default)")
    p.add_argument("--json", action="store_true")


def _p_answer(sub):
    p = sub.add_parser("answer",
                       help="grounded answer with cited spans, refusal gate")
    p.add_argument("question", nargs="+")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--model", default=os.environ.get("CW_MODEL", ""),
                   help="ollama:<model>, openai:<model>, "
                        "openrouter:<model>, stub:extract")
    p.add_argument("--mode", default="auto",
                   choices=["auto", "bm25", "vec", "hybrid"])
    p.add_argument("--budget", type=int, default=2400)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--rerank", default="", metavar="LLM_SPEC",
                   help="optional rerank stage before the allocator "
                        "(ollama:<model>, openrouter:<model>, stub:lexical)")
    p.add_argument("--seed", type=int, default=-1,
                   help="pin the sampler seed (-1 = provider default)")
    p.add_argument("--json", action="store_true")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="cw", description="document intelligence you can run on a laptop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _p_ingest(sub)
    _p_scan(sub)
    _p_gen(sub)
    _p_search(sub)
    _p_assemble(sub)
    _p_answer(sub)
    _p_simple(sub, "verify", "byte-check every cited span against the source text")
    _p_simple(sub, "stats", "index contents at a glance")
    args = ap.parse_args(argv)

    if args.cmd == "generate-corpus":
        from context_wrangler.corpus import generate_corpus, write_corpus
        c = generate_corpus(deals=args.deals, seed=args.seed)
        orphans = write_corpus(args.out_dir, c)
        print(f"wrote {len(c.pdfs)} PDFs and answer_key.json to {args.out_dir} "
              f"({len(c.questions)} questions, seed {args.seed})")
        for f in orphans:
            print(f"  WARNING orphan pdf this corpus does not claim: {f}",
                  file=sys.stderr)
        if orphans:
            print("  a stale corpus in this dir will poison retrieval; "
                  "delete the orphans or use a fresh out_dir", file=sys.stderr)
        return 0

    if args.cmd == "scanify":
        from context_wrangler.core.ocr import scanify_dir, tools_available
        if not tools_available():
            print("scanify needs pdftoppm + tesseract on PATH",
                  file=sys.stderr)
            return 2
        done = scanify_dir(args.src_dir, args.dst_dir, dpi=args.dpi)
        pages = sum(info["pages"] for _n, info in done)
        print(f"scanned {len(done)} PDFs ({pages} pages, {args.dpi} dpi) "
              f"-> {args.dst_dir}")
        return 0

    if args.cmd == "ingest":
        embedder = None
        if args.embed:
            from context_wrangler.core.embedder import make_embedder
            embedder = make_embedder(args.embed)
        ocr = None
        if args.ocr:
            from context_wrangler.core.ocr import make_ocr
            ocr = make_ocr(args.ocr)
        rep = ingest_dir(args.pdf_dir, args.index, embedder=embedder,
                         ocr=ocr, tables=args.tables)
        extra = f", embedded with {embedder.name}" if embedder else ""
        if embedder is not None and embedder.usage.cost_usd:
            extra += f" (metered ${embedder.usage.cost_usd:.6f})"
        if args.tables:
            extra += ", tables: layout-reflow"
        if ocr is not None:
            extra += f", ocr: {ocr.config}"
        backfill = f", backfilled {rep.backfilled} existing chunks" if rep.backfilled else ""
        ocr_line = f"  ocr pages {rep.ocr_pages}" if rep.ocr_pages else ""
        print(f"added {len(rep.added)}  updated {len(rep.updated)}  "
              f"moved {len(rep.moved)}  unchanged {len(rep.unchanged)}  "
              f"removed {len(rep.removed)}  duplicates {len(rep.duplicates)}  "
              f"errors {len(rep.errors)}{ocr_line}{extra}{backfill}")
        for path, msg in rep.errors:
            print(f"  ERROR {path}: {msg}", file=sys.stderr)
        return 1 if rep.errors else 0

    if args.cmd == "search":
        query = " ".join(args.query)
        conn = open_db(args.index)
        from context_wrangler.core.retrieve import candidates
        try:
            if args.use_or:
                if args.mode != "bm25":
                    print("--or only applies to bm25 mode", file=sys.stderr)
                    return 2
                hits = search(conn, args.index, query, k=args.k, use_or=True)
            else:
                hits = candidates(conn, args.index, query,
                                  mode=args.mode, k=args.k)
        finally:
            conn.close()
        if args.json:
            print(json.dumps({"query": query, "results": hits}, indent=2))
        elif not hits:
            print("no matches")
        else:
            for h in hits:
                section = f"  [{h['section']}]" if h["section"] else ""
                print(f"{h['filename']}  p{h['page']}  "
                      f"span {h['span'][0]}-{h['span'][1]}{section}")
                shown = h["text"] if args.full else h["snippet"]
                print(f"    {shown}")
        return 0

    if args.cmd in ("assemble", "answer"):
        question = " ".join(args.query if args.cmd == "assemble"
                            else args.question)
        conn = open_db(args.index)
        from context_wrangler.core.retrieve import candidates
        from context_wrangler.core.assemble import assemble
        from context_wrangler.core.index import read_canonical
        try:
            cands = candidates(conn, args.index, question,
                               mode=args.mode, k=args.k)
        finally:
            conn.close()

        rr_info = {"enabled": False}
        if getattr(args, "rerank", ""):
            from context_wrangler.core.rerank import make_reranker, rerank
            seed = None if getattr(args, "seed", -1) < 0 else args.seed
            rr = make_reranker(args.rerank, seed=seed)
            cands, rr_info = rerank(cands, question, reranker=rr)
        canon = {}
        for c in cands:
            canon.setdefault(c["doc_id"], read_canonical(args.index,
                                                         c["doc_id"]))
        conn = open_db(args.index)
        try:
            kw = candidates(conn, args.index, question, mode="bm25",
                            k=args.k)
        finally:
            conn.close()

        if args.cmd == "assemble":
            res = assemble(cands, budget=args.budget, canonical=canon)
            print_support = (f"keyword support: {len(kw)} chunks"
                             if args.mode != "bm25" else "")
            if args.json:
                print(json.dumps({
                    "query": question, "budget": res.budget,
                    "capacity": res.capacity, "used": res.used,
                    "rerank": rr_info,
                    "blocks": [{"label": lab, "doc_id": b.doc_id,
                                "filename": b.filename, "span": list(b.span),
                                "section": b.section, "cost": b.cost}
                               for lab, b in zip(res.labels, res.blocks)],
                    "dropped": [{"filename": b.filename, "span": list(b.span),
                                 "reason": r} for b, r in res.dropped],
                }, indent=2))
                return 0
            print(f"budget {res.budget} tokens (chars/4 heuristic), "
                  f"capacity {res.capacity} for blocks, used {res.used}")
            if rr_info.get("enabled"):
                print(_rerank_line(rr_info))
            if print_support:
                print(print_support)
            for lab, b in zip(res.labels, res.blocks):
                print(f"  [{lab}] {b.filename} span {b.span[0]}-{b.span[1]} "
                      f"cost {b.cost}")
            for b, reason in res.dropped:
                print(f"  dropped: {b.filename} span {b.span[0]}-{b.span[1]} "
                      f"({reason})")
            if not res.blocks:
                print("  nothing fit the budget")
            return 0

        # answer
        spec = args.model
        if not spec:
            print("no model given: pass --model (ollama:<model>, "
                  "openai:<model>, stub:extract) or set CW_MODEL",
                  file=sys.stderr)
            return 2
        from context_wrangler.core.answer import answer_query
        res = answer_query(question, cands, llm_spec=spec,
                           budget=args.budget, canonical=canon,
                           keyword_cands=kw,
                           seed=None if args.seed < 0 else args.seed)
        if args.json:
            out = dict(res)
            if rr_info.get("enabled"):
                out["rerank"] = rr_info
            print(json.dumps(out, indent=2))
        else:
            if rr_info.get("enabled"):
                print(_rerank_line(rr_info))
            print(f"status: {res['status']}"
                  + (f" ({res['gate']})" if res.get("gate") else ""))
            print(f"answer: {res['answer']}")
            for cit in res["citations"]:
                print(f"  [{cit['label']}] span {cit['span'][0]}-"
                      f"{cit['span'][1]}: {cit['quote'][:100]}")
            ctx = res["context"]
            print(f"context: {ctx['blocks']} blocks, {ctx['used']}/"
                  f"{ctx['capacity']} tokens, {len(ctx['dropped'])} dropped")
            meter = _meter_line(res, rr_info)
            if meter:
                print(meter)
            for reason in res.get("reasons", []):
                print(f"  reject: {reason}", file=sys.stderr)
        return 0 if res["status"] in ("ok", "refused") else 2

    if args.cmd == "verify":
        conn = open_db(args.index)
        res = verify_all(conn, args.index)
        conn.close()
        if res["ok"]:
            print(f"PASS  {res['checked']} spans verified byte-for-byte")
            return 0
        print(f"FAIL  {len(res['bad'])} of {res['checked']} spans corrupt")
        for b in res["bad"][:20]:
            print(f"  chunk {b['chunk_id']} doc {b['doc_id'][:12]} "
                  f"span {b['start']}-{b['end']}", file=sys.stderr)
        return 1

    if args.cmd == "stats":
        conn = open_db(args.index)
        docs = conn.execute("SELECT COUNT(*) c FROM docs").fetchone()["c"]
        chunks = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
        print(f"index: {args.index}")
        print(f"docs {docs}  chunks {chunks}")
        for r in conn.execute("""SELECT d.filename, COUNT(c.chunk_id) n,
                                        d.n_pages, d.text_len
                                 FROM docs d
                                 LEFT JOIN chunks c ON c.doc_id = d.doc_id
                                 GROUP BY d.doc_id ORDER BY d.filename"""):
            print(f"  {r['filename']}: {r['n']} chunks, "
                  f"{r['n_pages']} pages, {r['text_len']} chars")
        conn.close()
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
