"""Assemble the v0.5 comparison README blocks from saved eval reports.

Pure re-format: every number here is copied from a report JSON produced
by a command printed in the README next to the block. No metric is
recomputed, so nothing can drift from what the harness measured:

  naive.json    evals --index cw_corpus_index_naive --modes bm25
  latest.json   evals --index cw_corpus_index --modes bm25 vec hybrid
  scanned.json  evals --index cw_corpus_index_scanned --modes bm25 vec
                hybrid --scan-map
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "evals", "results")


def _load(name: str) -> dict:
    with open(os.path.join(RES, name), encoding="utf-8") as fh:
        return json.load(fh)


def _cell(rep: dict, mode: str, key: str) -> str:
    agg = rep["aggregate"].get(mode, {})
    v = agg.get(key)
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def main() -> None:
    naive = _load("naive.json")
    text = _load("latest.json")
    scan = _load("scanned.json")

    deals = text["key"]["deals"]
    nq = text["key"]["questions"]

    lines = [
        f"Table extraction, same seeded corpus ({deals} deals, {nq} "
        f"questions), k={text['k']}. 'table row@10' asks: does any "
        "top-10 chunk hold the planted row string, label AND its figure "
        "together? 'table recall@10' is the doc-level view.",
        "",
        "| Ingest regime | Mode | table row@10 | table recall@10 "
        "| exact recall@10 |",
        "|---|---|---|---|---|",
        f"| plain (v0.4 behavior) | bm25 "
        f"| {_cell(naive, 'bm25', 'table_row_top10')} "
        f"| {_cell(naive, 'bm25', 'table_recall')} "
        f"| {_cell(naive, 'bm25', 'exact_recall')} |",
    ]
    for mode in ("bm25", "vec", "hybrid"):
        lines.append(
            f"| --tables layout-reflow | {mode} "
            f"| {_cell(text, mode, 'table_row_top10')} "
            f"| {_cell(text, mode, 'table_recall')} "
            f"| {_cell(text, mode, 'exact_recall')} |")
    tables_md = "\n".join(lines) + "\n"

    lines = [
        f"OCR roundtrip, same corpus rasterized to image-only PDFs at "
        f"250 dpi and ingested through the system OCR tier (doc ids "
        f"mapped by filename), k={scan['k']}, run {scan['run_utc']}.",
        "",
        "| Corpus | Mode | exact recall@10 | exact MRR | table row@10 "
        "| trap top-1 | join@10 | refusal mean hits |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode in ("bm25", "vec", "hybrid"):
        lines.append(
            f"| text layer | {mode} | {_cell(text, mode, 'exact_recall')} "
            f"| {_cell(text, mode, 'exact_mrr')} "
            f"| {_cell(text, mode, 'table_row_top10')} "
            f"| {_cell(text, mode, 'trap_top1')} "
            f"| {_cell(text, mode, 'join_recall')} "
            f"| {_cell(text, mode, 'refusal_mean_hits')} |")
    for mode in ("bm25", "vec", "hybrid"):
        lines.append(
            f"| scanned, OCR | {mode} "
            f"| {_cell(scan, mode, 'exact_recall')} "
            f"| {_cell(scan, mode, 'exact_mrr')} "
            f"| {_cell(scan, mode, 'table_row_top10')} "
            f"| {_cell(scan, mode, 'trap_top1')} "
            f"| {_cell(scan, mode, 'join_recall')} "
            f"| {_cell(scan, mode, 'refusal_mean_hits')} |")
    ocr_md = "\n".join(lines) + "\n"

    with open(os.path.join(RES, "v05_tables.md"), "w",
              encoding="utf-8") as fh:
        fh.write(tables_md)
    with open(os.path.join(RES, "v05_ocr.md"), "w", encoding="utf-8") as fh:
        fh.write(ocr_md)
    print(tables_md)
    print(ocr_md)


if __name__ == "__main__":
    main()
