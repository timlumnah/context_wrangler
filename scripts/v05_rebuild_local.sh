#!/usr/bin/env bash
# v0.5 one-time rebuild: corpus, demo index, tables index, naive index,
# scanned twins, OCR+embed scanned index. All local, zero dollars.
set -exuo pipefail
cd /home/magnolia_gold/dev/claude_5-5-26/context_wrangler
PY=.venv/bin/python
CW=.venv/bin/cw

# wipe first: the seeded stream changed in v0.5, so regenerated filenames
# do not cover every v0.4 file. overwriting would leave orphan PDFs that
# get indexed as phantom documents (caught in this run: 116 added vs 58
# written). generate-corpus must run against an empty directory.
rm -rf corpus_out cw_index cw_corpus_index cw_corpus_index_naive \
       cw_corpus_index_scanned cw_corpus_index_api

"$CW" generate-corpus corpus_out --deals 25 --seed 42

"$CW" ingest demo_pdfs --index cw_index
"$CW" verify --index cw_index

"$CW" ingest corpus_out/pdfs --index cw_corpus_index \
    --embed ollama:nomic-embed-text --tables
"$CW" verify --index cw_corpus_index

"$CW" ingest corpus_out/pdfs --index cw_corpus_index_naive
"$CW" verify --index cw_corpus_index_naive

"$CW" scanify corpus_out/pdfs corpus_out/scanned_pdfs
"$CW" ingest corpus_out/scanned_pdfs --index cw_corpus_index_scanned \
    --ocr system --embed ollama:nomic-embed-text
"$CW" verify --index cw_corpus_index_scanned

echo "JOB A DONE"
