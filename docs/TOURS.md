# Tours by version

Step-by-step command tours for each version. The README has the 60-second quick start; these go deeper. Paths assume the repo root and the venv from the README.

## v0.1 to v0.5 tours

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pip install -e .

# demo corpus: the test fixture writer emits three synthetic transaction PDFs
.venv/bin/python -c "import sys; sys.path.insert(0, 'tests'); from fixtures_pdf import write_corpus; write_corpus('demo_pdfs')"

.venv/bin/cw ingest demo_pdfs --index cw_index
.venv/bin/cw search 'earnest money deposit' --index cw_index
.venv/bin/cw search '$12,500' --index cw_index --json
.venv/bin/cw verify --index cw_index    # PASS, every span byte-checked
.venv/bin/python -m pytest -q           # 164 tests
```

v0.1 answers nothing. It cites. The grounded answerer and refusal gate arrived in v0.3; the tours below run in build order.

The v0.2 tour, from the repo root (needs a running Ollama with `ollama pull nomic-embed-text` for the embed step):

```bash
.venv/bin/cw generate-corpus corpus_out --deals 25 --seed 42
.venv/bin/cw ingest corpus_out/pdfs --index cw_corpus_index --embed ollama:nomic-embed-text --tables
.venv/bin/python -m evals --index cw_corpus_index --key corpus_out/answer_key.json \
    --modes bm25 vec hybrid
.venv/bin/python -m evals --selftest    # no index, no model; proves the checks work
```

The v0.3 tour, answering through the gate. Needs the corpus and index from the v0.2 tour; a model is needed only for the last two lines, and `stub:extract` proves the loop offline:

```bash
.venv/bin/cw assemble 'earnest money deposit Whitfield 239 Granite' \
    --index cw_corpus_index                      # the budget allocation, no model
.venv/bin/cw answer 'earnest money deposit Whitfield 239 Granite' \
    --index cw_corpus_index --model stub:extract # plumbing tier, offline
.venv/bin/cw answer 'earnest money deposit Whitfield 239 Granite' \
    --index cw_corpus_index --model ollama:qwen2.5:7b
.venv/bin/python -m evals --answer --index cw_corpus_index \
    --key corpus_out/answer_key.json --model ollama:qwen2.5:7b --limit 16
```

The v0.4 tour: the cost/quality frontier and the rerank flag. Needs the corpus from the v0.2 tour plus an API-embedded index; `OPENROUTER_API_KEY` in the environment pays for the API tiers (cents), ollama serves the local ones:

```bash
.venv/bin/cw ingest corpus_out/pdfs --index cw_corpus_index_api \
    --embed openrouter:openai/text-embedding-3-small --tables  # prints metered ingest dollars
.venv/bin/python -m evals --frontier --key corpus_out/answer_key.json \
    --limit 16 --runs 3 --seed 42 --k 10 \
    --json evals/results/frontier.json --md evals/results/frontier.md
.venv/bin/cw answer 'earnest money deposit Whitfield 239 Granite' \
    --index cw_corpus_index_api --model openrouter:openai/gpt-4o-mini \
    --rerank openrouter:openai/gpt-4o-mini --seed 42  # metered line: $, tokens, ms
.venv/bin/python -m evals.readme_sync --check \
    FRONTIER=evals/results/frontier.md                 # the staleness gate CI runs
```

The v0.5 tour: the two extraction regimes. Needs the corpus from the v0.2 tour, `poppler-utils` + `tesseract-ocr` on PATH for the scanned leg, and fresh index dirs per regime (an index stamps its extraction regime at first ingest and refuses a later flip). Zero dollars; CPU minutes.

```bash
# the reflow regime: coordinate-aligned tables become pipe rows in canonical text
.venv/bin/cw ingest corpus_out/pdfs --index cw_corpus_index \
    --embed ollama:nomic-embed-text --tables
.venv/bin/cw verify --index cw_corpus_index   # PASS, reflowed rows byte-checked
.venv/bin/cw search 'title insurance amount Whitfield 239 Granite' \
    --index cw_corpus_index                    # a "Title Insurance | $500" row answers

# the v0.4-regime comparison index: same corpus bytes, no --tables
.venv/bin/cw ingest corpus_out/pdfs --index cw_corpus_index_naive

# the scanned twin: rasterize 58 text PDFs to image-only PDFs, ingest through OCR
.venv/bin/cw scanify corpus_out/pdfs corpus_out/scanned_pdfs
.venv/bin/cw ingest corpus_out/scanned_pdfs --index cw_corpus_index_scanned \
    --ocr system --embed ollama:nomic-embed-text
.venv/bin/cw verify --index cw_corpus_index_scanned  # PASS against the OCR'd canonicals

# the published v0.5 comparisons (also what scripts/regen_tables.sh runs)
.venv/bin/python -m evals --index cw_corpus_index --key corpus_out/answer_key.json \
    --modes bm25 vec hybrid --json evals/results/latest.json --md evals/results/baseline.md
.venv/bin/python -m evals --index cw_corpus_index_naive --key corpus_out/answer_key.json \
    --modes bm25 --json evals/results/naive.json --md evals/results/naive.md
.venv/bin/python -m evals --index cw_corpus_index_scanned --key corpus_out/answer_key.json \
    --modes bm25 vec hybrid --scan-map \
    --json evals/results/scanned.json --md evals/results/scanned.md
.venv/bin/python scripts/gen_v05_blocks.py   # V05_TABLES + V05_OCR README blocks
```
