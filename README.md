# context_wrangler

Document intelligence you can run on a laptop. Point it at a folder of messy PDFs. Get answers that cite the exact span in the source document, plus a test suite proving the citations are not invented.

Status: v0.1 through v0.5 are built; the roadmap shipped. No CI workflow has executed yet, and nothing is pushed. v0.1: ingest, structure-aware chunking, FTS5 BM25, search CLI with cited spans. v0.2: sqlite-vec embeddings, hybrid fusion with RRF, the seeded corpus generator with machine-checkable answer keys, and the eval harness that published the baseline tables below. v0.3: the token-budget context assembler, the grounded answering CLI, and the refusal gate enforced in code. v0.4: OpenRouter providers with exact per-call cost metering, the cost/quality frontier table across three model tiers, fixed-seed multi-run answer evals that publish measured instability, the rerank stage behind a flag with its latency and cost documented, and README tables that regenerate between marker pairs instead of rotting. v0.5: the two extraction regimes. Optional OCR (system pdftoppm + tesseract, per page, and only where a page's text layer is thin) and coordinate-aware table reflow, each stamped into index meta so one index can never mix extraction regimes. The corpus grew a settlement table drawn the way real table writers draw, column-major, which breaks label-value pairing in naive extraction and is what --tables repairs; table questions grade on the reflowed row string, at chunk level, where document-level recall cannot hide the damage. And the whole corpus now has a scanned image-only twin that scores against its text original, so the price of OCR is a published number instead of a vibe. The proof: 164 tests pass on Python 3.14.9 (2026-11-21), `cw verify` byte-checks every indexed span including the OCR'd ones, `python -m evals --selftest` proves the metric functions, the API tiers are verified live against a metered OpenRouter account, and every published table below was produced by a command printed next to it.

## Try it

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

## Why this exists

Otto's agent memory was structured: SQL-backed tools over deal facts (parties, dates, lender). Precise, and blind to anything not already in a schema. The documents that carry the real business knowledge, contracts, addenda, disclosures, statements, were outside the retrieval. context_wrangler is the unstructured sibling: hybrid retrieval over PDFs with mandatory provenance and evals that score honesty.

## The pipeline

```
ingest -> chunk -> index -> retrieve -> assemble -> answer
```

1. **Ingest:** extract PDF text with structure (headings, page numbers preserved). v0.5 added the two extraction regimes, each stamped in index meta and refused mid-index: `--ocr TIER` consults the system pdftoppm + tesseract tier only where a page's text layer is thinner than 24 chars, and `--tables` re-extracts pages in coordinate-aware layout mode, detects column-aligned grids, and reflows them to pipe rows ("Title Insurance | $500") in the canonical text. Pages with no grid keep the plain-mode bytes byte-for-byte. OCR'd page canonicals are the OCR text: spans into a scanned doc are byte-true to what the tier read, and the README says so instead of pretending they are glyph-true to the paper. Every document gets a content hash; re-runs only process what changed.
2. **Chunk:** structure-aware. Split on section and paragraph boundaries, pack to target size. No fixed-character windows that cut clauses in half.
3. **Index:** one SQLite file you own. FTS5 for BM25 keyword search. sqlite-vec for embeddings.
4. **Retrieve:** run BM25 and vector search side by side, fuse with reciprocal rank fusion. Optional rerank stage behind `--rerank`, built in v0.4, with its latency and cost metered into every published row. The rerank contract matches the answering contract: the model returns an order list of candidate indices, the code validates and completes the permutation, and a malformed reply after one retry leaves the ranking untouched. A reranker can reorder the shortlist; it can never invent a candidate.
5. **Assemble:** a context budget allocator fills a fixed token budget from ranked spans plus document metadata, and reports what it dropped. This budgeting step is the product; retrieval alone is a solved demo.
6. **Answer:** strict JSON out: `{answer, citations[]}`. The model emits a block label plus a verbatim quote; the tool derives doc id and character span by locating that quote inside the assembled block, and ships the bytes re-read from the canonical file. An answer quoting a span that was not in the assembled context fails the schema check: no citation, no answer. The refusal gate has three teeth: zero keyword support refuses before any model call, a model can refuse in the required form, and anything that fails verification twice is reported REJECTED with reasons, never printed as an answer.

## The corpus (and why it is synthetic)

Demo corpus: generated residential transaction document sets, purchase agreements, contingencies, addenda, closing statements, with facts planted at build time. v0.5 changed the closing statement: the settlement figures gained a five-row fee schedule drawn as positioned cells in column-major stream order, the way real table writers draw, so the corpus itself carries the extraction-damage case the evals measure. `cw scanify` adds the scanned twin: image-only PDFs of the whole corpus, same filenames, planted facts visible only through rasterization.

Synthetic on purpose, three reasons. No MLS data or client PII ever ships in a public repo. The answer key is machine-checkable because the facts were planted, so evals score truth instead of vibes. And version traps become possible: near-duplicate clauses across dated addenda where the naive top-k cites the superseded one.

## Evals (the part that makes this more than a demo)

- **Retrieval:** recall@10 and MRR against the planted key.
- **Grounding:** every citation span is re-checked against the source file byte-for-byte. One bad span fails the run.
- **Refusal:** questions with no answer in the corpus. The correct output is "not in these documents", not a confident paragraph.
- **Adversarial:** superseded-addendum traps, cross-document joins (the answer only exists if two documents are combined), exact figures (dates, dollar amounts, section numbers) where keyword precision beats embeddings.
- **Cost/quality frontier:** the same suite run across tiers: local Ollama embedder + local model, API embedder + local model, full API. Delivered in v0.4 as the table below, in dollars-per-1k-queries vs. score, with every dollar metered from provider usage reports. The CI workflow regenerates the API-tier block weekly; the local-tier blocks regenerate on the dev machine with one script.
- **Extraction price tags (v0.5):** the same suite run through the extraction regimes instead of the model tiers: tables-off vs table reflow, and OCR against the scanned twin of the identical corpus, published as comparison blocks with the commands that produce them.

### Baseline table, local tier

25 deals, 58 PDFs, 212 questions, embedder `ollama:nomic-embed-text` on CPU, zero dollar cost. Raw per-question rows are in `evals/results/latest.json`. This table lives between AUTO markers and is regenerated verbatim by `scripts/regen_tables.sh` (and, for the API tiers, by CI): if the numbers below ever drift from the pipeline, the regeneration command and the stale `--check` gate are printed right here.

<!-- BEGIN AUTO:BASELINE -->
Retrieval eval, run 2026-11-21T22:43:52+00:00. k=10, 25 deals, 212 questions. Python 3.14.9, SQLite 3.53.1, embedder ollama:nomic-embed-text.

| Mode | exact recall@10 | exact MRR | table recall@10 | table row@10 | trap top-1 | trap stale-in-top-3 | join@10 | refusal mean hits |
|---|---|---|---|---|---|---|---|---|
| bm25 | 1.000 | 1.000 | 1.000 | 1.000 | 0.875 | 0.375 | 0.000 | 0.000 |
| vec | 0.889 | 0.733 | 0.560 | 0.400 | 0.625 | 0.125 | 0.840 | 6.667 |
| hybrid | 1.000 | 0.996 | 1.000 | 1.000 | 1.000 | 0.250 | 0.840 | 6.667 |
<!-- END AUTO:BASELINE -->

What the numbers say, including the inconvenient ones:

- **Exact figures and dates favor keywords.** Vector-only drops exact MRR from 1.000 to 0.733 and exact recall to 0.889. Embeddings alone are the wrong call for contracts; this is the documented case for hybrid.
- **The table columns are the v0.5 case.** With `--tables` on, BM25 and hybrid hold a planted row, label AND figure in one quotable chunk, on 100% of table questions (row@10 = 1.000). The embedder-only tier gets rows in only 0.400. Adjacency is keyword terrain.
- **The join column is the fusion case.** No single chunk contains both sides of a cross-document question, so BM25 scores 0.000 on join@10 while vec and hybrid get 0.840.
- **Version traps measured honestly.** BM25 ranks the addendum first on 0.875 of trap questions but drags the superseded clause into the top 3 on 0.375. Fusion fixes the top slot (1.000) and cuts stale-in-top-3 to 0.250. The v0.4 table showed 0.000 in that column; the v0.5 rebuild of the same seed does not reproduce it. That is the honest texture of "measured, not promised": when the data regenerates, the number moves, and the pipeline republishes instead of defending the old cell. Retrieval can lower the risk; the refusal gate in v0.3 is what enforces it at answer time.
- **Refusal is a real gap, shown not hidden.** KNN always returns neighbors: vec and hybrid report about 6.7 retrieved documents for questions about identities that do not exist in the corpus. Retrieval cannot refuse; that is precisely why answering without a grounding gate is not offered.

### Answer eval, local tier, fixed seed

The v0.3 table below was one run per question at whatever the sampler felt like, and the known non-determinism (q0000 passed a manual probe and failed the eval) made that dishonest. v0.4 regenerates this table at seed 42 with 3 runs per question: the headline counts the MAJORITY, the unstable column counts questions whose verdict varied across runs at all, and costs are metered from the provider's own usage report. v0.5 rebuilds it on the new corpus: 17 stratified questions, the pool now including a table kind, same hybrid index (under `--tables`), same `ollama:qwen2.5:7b`.

<!-- BEGIN AUTO:ANSWERS_LOCAL -->
Answer eval, model ollama:qwen2.5:7b, run 2026-11-21T22:40:00+00:00. 17 sampled questions, mode hybrid, k=10, budget 2400 tokens, seed 42, 3 run(s) per question, rerank off.

| Kind | n | correct | refused | rejected | stale cited | unstable |
|---|---|---|---|---|---|---|
| exact | 4 | 2 | 1 | 0 | - | 0 |
| table | 3 | 3 | 0 | 0 | - | 0 |
| trap | 4 | 4 | 0 | 0 | 0 | 0 |
| join | 2 | 0 | 2 | 0 | - | 0 |
| refusal | 4 | 4 | 4 | 0 | - | 0 |

Metered cost: $0.0000 total, 33 model calls, 22035 prompt + 1644 completion tokens.
<!-- END AUTO:ANSWERS_LOCAL -->

The honest read, from the seeded 3-run table above:

- **The gate held.** Zero rejected across all 51 runs: every printed answer's citations byte-checked against the file. 4 of 4 traps cited the amendment and none cited the superseded figure.
- **The table kind answers on the free tier.** All 3 sampled row questions, 3 runs each, correct: label and figure quoted from one reflowed chunk. The extraction stage earns its keep without paying for a better model.
- **Refusal bias is real and stable.** Both join questions were refused by the keyword-support gate in all three runs: no single chunk contains the full term set of a cross-document question, so the conservative anchor says "not in these documents." That is the documented price of refusal-over-fluency. No gate loosening shipped since v0.4; the frontier table shows that paying money does not buy it either, and it stays open work.
- **Instability was measured, not assumed.** The v0.3 note about q0000 passing a manual probe and failing the eval was run variance, and the fixed-seed 3-run harness exists to catch exactly that. It reports 0 unstable questions across all five kinds at seed 42. The seed held here; it is a measurement, not a promise.
- **The two exact misses survive pinning.** One question (q0098) was answered wrong but grounded, a verbatim byte-checked citation that the key scores as incorrect: grounding is not truth, and the key scores facts, not fluency. One (q0148) was a model refusal with the answer present in context, a 7B limitation. Both verdicts were stable 0/3. The tier that fixes them is in the frontier table below, at about $0.22 per 1k queries, not a new local model.

### Why the model never picks offsets

The first contract made the model emit `span` numbers. qwen2.5:7b pointed at real content but miscounted characters, and a correct answer got rejected for arithmetic. The shipped contract: the model emits label + verbatim quote; the tool locates the quote inside the block and derives `doc_id + span` server-side, and the bytes that ship are re-read from the canonical file (whitespace transcription drift is flagged, never patched, content drift fails). 0 rejected across the 51 runs of the v0.5 rebuild, reflowed pipe rows included, which is the row-string contract working: LLMs cannot count characters, so the code stopped asking.

### Cost/quality frontier, three tiers

The v0.4 flagship, rebuilt in v0.5 on the new corpus: the same 17 stratified questions, 3 seeded runs each, across three stacks, with dollars METERED from provider usage reports and the ingest meta rather than modeled from price sheets. Regenerate with `python -m evals --frontier --key corpus_out/answer_key.json --limit 16 --runs 3 --seed 42 --k 10`; raw per-tier rows in `evals/results/frontier.json`.

<!-- BEGIN AUTO:FRONTIER -->
Cost/quality frontier, run 2026-11-21T22:42:47+00:00. hybrid retrieval, seed 42, 3 runs per question, 17 stratified questions, rerank off.

| Tier | Embedder | Chat model | correct | unstable | $/1k queries | ingest $ | median gen ms |
|---|---|---|---|---|---|---|---|
| local | ollama:nomic-embed-text | ollama:qwen2.5:7b | 13/17 | 0 | $0.0000 | $0.0000 | 2653.2 |
| api-embed + local chat | openrouter:openai/text-embedding-3-small | ollama:qwen2.5:7b | 14/17 | 0 | $0.0005 | $0.0007 | 2810.5 |
| full API (cheap) | openrouter:openai/text-embedding-3-small | openrouter:openai/gpt-4o-mini | 15/17 | 0 | $0.2232 | $0.0007 | 1333.8 |

$/1k queries counts per-query spend only (chat + query embedding) times 1000. The ingest column is what one-time corpus embedding cost, metered into index meta at ingest. Dashes mean the provider does not meter dollars; we do not model them. Local tiers are $0 because local compute is.
<!-- END AUTO:FRONTIER -->

What the frontier says, including the inconvenient parts:

- **The cost curve is flat cheap at this scale.** $0.00 per 1k queries local, $0.0005 with an API embedder answering still on the laptop, $0.2232 for the full API. The whole 25-deal corpus embedded for $0.0007 one-time. The full-API pass of the 17-question seeded suite cost $0.0038 metered, 33 model calls (51 answered minus 18 gate-refused makes no call and bills nothing). Frontier numbers here are cents, not dollars, and the honest reading is that at one person's paperwork scale, paying is not the interesting decision. Which tier you pick is a latency and privacy decision, and the gen-ms column says so: the API tier answers faster (1334 ms median) than the local 7B (2653 ms). That gap is the price of keeping the documents on your own machine.
- **The quality gap is two questions, both exact-figure.** Local 13/17, full API 15/17, and the delta sits entirely in the exact kind: the two questions where qwen2.5:7b grabbed the wrong deal or refused, gpt-4o-mini got both, grounded, zero rejects. Table, trap, join, and refusal CORRECT COUNTS are identical across tiers. Buying a bigger model fixes model problems, not gate problems, and this table separates them.
- **One stale row appeared at the paid tier.** The full-API run cited a superseded figure on one of four sampled trap questions (q0134) where the local tiers cited zero. The trap pool is small; one cell is a sample, not a law. It is published because it happened, not because it is comfortable.
- **Joins stay 0/2 at every price.** The keyword gate refuses cross-document questions before ANY model sees them, local or paid. Paying $0.22/1k does not buy that open. That is the honest ceiling of this tier run, and the reason gate work is still on the roadmap.
- **Seed 42 held everywhere.** 0 unstable questions in all three tiers, 51 question-runs each, majority verdicts. Measured, not promised.

### The API tiers, CI-regenerated

`python -m evals --frontier --tiers api-cheap` is the slice of the frontier a GitHub runner can compute without pulling a 4.7 GB model: corpus embed, 51 answered questions, metered dollars. The workflow in `.github/workflows/evals.yml` runs it weekly, syncs the block below between AUTO markers, and commits when the table moves. Claim hygiene: the workflow is written and has NEVER EXECUTED (repo not pushed as of 2026-11-21). The rows below are NOT a separate keyed pass: `scripts/regen_tables.sh` derives this block from the same metered frontier run above (the run stamps match), because re-billing 51 answered questions to publish one row would make the dollar column a lie.

<!-- BEGIN AUTO:FRONTIER_API -->
Cost/quality frontier, run 2026-11-21T22:42:47+00:00. hybrid retrieval, seed 42, 3 runs per question, 17 stratified questions, rerank off.

| Tier | Embedder | Chat model | correct | unstable | $/1k queries | ingest $ | median gen ms |
|---|---|---|---|---|---|---|---|
| full API (cheap) | openrouter:openai/text-embedding-3-small | openrouter:openai/gpt-4o-mini | 15/17 | 0 | $0.2232 | $0.0007 | 1333.8 |

$/1k queries counts per-query spend only (chat + query embedding) times 1000. The ingest column is what one-time corpus embedding cost, metered into index meta at ingest. Dashes mean the provider does not meter dollars; we do not model them. Local tiers are $0 because local compute is.
<!-- END AUTO:FRONTIER_API -->

### Rerank, measured

`--rerank` puts a model between fusion and the budget allocator. It is not free and the numbers say so: every rerank call is metered in dollars and milliseconds in the same rows as everything else. `python -m evals --answer --index cw_corpus_index_api --model openrouter:openai/gpt-4o-mini --rerank openrouter:openai/gpt-4o-mini --limit 16 --runs 3 --seed 42 --k 10` produces this table.

<!-- BEGIN AUTO:RERANK -->
Answer eval, model openrouter:openai/gpt-4o-mini, run 2026-11-21T22:45:52+00:00. 17 sampled questions, mode auto, k=10, budget 2400 tokens, seed 42, 3 run(s) per question, rerank openrouter:openai/gpt-4o-mini.

| Kind | n | correct | refused | rejected | stale cited | unstable |
|---|---|---|---|---|---|---|
| exact | 4 | 4 | 0 | 0 | - | 0 |
| table | 3 | 3 | 0 | 0 | - | 0 |
| trap | 4 | 4 | 0 | 0 | 1 | 0 |
| join | 2 | 0 | 2 | 0 | - | 0 |
| refusal | 4 | 4 | 4 | 0 | - | 0 |

Metered cost: $0.0100 total, 84 model calls, 53917 prompt + 3239 completion tokens.
<!-- END AUTO:RERANK -->

The measured verdict, same 17 questions and same API tier as the frontier table, rerank off versus on:

- **Quality: 15/17 with or without.** The reranker bought nothing on the headline, again.
- **The v0.4 trap-amplification finding did NOT reproduce.** In v0.4, rerank lifted stale citations from 0/4 to 3/4. On the rebuilt key it is 1/4 rerank-off and 1/4 rerank-on, the same question (q0134) in both passes: the superseded figure was already being cited before any reranker touched the window. The honest rewrite of the earlier claim: rerank amplified a trap risk that this corpus stream did not contain; the amplification was a property of a sample, not a law of the stage. Which is exactly why the number gets republished instead of defended.
- **Cost and latency: 2.6x.** $0.0100 versus $0.0038 for the same pass, 51 extra metered calls (84 versus 33), and the reranker moved a median of 8 of 12 candidates per window. Moving most of the window for zero headline gain is the definition of a stage that has not earned its keep.
- **Ship decision, documented:** the flag exists so the trade is visible, and it defaults off. Nothing here argues for turning it on; the case for off got stronger, not weaker. It may still earn its keep on much larger corpora where fusion signal is thinner. Until measured there, off is the honest default.

### Tables, measured

Same seeded corpus, same k=10, one ingestion difference: the naive
column-major closing statements in a tables-off index versus the reflowed
pipe rows in a tables-on index. Regenerate with the commands printed in
the v0.5 tour; `scripts/gen_v05_blocks.py` assembles the numbers below
from the saved report JSONs without recomputing anything.

<!-- BEGIN AUTO:V05_TABLES -->
Table extraction, same seeded corpus (25 deals, 212 questions), k=10. 'table row@10' asks: does any top-10 chunk hold the planted row string, label AND its figure together? 'table recall@10' is the doc-level view.

| Ingest regime | Mode | table row@10 | table recall@10 | exact recall@10 |
|---|---|---|---|---|
| plain (v0.4 behavior) | bm25 | 0.000 | 1.000 | 1.000 |
| --tables layout-reflow | bm25 | 1.000 | 1.000 | 1.000 |
| --tables layout-reflow | vec | 0.400 | 0.560 | 0.889 |
| --tables layout-reflow | hybrid | 1.000 | 1.000 | 1.000 |
<!-- END AUTO:V05_TABLES -->

What the comparison says:

- **The flag moved exactly one number.** Comparing the tables-off and tables-on BM25 runs aggregate by aggregate, the ONLY field that differs is `table_row_top10`: 0.000 becomes 1.000. Doc recall, exact, trap top-1, stale-in-top-3, join, refusal: identical scores, because the reflow only rewrites pages that carry a grid. A targeted repair, measured as targeted.
- **Doc-level recall was blind to the damage, which is the lesson.** Both regimes find the closing statement 100% of the time. A pipeline that graded only document hits would have called column-major extraction perfect while every table row sat shredded across chunks. Grounding to canonical bytes plus a chunk-level row metric is what makes the breakage visible; the answer eval then refuses to quote what the bytes do not contain.
- **The embedder tier still half-misses rows** (vec row@10 0.400, table recall 0.560): aligned short rows are exact-token terrain, the same finding as the exact-figure column, one level down.
- **At answer time the payoff is free.** The 3 sampled table questions in the local-tier answer eval are 3/3 correct at $0.0000: the model quotes a reflowed row verbatim and the gate byte-checks it.

### Scanned documents, measured

The whole corpus rasterized at 250 dpi into image-only PDFs, ingested
through the system OCR tier, scored against the identical seeded key via
the filename-joined scan map. The commands are in the v0.5 tour; the
block is assembled by `scripts/gen_v05_blocks.py` from `latest.json` and
`scanned.json`.

<!-- BEGIN AUTO:V05_OCR -->
OCR roundtrip, same corpus rasterized to image-only PDFs at 250 dpi and ingested through the system OCR tier (doc ids mapped by filename), k=10, run 2026-11-21T22:43:57+00:00.

| Corpus | Mode | exact recall@10 | exact MRR | table row@10 | trap top-1 | join@10 | refusal mean hits |
|---|---|---|---|---|---|---|---|
| text layer | bm25 | 1.000 | 1.000 | 1.000 | 0.875 | 0.000 | 0.000 |
| text layer | vec | 0.889 | 0.733 | 0.400 | 0.625 | 0.840 | 6.667 |
| text layer | hybrid | 1.000 | 0.996 | 1.000 | 1.000 | 0.840 | 6.667 |
| scanned, OCR | bm25 | 1.000 | 1.000 | 0.000 | 0.875 | 0.000 | 0.000 |
| scanned, OCR | vec | 0.923 | 0.670 | 0.000 | 0.625 | 0.800 | 6.583 |
| scanned, OCR | hybrid | 1.000 | 0.996 | 0.000 | 0.875 | 0.800 | 6.583 |
<!-- END AUTO:V05_OCR -->

What the OCR bill looks like, line by line:

- **Facts survive the paper roundtrip.** BM25 exact recall@10 and MRR are 1.000/1.000 on the scanned corpus, identical to the text layer: at 250 dpi with psm 6, tesseract read every planted dollar figure and date byte-exactly into the canonical text, and the verify gate passed all 274 spans against the OCR'd canonicals.
- **Tables survive as facts, not as rows.** Scanned doc-level table recall stays 1.000, but `table row@10` is 0.000: tesseract pairs label and amount with a single space ("Title Insurance $500"), which is good reading, and the pipe-row string simply does not exist in OCR output because there is no column geometry left to align. Reflow operates on text layers; there is nothing to reflow downstream of OCR. That is a real open gap, published, not smoothed.
- **The embedder tier moves a few points in both directions.** Scanned vec exact recall is 0.923 (text: 0.889) while MRR drops to 0.670 (text: 0.733): OCR injects its own whitespace (a blank line after nearly every read line) and the occasional clipped glyph, so the vectors are computed on different bytes even where every word survives. Nobody gets to claim embeddings are robust to extraction churn; here is the churn.
- **Trap top-1 loses one question to OCR.** Text hybrid gets every trap first (1.000); scanned hybrid gets 0.875. The misread header row ("tem Amount") is visible in at least one canonical, and one deal's ranking pays for it. Small pool, real effect, published.
- **The cost is honest: $0.0000.** pdftoppm and tesseract are subprocesses on the laptop; the scanned index embedded with the same free ollama tier. The bill for reading paper is CPU minutes and the two small regressions above, and the gate holds: a scanned span is byte-true to the canonical the tier wrote, and the README says which truth that is.

## Design decisions, documented

- SQLite over a vector-database service: correct for 10^4 to 10^5 documents, which is the scale one person's paperwork lives at. The breakage point is written down, not pretended away.
- Hybrid over vector-only: contracts are full of exact tokens (dates, section numbers, dollar figures). BM25 catches them for free.
- RRF over score fusion: no normalizing incomparable scores across retrievers.
- Refusal over fluency: "not in these documents" is a first-class answer.
- Model quotes, server offsets: the model copies text, the code computes doc id and span. LLM character counting is unreliable, so it is not in the trusted path.
- Keyword support anchors the refusal gate: if BM25 AND finds no chunk holding every query term, the question's identity is not in the corpus and KNN neighbors are the least-wrong documents, not evidence. The gate refuses before the model. The known cost is false refusals on cross-document and paraphrased questions; both are measured in the answer eval above.
- Metered, not modeled: every published dollar comes from a provider usage report or the ingest meta, never from a price sheet multiplied in someone's head. A tier whose dollars cannot be metered shows a dash. A dash is a claim of ignorance, which is always better than a claim of arithmetic.
- No-call rows are $0, unknown-call rows are not: a question refused by the keyword gate made no model call and is zero dollars as a fact. A request that died mid-flight may have been billed anyway, so it is unknown, and one unknown makes the tier's money column a dash.
- Model orders, code permutes: the reranker returns an index list, the code validates, dedupes, and completes it. A malformed rerank reply after one retry leaves the retrieval order untouched rather than half-trusting a broken permutation.
- Instability is published, not assumed: seeds are sent where the API accepts them, but the eval measures whether the model actually repeated itself (unstable column) instead of claiming determinism it cannot verify upstream.
- Text layers always win: the OCR tier is consulted only on pages whose text layer is thinner than 24 chars. A real text layer is the PDF's own bytes; an OCR read of the same page is a second-hand copy, so the tool never trades up.
- One extraction regime per index: `--ocr` and `--tables` settings are stamped in meta at first ingest and a disagreeing re-ingest raises. Canonical texts under two regimes in one index is two truths, and citations are only worth what their ground truth is worth.
- Adjacency is a byte-level fact: the reflowed pipe row lives IN the canonical text, so label-and-figure pairing is quotable, byte-checked, and graded at chunk level. A metric that only looks at document ids cannot see a broken table, so the eval does not look at only that.
- Report orphans, never delete: `generate-corpus` into a dirty directory lists the PDFs it cannot claim instead of removing them. Deleting files the tool did not write is not a library's business; letting them poison an index silently is.

## Stack

- Python 3.12+, no runtime dependencies required for the search layer
- pypdf (ingest), SQLite FTS5 + sqlite-vec (index)
- optional OCR: system pdftoppm (poppler-utils) + tesseract via subprocess, zero python deps; optional table reflow: pypdf layout mode
- Bring-your-own model interface: OpenAI API, OpenRouter, or Ollama, shared pattern with open_otto; OpenRouter meters exact dollars per call
- pytest, GitHub Actions (workflow written 2026-11-21, updated for v0.5 the same day, unexecuted until the repo is pushed; CI installs poppler-utils + tesseract-ocr so the OCR roundtrip tests run rather than skip)

## Roadmap

- [x] v0.1: ingest, structure-aware chunking, FTS5 index, CLI search with cited spans (BM25 only). Built and tested 2026-11-20.
- [x] v0.2: embeddings + hybrid fusion; synthetic corpus generator; eval harness with published baseline table. Built and tested 2026-11-20, table generated 2026-11-21.
- [x] v0.3: token-budget context assembler + grounded answering CLI with refusal gate. Built and tested 2026-11-21, answer eval table generated same day.
- [x] v0.4: cost/quality benchmark across model tiers, CI-regenerated README table, fixed-seed answer evals, the retrieve rerank flag with its latency and cost documented. Built and tested 2026-11-21; frontier tables generated the same day from a live metered OpenRouter account. The CI workflow file exists but has not executed (nothing pushed).
- [x] v0.5: optional OCR and table extraction. Built and tested 2026-11-21: pdftoppm + tesseract tier with regime guards and per-page thresholds, coordinate-aware pipe-row reflow, column-major settlement tables planted in the corpus with a labeling gate that re-runs the real extraction, the image-only scanned twin of the whole corpus scored against its text original, and the two comparison blocks in the evals section generated by script. No v0.6 is planned; what is left is named in the tables: the join gate still refuses cross-document questions at every price, and the OCR cost is measured, not eliminated.

## Non-goals

- Chat UI or "GPT wrapper" theater; CLI and library first
- A hosted version
- Acting on retrieved context with agents; that is open_otto's job. This is the retrieval and grounded answering layer, and it acts on nothing
- Fine-tuning
- Compliance claims. This tool cites. It does not certify

## License

AGPL-3.0. The full text is in [LICENSE](LICENSE). Copyright 2026 Tim
Lumnah. Free to use, modify, and self-host. Anyone who runs a modified
version as a service publishes their changes.
