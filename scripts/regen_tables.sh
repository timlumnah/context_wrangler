#!/usr/bin/env bash
# Regenerate every README auto-block from one run. Needs: ollama with
# nomic-embed-text + qwen2.5:7b pulled, a corpus + both indexes already
# built (the README tours), and OPENROUTER_API_KEY in the environment.
#
#   OPENROUTER_API_KEY=*** scripts/regen_tables.sh
#
# CI (evals.yml) regenerates only FRONTIER_API; this script owns the
# local-tier blocks that a shared runner should not compute.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY required"; exit 1; }

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

"$PY" -m evals --frontier --key corpus_out/answer_key.json \
    --limit 16 --runs 3 --seed 42 --k 10 \
    --json evals/results/frontier.json --md evals/results/frontier.md

# The CI slice: same frontier run's api-cheap row, derived, never
# re-billed. The full frontier pass above already paid for the 51
# answered questions; a second api-only pass would double that tier's
# dollars for a block that shows one row.
"$PY" - <<'EOF'
import json
from evals.frontier import to_markdown as md_frontier
front = json.load(open("evals/results/frontier.json"))
api_rows = [r for r in front["rows"] if r["tier"] == "full API (cheap)"]
assert api_rows, "no api-cheap tier in this frontier run"
slice_ = dict(front, rows=api_rows)
md = md_frontier(slice_)
with open("evals/results/frontier_api.md", "w", encoding="utf-8") as fh:
    fh.write(md + "\n")
with open("evals/results/frontier_api.json", "w", encoding="utf-8") as fh:
    json.dump(slice_, fh, indent=2, default=str)
print("frontier_api derived from the same run")
EOF

# The local answer table is frontier tier A's own report: extracting it
# keeps the published ANSWERS_LOCAL and FRONTIER rows from the SAME runs,
# and saves re-billing 48 ollama generations a second time.
"$PY" - <<'EOF'
import json
from evals.answers import to_markdown
front = json.load(open("evals/results/frontier.json"))
rep = front["rows"][0]["report"]
with open("evals/results/answers_local.md", "w") as fh:
    fh.write(to_markdown(rep) + "\n")
with open("evals/results/answers_local.json", "w") as fh:
    json.dump(rep, fh, indent=2)
print("answers_local extracted from frontier tier A")
EOF

"$PY" -m evals --index cw_corpus_index --key corpus_out/answer_key.json \
    --modes bm25 vec hybrid --json evals/results/latest.json \
    --md evals/results/baseline.md

# v0.5 blocks: the tables-off comparison index and the scanned twins are
# built once (README v0.5 tour commands), this recomputes their numbers.
"$PY" -m evals --index cw_corpus_index_naive --key \
    corpus_out/answer_key.json --modes bm25 \
    --json evals/results/naive.json --md evals/results/naive.md
"$PY" -m evals --index cw_corpus_index_scanned --key \
    corpus_out/answer_key.json --modes bm25 vec hybrid --scan-map \
    --json evals/results/scanned.json --md evals/results/scanned.md
"$PY" scripts/gen_v05_blocks.py

"$PY" -m evals --answer --index cw_corpus_index_api \
    --model openrouter:openai/gpt-4o-mini --key corpus_out/answer_key.json \
    --limit 16 --runs 3 --seed 42 --k 10 --rerank openrouter:openai/gpt-4o-mini \
    --json evals/results/answers_rerank.json --md evals/results/answers_rerank.md

"$PY" -m evals.readme_sync --write \
    BASELINE=evals/results/baseline.md \
    ANSWERS_LOCAL=evals/results/answers_local.md \
    FRONTIER=evals/results/frontier.md \
    FRONTIER_API=evals/results/frontier_api.md \
    RERANK=evals/results/answers_rerank.md \
    V05_TABLES=evals/results/v05_tables.md \
    V05_OCR=evals/results/v05_ocr.md

"$PY" -m evals.readme_sync --check \
    BASELINE=evals/results/baseline.md \
    ANSWERS_LOCAL=evals/results/answers_local.md \
    FRONTIER=evals/results/frontier.md \
    FRONTIER_API=evals/results/frontier_api.md \
    RERANK=evals/results/answers_rerank.md \
    V05_TABLES=evals/results/v05_tables.md \
    V05_OCR=evals/results/v05_ocr.md

echo "all README auto-blocks regenerated and verified"
