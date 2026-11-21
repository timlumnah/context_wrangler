#!/usr/bin/env bash
# v0.5 job B: the API-keyed legs. api-embedded reflow index, then the
# full README regeneration (frontier, api tier, naive/scanned evals,
# v0.5 blocks, rerank, sync+check). OpenRouter dollars are metered; the
# whole job is budgeted under $0.05.
set -exuo pipefail
cd /home/magnolia_gold/dev/claude_5-5-26/context_wrangler
export OPENROUTER_API_KEY="$(cat ../open_router_key)"
CW=.venv/bin/cw

"$CW" ingest corpus_out/pdfs --index cw_corpus_index_api \
    --embed openrouter:openai/text-embedding-3-small --tables
"$CW" verify --index cw_corpus_index_api

OPENROUTER_API_KEY="$OPENROUTER_API_KEY" scripts/regen_tables.sh

# the live CLI probe uses the FIRST exact question of the freshly seeded
# corpus, never a hard-coded identity from a previous seed stream
QUERY=$(.venv/bin/python -c "
import json
k = json.load(open('corpus_out/answer_key.json'))
print(next(q['query'] for q in k['questions'] if q['kind'] == 'exact'))")
echo "probe query: $QUERY"
"$CW" answer "$QUERY" \
    --index cw_corpus_index_api --model openrouter:openai/gpt-4o-mini \
    --rerank openrouter:openai/gpt-4o-mini --seed 42
"$CW" answer "$QUERY" \
    --index cw_corpus_index --model ollama:qwen2.5:7b --seed 42

curl -s -m 20 https://openrouter.ai/api/v1/key \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" | head -c 400
echo; echo "JOB B DONE"
