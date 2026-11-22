#!/bin/bash
# Plays the demo with typed-out commands. Record it from the repo root with:
#   asciinema rec -q -y --overwrite --cols 100 --rows 34 -c "bash docs/record_demo.sh" docs/demo.cast
#   asciinema-agg --font-size 15 --idle-time-limit 2 docs/demo.cast docs/demo.gif
# Needs the .venv from the README, plus a running Ollama with qwen2.5:7b (local, $0, no key).
# Works in a throwaway directory; BM25 only, so no embedding model is needed.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$REPO/.venv/bin:$PATH"
WORK="$(mktemp -d)"; cd "$WORK" || exit 1
type_cmd() { printf '\033[1;32m$\033[0m '; for ((i=0;i<${#1};i++)); do printf '%s' "${1:i:1}"; sleep 0.03; done; sleep 0.5; echo; }
say() { printf '\033[1;36m# %s\033[0m\n' "$1"; sleep 1.5; }
M="--index idx --model ollama:qwen2.5:7b --mode bm25"

say "Synthetic closing documents with planted facts, ingested with table reflow."
type_cmd "cw generate-corpus out --deals 4 --seed 42"; cw generate-corpus out --deals 4 --seed 42; sleep 1
type_cmd "cw ingest out/pdfs --index idx --tables";    cw ingest out/pdfs --index idx --tables; sleep 2
echo
say "A grounded answer: the model quotes, the code derives the cited span."
type_cmd "cw answer 'earnest money deposit Whitfield 239 Granite' $M"
cw answer earnest money deposit Whitfield 239 Granite $M; sleep 3
echo
say "A table row, quoted from one reflowed chunk."
type_cmd "cw answer 'title insurance amount Whitfield 239 Granite' $M"
cw answer title insurance amount Whitfield 239 Granite $M; sleep 3
echo
say "A question about a deal that does not exist. Refused before any model call."
type_cmd "cw answer 'earnest money deposit Hargrove 14 Elm' $M"
cw answer earnest money deposit Hargrove 14 Elm $M; sleep 3
echo
say "Every indexed span re-checked against the source bytes."
type_cmd "cw verify --index idx"; cw verify --index idx; sleep 3
rm -rf "$WORK"
