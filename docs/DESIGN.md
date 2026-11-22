# Design decisions

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
