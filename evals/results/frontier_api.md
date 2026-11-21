Cost/quality frontier, run 2026-11-21T22:42:47+00:00. hybrid retrieval, seed 42, 3 runs per question, 17 stratified questions, rerank off.

| Tier | Embedder | Chat model | correct | unstable | $/1k queries | ingest $ | median gen ms |
|---|---|---|---|---|---|---|---|
| full API (cheap) | openrouter:openai/text-embedding-3-small | openrouter:openai/gpt-4o-mini | 15/17 | 0 | $0.2232 | $0.0007 | 1333.8 |

$/1k queries counts per-query spend only (chat + query embedding) times 1000. The ingest column is what one-time corpus embedding cost, metered into index meta at ingest. Dashes mean the provider does not meter dollars; we do not model them. Local tiers are $0 because local compute is.
