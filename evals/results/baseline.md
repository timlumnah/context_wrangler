Retrieval eval, run 2026-11-21T22:43:52+00:00. k=10, 25 deals, 212 questions. Python 3.14.9, SQLite 3.53.1, embedder ollama:nomic-embed-text.

| Mode | exact recall@10 | exact MRR | table recall@10 | table row@10 | trap top-1 | trap stale-in-top-3 | join@10 | refusal mean hits |
|---|---|---|---|---|---|---|---|---|
| bm25 | 1.000 | 1.000 | 1.000 | 1.000 | 0.875 | 0.375 | 0.000 | 0.000 |
| vec | 0.889 | 0.733 | 0.560 | 0.400 | 0.625 | 0.125 | 0.840 | 6.667 |
| hybrid | 1.000 | 0.996 | 1.000 | 1.000 | 1.000 | 0.250 | 0.840 | 6.667 |
