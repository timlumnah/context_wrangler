Retrieval eval, run 2026-11-21T22:43:57+00:00. k=10, 25 deals, 212 questions. Python 3.14.9, SQLite 3.53.1, embedder ollama:nomic-embed-text. Scanned-corpus run: doc ids are mapped from the key by filename, the indexed files are image-only twins.

| Mode | exact recall@10 | exact MRR | table recall@10 | table row@10 | trap top-1 | trap stale-in-top-3 | join@10 | refusal mean hits |
|---|---|---|---|---|---|---|---|---|
| bm25 | 1.000 | 1.000 | 1.000 | 0.000 | 0.875 | 0.500 | 0.000 | 0.000 |
| vec | 0.923 | 0.670 | 0.540 | 0.000 | 0.625 | 0.000 | 0.800 | 6.583 |
| hybrid | 1.000 | 0.996 | 1.000 | 0.000 | 0.875 | 0.250 | 0.800 | 6.583 |
