Table extraction, same seeded corpus (25 deals, 212 questions), k=10. 'table row@10' asks: does any top-10 chunk hold the planted row string, label AND its figure together? 'table recall@10' is the doc-level view.

| Ingest regime | Mode | table row@10 | table recall@10 | exact recall@10 |
|---|---|---|---|---|
| plain (v0.4 behavior) | bm25 | 0.000 | 1.000 | 1.000 |
| --tables layout-reflow | bm25 | 1.000 | 1.000 | 1.000 |
| --tables layout-reflow | vec | 0.400 | 0.560 | 0.889 |
| --tables layout-reflow | hybrid | 1.000 | 1.000 | 1.000 |
