OCR roundtrip, same corpus rasterized to image-only PDFs at 250 dpi and ingested through the system OCR tier (doc ids mapped by filename), k=10, run 2026-11-21T22:43:57+00:00.

| Corpus | Mode | exact recall@10 | exact MRR | table row@10 | trap top-1 | join@10 | refusal mean hits |
|---|---|---|---|---|---|---|---|
| text layer | bm25 | 1.000 | 1.000 | 1.000 | 0.875 | 0.000 | 0.000 |
| text layer | vec | 0.889 | 0.733 | 0.400 | 0.625 | 0.840 | 6.667 |
| text layer | hybrid | 1.000 | 0.996 | 1.000 | 1.000 | 0.840 | 6.667 |
| scanned, OCR | bm25 | 1.000 | 1.000 | 0.000 | 0.875 | 0.000 | 0.000 |
| scanned, OCR | vec | 0.923 | 0.670 | 0.000 | 0.625 | 0.800 | 6.583 |
| scanned, OCR | hybrid | 1.000 | 0.996 | 0.000 | 0.875 | 0.800 | 6.583 |
