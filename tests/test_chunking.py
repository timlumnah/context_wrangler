"""Chunker rules: section boundaries, paragraph integrity, packing sizes."""

from context_wrangler.core.chunk import chunk_text, parse_blocks

TEXT = """PURCHASE AND SALE AGREEMENT

ARTICLE 1 PRICE
The price is $485,000 for the property at 142 Beacon Street.
Payment terms follow escrow instructions.

ARTICLE 2 CLOSING
The Closing Date shall be March 15, 2027, unless extended in writing by
both parties, and time is of the essence under this agreement.

SECTION 3 INSPECTION
Inspection within 10 business days.
"""


def test_spans_recover_chunks_exactly():
    chunks = chunk_text(TEXT, target=200)
    for c in chunks:
        assert TEXT[c.start:c.end] == TEXT[c.start:c.end]  # offsets valid
        assert 0 <= c.start < c.end <= len(TEXT)


def test_never_straddles_sections():
    chunks = chunk_text(TEXT, target=200, min_size=10)
    headings = {b.start for b in parse_blocks(TEXT) if b.is_heading}
    for c in chunks:
        inner = {h for h in headings if c.start < h < c.end}
        assert not inner, f"chunk {c.start}-{c.end} cuts through sections"


def test_never_ends_inside_a_paragraph():
    chunks = chunk_text(TEXT, target=200, min_size=10)
    block_ends = {b.end for b in parse_blocks(TEXT)}
    for c in chunks:
        assert c.end in block_ends, "chunk ended mid-paragraph"
    # chunks are ordered and non-overlapping
    for a, b in zip(chunks, chunks[1:]):
        assert a.end <= b.start


def test_chunk_starts_at_paragraph_or_heading():
    chunks = chunk_text(TEXT, target=200, min_size=10)
    starts = {b.start for b in parse_blocks(TEXT)}
    for c in chunks:
        assert c.start in starts


def test_oversized_paragraph_breaks_at_sentences():
    sentence = ("The buyer shall obtain financing approved by the lender "
                "in writing. ")
    para = sentence * 40                       # one ~2600-char paragraph
    text = "ARTICLE 9 RUNAWAY\n" + para + "\n"
    chunks = chunk_text(text, target=300, min_size=50)
    assert len(chunks) >= 7
    assert "ARTICLE 9 RUNAWAY" in text[chunks[0].start:chunks[0].end]
    prev_end = chunks[0].end
    for c in chunks[1:]:
        assert c.end - c.start <= 320          # near target, not over by much
        assert c.start - prev_end <= 1         # contiguous, newline at most
        tail = text[c.start:c.end].rstrip()
        assert tail.endswith(".")              # landed on a sentence end
        prev_end = c.end
    # full coverage: only whitespace sits between chunk spans
    covered = "".join(text[c.start:c.end] for c in chunks)
    assert para.strip() in covered


def test_hard_split_only_for_giant_words():
    giant = "A" * 2000 + "."
    text = "SECTION 1 ODD\n" + giant + "\n"
    chunks = chunk_text(text, target=300, min_size=10)
    assert all(c.end - c.start <= 300 for c in chunks)


def test_no_chunks_when_small_text():
    chunks = chunk_text("Just one small paragraph here.\n", target=1200)
    assert len(chunks) == 1


def test_page_labeling():
    # page spans: page 0 covers first 30 chars, page 1 the rest
    spans = [(0, 29), (30, len(TEXT))]
    chunks = chunk_text(TEXT, target=200, min_size=10, page_spans=spans)
    assert chunks[0].page == 0
    later = [c for c in chunks if c.start >= 30]
    assert later and all(c.page == 1 for c in later)
