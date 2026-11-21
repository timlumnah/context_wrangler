"""Token-budget context assembler: the step that turns retrieval into a
product, per the README.

Contract: fill a fixed token budget from ranked spans plus document
metadata, and report every span dropped. Token math is a chars/4
heuristic by standing decision (no tokenizer dependency); it is a budget,
not a promise to a specific model's counter, and the model gets no more
text than the budget names.

Rules, in order of application:
  1. Reserve prompt scaffolding (system + question) before any block.
  2. Blocks go in retrieval order. A block costs its text plus a label
     line (doc id, span, section); both counted.
  3. Per-document cap keeps one long agreement from eating the budget and
     starving the addendum.
  4. Spans from one document that touch or overlap merge into a single
     block, counted once.
  5. Everything not admitted is in `dropped`, with a reason. Silence about
     what you cut is how a context tool starts lying.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

DEFAULT_BUDGET = 2400          # tokens (heuristic)
DEFAULT_PER_DOC_CAP = 2
LABEL_OVERHEAD_TOKENS = 24     # label line allowance per block
SCAFFOLD_TOKENS = 800          # system prompt + question + answer headroom


def tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


@dataclass
class Block:
    doc_id: str
    filename: str
    span: tuple[int, int]
    section: str | None
    text: str

    @property
    def cost(self) -> int:
        return tokens(self.text) + LABEL_OVERHEAD_TOKENS


@dataclass
class Assembled:
    blocks: list[Block] = field(default_factory=list)
    dropped: list[tuple[Block, str]] = field(default_factory=list)
    used: int = 0
    budget: int = DEFAULT_BUDGET
    capacity: int = 0

    @property
    def labels(self) -> list[str]:
        """Stable citation labels: c1, c2... in budget order."""
        return [f"c{i + 1}" for i in range(len(self.blocks))]

    def block_by_label(self, label: str) -> Block | None:
        try:
            i = int(label.removeprefix("c")) - 1
        except ValueError:
            return None
        if 0 <= i < len(self.blocks):
            return self.blocks[i]
        return None


def _merge_touching(blocks: list[Block],
                    canonical: dict[str, str] | None) -> list[Block]:
    """Grow one anchor block per document as touching spans arrive, in
    rank order. The merged block keeps the anchor's position so retrieval
    ordering survives.

    The invariant is sacred: Block.text must equal canonical[span]
    byte-for-byte, because every citation gets verified that way. Merge
    only when the text is rebuildable exactly: with a canonical map, from
    the union span; without one, only across zero-gap adjacency, where
    concatenation is provably the canonical bytes. Never invent a byte.
    """
    out: list[Block] = []
    anchor: dict[str, int] = {}          # doc_id -> index of its block in out
    for b in blocks:
        pos = anchor.get(b.doc_id)
        if pos is not None:
            p = out[pos]
            merged = None
            if canonical is not None and b.span[0] <= p.span[1] + 1 \
                    and b.span[1] >= p.span[0]:
                union = (p.span[0], max(p.span[1], b.span[1]))
                merged = Block(p.doc_id, p.filename, union, p.section,
                               canonical[p.doc_id][union[0]:union[1]])
            elif b.span[0] == p.span[1]:
                union = (p.span[0], b.span[1])
                if len(p.text) + len(b.text) == union[1] - union[0]:
                    merged = Block(p.doc_id, p.filename, union, p.section,
                                   p.text + b.text)
            if merged is not None:
                out[pos] = merged
                continue
        out.append(b)
        anchor[b.doc_id] = len(out) - 1
    return out


def assemble(candidates: list[dict], *, budget: int = DEFAULT_BUDGET,
             per_doc_cap: int = DEFAULT_PER_DOC_CAP,
             min_block_tokens: int = 8,
             canonical: dict[str, str] | None = None) -> Assembled:
    """candidates: retrieval records in rank order (hybrid or bm25).
    canonical: doc_id -> full canonical text, enabling exact merges across
    small gaps. Returns the assembled context with a full drop report."""
    capacity = budget - SCAFFOLD_TOKENS
    result = Assembled(budget=budget, capacity=capacity)

    queue: list[Block] = []
    for c in candidates:
        b = Block(c["doc_id"], c["filename"], tuple(c["span"]),
                  c.get("section"), c["text"])
        if b.span[1] - b.span[0] < min_block_tokens * 4:
            result.dropped.append((b, "too_small"))
            continue
        queue.append(b)

    # merge touching same-doc spans BEFORE the cap and cost passes, so one
    # merged document block does not spend two slots
    queue = _merge_touching(queue, canonical)

    seen_per_doc: dict[str, int] = {}
    for b in queue:
        if seen_per_doc.get(b.doc_id, 0) >= per_doc_cap:
            result.dropped.append((b, "per_doc_cap"))
            continue
        seen_per_doc[b.doc_id] = seen_per_doc.get(b.doc_id, 0) + 1
        if result.used + b.cost > capacity:
            result.dropped.append((b, "budget"))
            continue
        result.blocks.append(b)
        result.used += b.cost
    return result


def render_context(assembled: Assembled) -> str:
    """The prompt-visible context. Labels match block_by_label so verified
    citations map back to byte offsets."""
    parts = []
    for label, b in zip(assembled.labels, assembled.blocks):
        head = (f"[{label}] {b.filename} span {b.span[0]}-{b.span[1]}"
                f" section: {b.section or '-'}")
        parts.append(head + "\n" + b.text)
    return "\n\n".join(parts)
