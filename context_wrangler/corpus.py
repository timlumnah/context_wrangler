"""Synthetic residential-transaction corpus with planted, machine-checkable
answer keys.

Public repos ship zero real deal data: no MLS pulls, no client PII. Instead
the generator plants facts at build time into generated PDFs and records the
answer key alongside. That makes every eval question checkable by exact
string identity, and it makes version traps possible on purpose: addenda
that supersede a clause in the agreement, where a naive top-k that cites
the original is wrong and the eval says so.

Determinism: same seed -> same PDF bytes -> same content hashes -> same key.

v0.5 adds the settlement table: label-value grids drawn column-major,
the adversarial-but-real case, plus `table` questions whose expected
text is the reflowed row ("Title Insurance | $410"). The generator's
labeling gate re-extracts every table question through the actual
tables.reflow path, so a key that drifted from what ingest produces
fails the build, not the eval.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------- PDF bytes

@dataclass
class Table:
    """A grid of cells. The builder draws it positioned, cell by cell, in
    COLUMN-MAJOR stream order (every cell of column 0, then column 1):
    that is what real table-writing PDF writers do, and it is the case
    v0.5's --tables reflow exists to fix. The text layer, read naively,
    loses every label-value pairing; the coordinates still hold the grid."""
    rows: list[list[str]]
    col_x: tuple[float, ...] = (72, 300)

    def __post_init__(self):
        if not self.rows or any(len(r) != len(self.col_x) for r in self.rows):
            raise ValueError("table rows must all match col_x width")


def _esc(s: str) -> str:
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf_bytes(pages: list[list[str | Table]]) -> bytes:
    """Valid single-font PDF. Text lines flow one per PDF line like ever
    before; a Table item is emitted as positioned cells in column-major
    order, in its own block, and the flow cursor resumes below it."""
    objs: dict[int, bytes] = {}
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode()
    objs[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    for i, lines in enumerate(pages):
        page_obj = 4 + 2 * i
        cont_obj = 5 + 2 * i
        parts: list[str] = []
        y = 760.0
        flow_open = False

        def open_flow(y_at: float) -> None:
            nonlocal flow_open
            parts.append(f"BT /F1 11 Tf 50 {y_at:.2f} Td 14 TL")
            flow_open = True

        def close_flow() -> None:
            nonlocal flow_open
            if flow_open:
                parts.append("ET")
                flow_open = False

        for ln in lines:
            if isinstance(ln, Table):
                close_flow()
                # column-major: all of column 0 top-to-bottom, then column 1
                for c, x in enumerate(ln.col_x):
                    for r, row in enumerate(ln.rows):
                        y_at = y - 14.0 * (r + 1)
                        parts.append(
                            f"BT /F1 11 Tf {x:.2f} {y_at:.2f} Td "
                            f"({_esc(row[c])}) Tj ET")
                y -= 14.0 * (len(ln.rows) + 1)
                continue
            if not flow_open:
                open_flow(y)
            text = ln if ln else " "
            parts.append(f"({_esc(text)}) Tj T*")
            y -= 14.0
        close_flow()
        stream = "\n".join(parts).encode()
        objs[page_obj] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {cont_obj} 0 R >>").encode()
        objs[cont_obj] = (
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
            + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objs[num] + b"\nendobj\n"
    xref_pos = len(out)
    total = len(objs) + 1
    out += f"xref\n0 {total}\n".encode()
    out += b"0000000000 65535 f \n"
    for num in sorted(objs):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {total} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n").encode()
    return bytes(out)


def make_pdf(path: str, pages: list[list[str | Table]]) -> None:
    with open(path, "wb") as fh:
        fh.write(build_pdf_bytes(pages))


# ---------------------------------------------------------------- vocab

FIRST = ["Alice", "Marcus", "Priya", "Dana", "Victor", "Nora", "Felix",
         "Rosa", "Ken", "Ingrid", "Omar", "Tessa", "Bruno", "Leila",
         "Hank", "Mina", "Cody", "Rafa", "Gwen", "Silas"]
LAST = ["Whitfield", "Okafor", "Lindqvist", "Marchetti", "Nguyen",
        "Bramble", "Kowalski", "Santos", "Fitzgerald", "Hendricks",
        "Yoon", "Castellano", "Amir", "Petrov", "Delgado", "Ashford",
        "Reyes", "Thornbury", "Vasquez", "Kirby"]
STREETS = ["Beacon", "Calder", "Elm", "Foundry", "Granite", "Harbor",
           "Ironwood", "Juniper", "Kestrel", "Larkspur", "Mott",
           "Norwood", "Orchard", "Pine Hollow", "Quarry", "Rowan",
           "Saltmarsh", "Temple", "Underhill", "Verge"]
KINDS = ["Street", "Avenue", "Road", "Lane", "Way"]
BANKS = ["Quinsam Savings", "Blackwood Credit Union", "Harborline Mortgage",
         "Cedar Fork Bank", "Meridiant Lending", "Anchor & Wheel Trust"]
TITLE_CO = ["Beacon Title Company", "Foundry Settlement Group",
            "Granite Escrow Services", "Rowan & Associates Title",
            "Kestrel Closing Bureau"]

# pools used only for refusal questions: these never appear in a corpus
GHOST_LAST = ["Zadvorsky", "Flaminius", "Gronberg", "Pemberton-Smythe",
              "Tritopoulos"]
GHOST_STREETS = ["Wollaston", "Xanthus", "Yarrow Point", "Zircon"]


def money(n: int) -> str:
    return f"${n:,}"


# ---------------------------------------------------------------- generator

@dataclass
class Deal:
    deal_id: str
    buyer: str
    seller: str
    address: str
    price: int
    earnest: int
    closing: datetime.date
    insp_days: int
    fin_pct: int
    fin_deadline: datetime.date
    lender: str
    agent: str
    costs: int                    # cents
    financed: int
    settlement_no: str
    amended: str | None           # "earnest" | "closing" | None
    new_earnest: int = 0
    new_closing: datetime.date | None = None
    title_ins: int = 0            # settlement table, whole dollars
    transfer_tax: int = 0
    recording_fee: int = 0
    escrow_fee: int = 0
    reconveyance: int = 0


@dataclass
class Corpus:
    pdfs: dict[str, bytes] = field(default_factory=dict)   # filename -> bytes
    doc_ids: dict[str, str] = field(default_factory=dict)  # filename -> sha256
    questions: list[dict] = field(default_factory=list)
    deals: list[Deal] = field(default_factory=list)


def _pick_dates(rng: random.Random) -> tuple[datetime.date, datetime.date]:
    year = rng.choice([2026, 2027, 2028])
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    closing = datetime.date(year, month, day)
    fin_deadline = closing - datetime.timedelta(days=rng.randint(30, 60))
    return closing, fin_deadline


def fmt_date(d: datetime.date) -> str:
    return d.strftime("%B %d, %Y").replace(" 0", " ")


def _make_deal(rng: random.Random, i: int) -> Deal:
    buyer = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    seller = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    address = f"{rng.randint(11, 980)} {rng.choice(STREETS)} {rng.choice(KINDS)}"
    price = rng.randint(320, 890) * 1000 + rng.choice([0, 500])
    earnest = max(2500, round(price * rng.randint(1, 3) / 100 / 250) * 250)
    closing, fin_deadline = _pick_dates(rng)
    amended = rng.choice([None, None, "earnest", "closing"])
    deal = Deal(
        deal_id=f"d{i:03d}", buyer=buyer, seller=seller, address=address,
        price=price, earnest=earnest, closing=closing,
        insp_days=rng.randint(7, 14), fin_pct=rng.choice([80, 85, 90, 95]),
        fin_deadline=fin_deadline, lender=rng.choice(BANKS),
        agent=rng.choice(TITLE_CO),
        costs=rng.randint(480000, 1420000),
        financed=int(price * rng.choice([80, 85, 90, 95]) / 100),
        settlement_no=f"{closing.year}-{closing.month:02d}{closing.day:02d}"
                      f"-{rng.randint(1000, 9999)}",
        amended=amended,
        title_ins=rng.randrange(220, 760, 10),
        transfer_tax=int(round(price * rng.choice([0.0010, 0.0015,
                                                   0.0020]))),
        recording_fee=rng.choice([90, 105, 122, 155, 185]),
        escrow_fee=rng.choice([280, 310, 350, 390]),
        reconveyance=rng.choice([75, 85, 95, 125]))
    if amended == "earnest":
        deal.new_earnest = earnest + rng.choice([2500, 5000, 7500])
    elif amended == "closing":
        deal.new_closing = closing + datetime.timedelta(
            days=rng.randint(15, 45))
    return deal


def _purchase_agreement_pages(d: Deal) -> list[list[str]]:
    p1 = [
        "PURCHASE AND SALE AGREEMENT",
        "",
        f"Agreement between {d.buyer} (Buyer) and {d.seller} (Seller) for",
        f"the property at {d.address}.",
        "",
        "ARTICLE 1 PURCHASE PRICE",
        f"Buyer agrees to pay the total purchase price of {money(d.price)}",
        "for the property, payable at closing.",
        "",
        "ARTICLE 2 EARNEST MONEY",
        f"The Earnest Money Deposit is {money(d.earnest)}, payable to the",
        "escrow agent within 72 hours of acceptance of this offer.",
        "",
        "ARTICLE 3 CLOSING",
        f"The Closing Date shall be {fmt_date(d.closing)}, unless extended",
        "in writing by both parties.",
    ]
    p2 = [
        "ARTICLE 4 CONTINGENCIES",
        f"This offer is contingent on a home inspection completed within",
        f"{d.insp_days} business days of acceptance.",
        "",
        f"Financing contingency: Buyer must obtain a loan commitment for",
        f"{d.fin_pct} percent of the purchase price by {fmt_date(d.fin_deadline)}.",
        "",
        "ARTICLE 5 DISCLOSURES",
        "Seller shall deliver the lead paint disclosure required by federal",
        "law within five business days of acceptance.",
    ]
    return [p1, p2]


def _addendum_pages(d: Deal) -> list[list[str]]:
    head = [
        "ADDENDUM ONE TO PURCHASE AND SALE AGREEMENT",
        "",
        f"Date: {fmt_date(d.closing - datetime.timedelta(days=14))}",
        f"Between {d.buyer} and {d.seller}, regarding {d.address}.",
        "",
    ]
    if d.amended == "earnest":
        body = [
            "SECTION 1 AMENDMENT OF EARNEST MONEY",
            f"The Earnest Money Deposit is amended to {money(d.new_earnest)}.",
            "The amount stated in Article 2 is superseded.",
            "",
            "SECTION 2 NO OTHER CHANGES",
            "All other terms and conditions of the Agreement remain",
            "unchanged.",
        ]
    else:
        body = [
            "SECTION 1 AMENDMENT OF CLOSING DATE",
            f"The Closing Date is amended to {fmt_date(d.new_closing)}.",
            "The date stated in Article 3 is superseded.",
            "",
            "SECTION 2 NO OTHER CHANGES",
            "All other terms and conditions of the Agreement remain",
            "unchanged.",
        ]
    return [head + body]


def _closing_statement_pages(d: Deal) -> list[list[str | Table]]:
    dollars, cents = divmod(d.costs, 100)
    settlement_table = Table([
        ["Item", "Amount"],
        ["Title Insurance", money(d.title_ins)],
        ["Transfer Tax", money(d.transfer_tax)],
        ["Recording Fee", money(d.recording_fee)],
        ["Escrow Fee", money(d.escrow_fee)],
        ["Reconveyance Fee", money(d.reconveyance)],
    ])
    return [[
        "CLOSING STATEMENT",
        "",
        f"Property: {d.address}. Buyer: {d.buyer}.",
        f"Settlement Agent: {d.agent}.",
        "",
        "SECTION 1 SETTLEMENT FIGURES",
        f"Buyer's closing costs total ${dollars:,}.{cents:02d} at settlement.",
        f"Final financed amount {money(d.financed)} from {d.lender}.",
        f"Lender wire instructions reference settlement number {d.settlement_no}.",
        "",
        "SECTION 2 DISBURSEMENTS",
        "Disbursements to seller occur immediately after recording.",
        "",
        "SECTION 3 SETTLEMENT TABLE",
        settlement_table,
    ]]


def generate_corpus(deals: int = 25, seed: int = 42) -> Corpus:
    """Build the whole corpus in memory: PDFs plus the answer key."""
    rng = random.Random(seed)
    corpus = Corpus()

    def record(fname: str, pages: list[list[str | Table]]) -> str:
        data = build_pdf_bytes(pages)
        digest = hashlib.sha256(data).hexdigest()
        corpus.pdfs[fname] = data
        corpus.doc_ids[fname] = digest
        return digest

    qid = 0

    def q(**kw):
        nonlocal qid
        kw["id"] = f"q{qid:04d}"
        qid += 1
        corpus.questions.append(kw)
        return kw["id"]

    for i in range(deals):
        d = _make_deal(rng, i)
        corpus.deals.append(d)
        pa_name = f"{d.deal_id}_{d.buyer.split()[1].lower()}_purchase_agreement.pdf"
        cs_name = f"{d.deal_id}_{d.buyer.split()[1].lower()}_closing_statement.pdf"
        pa_id = record(pa_name, _purchase_agreement_pages(d))
        cs_id = record(cs_name, _closing_statement_pages(d))
        add_id = None
        add_name = None
        if d.amended:
            add_name = f"{d.deal_id}_{d.buyer.split()[1].lower()}_addendum_one.pdf"
            add_id = record(add_name, _addendum_pages(d))

        ident = f"{d.buyer.split()[1]} {d.address.split()[0]} {d.address.split()[1]}"

        # exact-figure questions (keyword precision beats embeddings)
        if d.amended != "earnest":
            q(kind="exact", deal=d.deal_id,
              query=f"earnest money deposit {ident}",
              expected_docs=[pa_id],
              expected_text=f"Earnest Money Deposit is {money(d.earnest)}")
        if d.amended != "closing":
            q(kind="exact", deal=d.deal_id,
              query=f"closing date {ident}",
              expected_docs=[pa_id],
              expected_text=f"Closing Date shall be {fmt_date(d.closing)}")
        q(kind="exact", deal=d.deal_id,
          query=f"total purchase price {ident}",
          expected_docs=[pa_id],
          expected_text=f"total purchase price of {money(d.price)}")
        q(kind="exact", deal=d.deal_id,
          query=f"settlement number wire {ident}",
          expected_docs=[cs_id],
          expected_text=f"settlement number {d.settlement_no}")
        dollars, cents = divmod(d.costs, 100)
        q(kind="exact", deal=d.deal_id,
          query=f"buyer closing costs {ident}",
          expected_docs=[cs_id],
          expected_text=f"closing costs total ${dollars:,}.{cents:02d}")

        # table facts: label-value pairs planted in a grid drawn column-
        # major. the row only exists in bytes under --tables reflow, so
        # these questions measure the extraction regime, not vibes.
        q(kind="table", deal=d.deal_id,
          query=f"title insurance amount {ident}",
          expected_docs=[cs_id],
          expected_text=f"Title Insurance | {money(d.title_ins)}",
          table_cells=["Title Insurance", money(d.title_ins)])
        q(kind="table", deal=d.deal_id,
          query=f"escrow fee amount {ident}",
          expected_docs=[cs_id],
          expected_text=f"Escrow Fee | {money(d.escrow_fee)}",
          table_cells=["Escrow Fee", money(d.escrow_fee)])

        # version trap: same tokens as the exact question, but after an
        # amendment ONLY the addendum is acceptable. nothing in the query
        # hints which document wins; that is the whole point.
        if d.amended == "earnest":
            q(kind="trap", deal=d.deal_id,
              query=f"earnest money deposit {ident}",
              expected_docs=[add_id],
              superseded_docs=[pa_id],
              expected_text=f"Deposit is amended to {money(d.new_earnest)}",
              stale_text=f"Earnest Money Deposit is {money(d.earnest)}")
        elif d.amended == "closing":
            q(kind="trap", deal=d.deal_id,
              query=f"closing date {ident}",
              expected_docs=[add_id],
              superseded_docs=[pa_id],
              expected_text=f"Closing Date is amended to {fmt_date(d.new_closing)}",
              stale_text=f"Closing Date shall be {fmt_date(d.closing)}")

        # cross-document join: the number is in the agreement, the lender
        # answer only appears in the statement; both must be retrieved
        q(kind="join", deal=d.deal_id,
          query=f"loan commitment financed amount {ident}",
          expected_docs=[pa_id, cs_id])

    # refusal questions: identities that cannot exist in this corpus
    rng_ref = random.Random(seed * 1_000_003)
    for j in range(12):
        ghost = f"{rng_ref.choice(FIRST)} {rng_ref.choice(GHOST_LAST)}"
        street = (f"{rng_ref.randint(11, 980)} {rng_ref.choice(GHOST_STREETS)} "
                  f"{rng_ref.choice(KINDS)}")
        q(kind="refusal", deal=None,
          query=f"earnest money deposit {ghost.split()[1]} "
                f"{street.split()[0]} {street.split()[1]}",
          expected_docs=[],
          absent_tokens=[ghost.split()[1], street.split()[1]])

    _verify_corpus(corpus)
    return corpus


def _verify_corpus(corpus: Corpus) -> None:
    """Generator self-check. A planted fact that failed to land in its own
    PDF bytes is a labeling bug, and labeling bugs poison every eval number.
    Raise loudly instead.

    The table branch is stronger than a byte search: it re-runs the REAL
    v0.5 extraction pipeline (layout mode + tables.reflow) over the built
    PDF bytes and demands the answer key's expected_text inside the
    reflowed canonical text. The key is checked against what ingest
    actually produces, not against what the generator wished to write."""
    import io
    from pypdf import PdfReader
    from context_wrangler.core.tables import page_text as ext_page_text

    for qq in corpus.questions:
        if qq["kind"] in ("exact", "trap"):
            assert qq["expected_docs"], qq
            fname = next(f for f, did in corpus.doc_ids.items()
                         if did == qq["expected_docs"][0])
            needle = qq["expected_text"].encode()
            if needle not in corpus.pdfs[fname]:
                raise RuntimeError(
                    f"planted fact missing from {fname}: {qq['expected_text']!r}")
        if qq["kind"] == "table":
            assert qq["expected_docs"], qq
            fname = next(f for f, did in corpus.doc_ids.items()
                         if did == qq["expected_docs"][0])
            for cell in qq["table_cells"]:
                if cell.encode() not in corpus.pdfs[fname]:
                    raise RuntimeError(
                        f"table cell missing from {fname}: {cell!r}")
            page = PdfReader(io.BytesIO(corpus.pdfs[fname])).pages[0]
            reflowed = ext_page_text(page, tables=True)
            if qq["expected_text"] not in reflowed:
                raise RuntimeError(
                    f"table row not reconstructable from {fname}: "
                    f"{qq['expected_text']!r}")
        if qq["kind"] == "refusal":
            blob = b"".join(corpus.pdfs.values())
            for tok in qq["absent_tokens"]:
                if tok.encode() in blob:
                    raise RuntimeError(
                        f"refusal token {tok!r} leaked into the corpus")


def write_corpus(out_dir: str, corpus: Corpus) -> list[str]:
    """Emit pdfs/ + answer_key.json under out_dir. Returns orphan PDFs:
    *.pdf files present after the write whose names this corpus does not
    claim. Regenerating with a different seed/rng stream changes filenames,
    so an old pdfs/ dir can survive into a new corpus as phantom documents
    that the answer key does not know about. We report those loudly; we do
    not delete files this generator did not write."""
    pdf_dir = os.path.join(out_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    for fname, data in corpus.pdfs.items():
        with open(os.path.join(pdf_dir, fname), "wb") as fh:
            fh.write(data)
    orphans = sorted(f for f in os.listdir(pdf_dir)
                     if f.endswith(".pdf") and f not in corpus.pdfs)
    key = {
        "generated_utc": datetime.datetime.now(
            datetime.UTC).isoformat(timespec="seconds"),
        "deals": len(corpus.deals),
        "doc_ids": corpus.doc_ids,
        "questions": corpus.questions,
    }
    with open(os.path.join(out_dir, "answer_key.json"), "w",
              encoding="utf-8") as fh:
        json.dump(key, fh, indent=2)
    return orphans
