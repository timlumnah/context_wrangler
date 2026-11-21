"""Tiny PDF fixture corpus for v0.1 tests.

The PDF writer itself now lives in context_wrangler.corpus (it graduated
when the real generator needed it); v0.1 keeps three hand-written docs so
the early tests stay readable and independent of the seeded generator.
"""

from __future__ import annotations

from context_wrangler.corpus import make_pdf  # re-export: tests use this


PURCHASE_AGREEMENT = [
    [
        "PURCHASE AND SALE AGREEMENT",
        "",
        "ARTICLE 1 PURCHASE PRICE",
        "Buyer agrees to pay the total purchase price of $485,000 for the",
        "property located at 142 Beacon Street.",
        "",
        "ARTICLE 2 EARNEST MONEY",
        "The Earnest Money Deposit is $12,500, payable to the escrow agent",
        "within 72 hours of acceptance of this offer.",
        "",
        "ARTICLE 3 CLOSING",
        "The Closing Date shall be March 15, 2027, unless extended in",
        "writing by both parties.",
    ],
    [
        "ARTICLE 4 CONTINGENCIES",
        "This offer is contingent on a home inspection completed within 10",
        "business days of acceptance.",
        "",
        "Financing contingency: Buyer must obtain a loan commitment for 80",
        "percent of the purchase price by February 1, 2027.",
    ],
]

ADDENDUM_ONE = [
    [
        "ADDENDUM ONE TO PURCHASE AND SALE AGREEMENT",
        "",
        "Date: April 2, 2027",
        "",
        "SECTION 1 AMENDMENT OF EARNEST MONEY",
        "The Earnest Money Deposit is amended to $18,000. The original",
        "amount in Article 2 is superseded.",
        "",
        "SECTION 2 NO OTHER CHANGES",
        "All other terms and conditions of the Agreement remain unchanged.",
    ],
]

CLOSING_STATEMENT = [
    [
        "CLOSING STATEMENT",
        "",
        "SECTION 1 PARTIES",
        "Settlement Agent: Beacon Title Company.",
        "",
        "SECTION 2 SETTLEMENT FIGURES",
        "Buyer's closing costs total $9,340.75 at settlement.",
        "Lender wire instructions reference settlement number 2027-0315-44.",
    ],
]

DOCS = {
    "purchase_agreement.pdf": PURCHASE_AGREEMENT,
    "addendum_one.pdf": ADDENDUM_ONE,
    "closing_statement.pdf": CLOSING_STATEMENT,
}


def write_corpus(pdf_dir: str) -> dict[str, str]:
    """Write the fixture corpus into pdf_dir. Returns name -> path."""
    import os
    os.makedirs(pdf_dir, exist_ok=True)
    paths = {}
    for name, pages in DOCS.items():
        p = os.path.join(pdf_dir, name)
        make_pdf(p, pages)
        paths[name] = p
    return paths
