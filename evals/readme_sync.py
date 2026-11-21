"""README tables that regenerate, so the README never goes stale.

The README wraps each machine-owned block in marker comments:

    <!-- BEGIN AUTO:FRONTIER -->
    ...generated table lives here...
    <!-- END AUTO:FRONTIER -->

`--write` replaces the content between markers with freshly generated
text; `--check` (what CI uses) exits 1 when the committed README does not
match what regeneration would produce, so a stale published number fails
the build instead of shipping. Rules enforced in code, not by habit:

- missing or unpaired markers: loud error, no silent append
- duplicated markers: loud error (which block wins would be a coin flip)
- content changed to the identical bytes: no write, exit 0 (vacuous
  input earns nothing, same selftest rule as every other check here)
"""

from __future__ import annotations

import argparse
import os
import re
import sys

MARK = "AUTO:{}"


def _marker_re(name: str, kind: str) -> re.Pattern:
    return re.compile(r"[ \t]*<!--\s+" + kind + r"\s+AUTO:" +
                      re.escape(name) + r"\s*-->")


def find_blocks(readme: str) -> dict[str, tuple[int, int]]:
    """name -> (begin_span_end, end_span_start) of the CONTENT region."""
    out: dict[str, tuple[int, int]] = {}
    for name in _declared_names(readme):
        b, e = _marker_re(name, "BEGIN"), _marker_re(name, "END")
        if len(b.findall(readme)) != 1 or len(e.findall(readme)) != 1:
            raise ValueError(f"block {name!r}: markers must appear exactly "
                             "once each; refusing to guess")
        ms = b.search(readme).span()[1]
        me = e.search(readme).span()[0]
        if me < ms:
            raise ValueError(f"block {name!r}: END marker before BEGIN")
        out[name] = (ms, me)
    return out


def _declared_names(readme: str) -> list[str]:
    names = []
    for m in re.finditer(r"<!--\s+(?:BEGIN|END)\s+AUTO:([A-Z0-9_]+)\s*-->",
                         readme):
        if m.group(1) not in names:
            names.append(m.group(1))
    # a name that appears only as BEGIN or only as END trips the
    # exactly-once rule inside find_blocks; no separate pass needed.
    return names


def sync_text(readme: str, blocks: dict[str, str]) -> str:
    """Return the README with each named region replaced. Regions are
    rewritten highest-offset first so earlier spans stay valid."""
    regions = find_blocks(readme)
    for name in blocks:
        if name not in regions:
            raise ValueError(f"README has no AUTO:{name} markers")
    out = readme
    for name, text in sorted(blocks.items(),
                             key=lambda kv: -regions[kv[0]][0]):
        s, e = regions[name]
        body = "\n" + text.strip("\n") + "\n"
        out = out[:s] + body + out[e:]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="readme_sync")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if regeneration would change the file")
    ap.add_argument("--write", action="store_true",
                    help="rewrite the marked regions in place")
    ap.add_argument("blocks", nargs="*", metavar="NAME=file.md",
                    help="block name -> generated markdown file")
    args = ap.parse_args(argv)
    if args.check == args.write:
        print("pick exactly one of --check or --write", file=sys.stderr)
        return 2
    payload: dict[str, str] = {}
    for spec in args.blocks:
        name, _, path = spec.partition("=")
        if not path or not os.path.exists(path):
            print(f"bad block spec {spec!r} (want NAME=file.md that exists)",
                  file=sys.stderr)
            return 2
        payload[name] = open(path, encoding="utf-8").read()
    try:
        current = open(args.readme, encoding="utf-8").read()
        new = sync_text(current, payload)
    except ValueError as exc:
        print(f"readme_sync: {exc}", file=sys.stderr)
        return 1
    if new == current:
        print(f"README up to date ({len(payload)} block(s) verified)")
        return 0
    if args.check:
        print("README tables are stale: regeneration would change them",
              file=sys.stderr)
        return 1
    with open(args.readme, "w", encoding="utf-8") as fh:
        fh.write(new)
    print(f"synced {len(payload)} block(s) into {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
