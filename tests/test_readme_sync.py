"""readme_sync: the marker machinery behind CI-regenerated tables."""

import pytest

from evals.readme_sync import find_blocks, sync_text

README = (
    "# Project\n\n"
    "Intro prose.\n\n"
    "<!-- BEGIN AUTO:FRONTIER -->\nold table\n<!-- END AUTO:FRONTIER -->\n\n"
    "Middle prose.\n\n"
    "<!-- BEGIN AUTO:ANSWERS -->\nstale rows\n<!-- END AUTO:ANSWERS -->\n\n"
    "Tail.\n")


def test_replaces_named_regions_only():
    out = sync_text(README, {"FRONTIER": "| a | b |\n|---|---|\n| 1 | 2 |"})
    assert "| 1 | 2 |" in out
    assert "old table" not in out
    assert "stale rows" in out              # other block untouched
    assert "Intro prose." in out and "Tail." in out
    assert out.count("<!-- BEGIN AUTO:FRONTIER -->") == 1


def test_idempotent_write_is_a_noop():
    once = sync_text(README, {"ANSWERS": "same\ncontent"})
    twice = sync_text(once, {"ANSWERS": "same\ncontent"})
    assert once == twice                   # vacuous input earns nothing


def test_missing_markers_raise():
    with pytest.raises(ValueError, match="no AUTO"):
        sync_text("# nothing here\n", {"FRONTIER": "x"})


def test_unpaired_markers_raise():
    broken = README.replace("<!-- END AUTO:FRONTIER -->", "")
    with pytest.raises(ValueError, match="exactly once"):
        find_blocks(broken)


def test_duplicated_markers_raise():
    broken = README + "<!-- BEGIN AUTO:FRONTIER -->\nz\n" \
                      "<!-- END AUTO:FRONTIER -->\n"
    with pytest.raises(ValueError, match="exactly once"):
        find_blocks(broken)


def test_reversed_markers_raise():
    broken = ("<!-- END AUTO:X -->\n<!-- BEGIN AUTO:X -->\n")
    with pytest.raises(ValueError, match="END marker before BEGIN"):
        find_blocks(broken)


def test_multiple_blocks_replaced_in_one_pass():
    out = sync_text(README, {"FRONTIER": "F", "ANSWERS": "A"})
    assert "F" in out and "A" in out
    assert "old table" not in out and "stale rows" not in out


def test_cli_check_detects_stale(tmp_path, capsys):
    from evals.readme_sync import main
    rd = tmp_path / "README.md"
    rd.write_text(README)
    block = tmp_path / "gen.md"
    block.write_text("fresh table")
    assert main(["--check", "--readme", str(rd),
                 f"FRONTIER={block}"]) == 1      # stale: would change
    assert "stale" in capsys.readouterr().err
    assert main(["--write", "--readme", str(rd),
                 f"FRONTIER={block}"]) == 0
    assert "fresh table" in rd.read_text()
    assert main(["--check", "--readme", str(rd),
                 f"FRONTIER={block}"]) == 0      # now identical: no lie
    capsys.readouterr()
    assert main(["--check", "--readme", str(rd),
                 "FRONTIER=missing_file.md"]) == 2
