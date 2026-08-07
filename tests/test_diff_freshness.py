"""Offline tests for the provenance diff-freshness validator's logic.

These test ``check_entry`` / ``aggregate`` / ``run`` (with an injected fake
``fetch``) entirely with local fixtures — no network I/O. The real
network-based freshness check runs exactly once, in CI, via the standalone
``python3 scripts/validate_diff_freshness.py`` step
(``.github/workflows/validate.yml``); this file must not duplicate that fetch,
so it never imports/calls the real ``_fetch`` and never hits GitHub.

This also demonstrates the required tri-state contract: NOT_RUN (upstream
unreachable) is a distinct outcome from PASS, and must map to a real
``pytest.skip()`` — never a silent/false PASSED — while FAIL (stale diff) must
map to ``pytest.fail()``.
"""
from pathlib import Path

import pytest

import validate_diff_freshness as vdf


# ---------------------------------------------------------------------------
# check_entry — pure comparison logic, local files only
# ---------------------------------------------------------------------------

def test_check_entry_verbatim_pass(tmp_path: Path):
    base = b"identical content\n"
    overlay = tmp_path / "overlay.py"
    overlay.write_bytes(base)
    result = vdf.check_entry(
        label="x", overlay_path=overlay, diff_path=tmp_path / "unused.patch",
        upath="a/b.py", mode="verbatim", base_bytes=base,
    )
    assert result.status == vdf.PASS


def test_check_entry_verbatim_fail(tmp_path: Path):
    overlay = tmp_path / "overlay.py"
    overlay.write_bytes(b"drifted content\n")
    result = vdf.check_entry(
        label="x", overlay_path=overlay, diff_path=tmp_path / "unused.patch",
        upath="a/b.py", mode="verbatim", base_bytes=b"original content\n",
    )
    assert result.status == vdf.FAIL


def test_check_entry_diff_pass_when_stored_diff_matches_regenerated(tmp_path: Path):
    base = b"line1\nline2\nline3\n"
    overlay = tmp_path / "overlay.py"
    overlay.write_bytes(b"line1\nCHANGED\nline3\n")

    # Regenerate the diff exactly as check_entry(mode="diff") would internally,
    # write it out as the "stored" diff, then confirm comparing against itself passes.
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".py") as bf:
        bf.write(base)
        bf.flush()
        proc = subprocess.run(
            ["diff", "-u", "--label=a/pkg/base.py", f"--label=b/{overlay.name}",
             bf.name, str(overlay)],
            capture_output=True, text=True)
    stored = tmp_path / "stored.patch"
    stored.write_text(proc.stdout)

    result = vdf.check_entry(
        label="x", overlay_path=overlay, diff_path=stored,
        upath="pkg/base.py", mode="diff", base_bytes=base,
    )
    assert result.status == vdf.PASS, result.detail


def test_check_entry_diff_fail_when_overlay_changed_but_diff_stale(tmp_path: Path):
    base = b"line1\nline2\nline3\n"
    overlay = tmp_path / "overlay.py"
    overlay.write_bytes(b"line1\nCHANGED\nline3\n")

    # Stored diff reflects a DIFFERENT (older) change than the overlay now has —
    # this is exactly the bug class this validator exists to catch.
    stale_diff = (
        "--- a/pkg/base.py\n"
        "+++ b/overlay.py\n"
        "@@ -1,3 +1,3 @@\n"
        " line1\n"
        "-line2\n"
        "+SOME OTHER OLD CHANGE\n"
        " line3\n"
    )
    stored = tmp_path / "stored.patch"
    stored.write_text(stale_diff)

    result = vdf.check_entry(
        label="x", overlay_path=overlay, diff_path=stored,
        upath="pkg/base.py", mode="diff", base_bytes=base,
    )
    assert result.status == vdf.FAIL
    assert "stale" in result.detail


def test_check_entry_not_run_when_base_unavailable(tmp_path: Path):
    result = vdf.check_entry(
        label="x", overlay_path=tmp_path / "overlay.py", diff_path=tmp_path / "d.patch",
        upath="pkg/base.py", mode="diff", base_bytes=None,
    )
    assert result.status == vdf.NOT_RUN


# ---------------------------------------------------------------------------
# aggregate — roll-up semantics
# ---------------------------------------------------------------------------

def _e(status: str) -> vdf.EntryResult:
    return vdf.EntryResult(label="x", status=status, detail="")


def test_aggregate_all_pass_is_pass():
    assert vdf.aggregate([_e(vdf.PASS), _e(vdf.PASS)]) == vdf.PASS


def test_aggregate_any_fail_is_fail():
    assert vdf.aggregate([_e(vdf.PASS), _e(vdf.FAIL)]) == vdf.FAIL


def test_aggregate_all_not_run_is_not_run():
    assert vdf.aggregate([_e(vdf.NOT_RUN), _e(vdf.NOT_RUN)]) == vdf.NOT_RUN


def test_aggregate_empty_is_not_run():
    assert vdf.aggregate([]) == vdf.NOT_RUN


def test_aggregate_partial_not_run_mixed_with_pass_is_fail():
    """A partial network outage (some fetched, some not) must not silently
    read as an overall PASS — unverified entries make the run unreliable."""
    assert vdf.aggregate([_e(vdf.PASS), _e(vdf.NOT_RUN)]) == vdf.FAIL


# ---------------------------------------------------------------------------
# run() wiring — injected fake fetch, still zero network I/O
# ---------------------------------------------------------------------------

def test_run_with_all_fetches_unavailable_is_not_run(tmp_path: Path, capsys):
    fake_entries = [
        ("overlay_a.py", "diff_a.patch", "org/repo", "deadbeef", "pkg/a.py", "diff"),
        ("overlay_b.py", "diff_b.patch", "org/repo", "deadbeef", "pkg/b.py", "verbatim"),
    ]
    (tmp_path / "overlay_a.py").write_bytes(b"x")
    (tmp_path / "overlay_b.py").write_bytes(b"x")

    def fetch_none(repo, rev, path):
        return None  # simulates total network unavailability — no real I/O

    result = vdf.run(entries=fake_entries, fetch=fetch_none, repo_root=tmp_path)
    assert result.overall == vdf.NOT_RUN
    assert all(e.status == vdf.NOT_RUN for e in result.entries)


def test_run_detects_stale_diff_with_injected_fetch(tmp_path: Path):
    base = b"a\nb\nc\n"
    overlay = tmp_path / "overlay_a.py"
    overlay.write_bytes(b"a\nCHANGED\nc\n")
    diff_file = tmp_path / "diff_a.patch"
    diff_file.write_text("--- a/pkg/a.py\n+++ b/overlay_a.py\n@@ stale @@\n")

    entries = [("overlay_a.py", "diff_a.patch", "org/repo", "deadbeef", "pkg/a.py", "diff")]

    def fetch_fixed(repo, rev, path):
        return base  # local fixture bytes, not a network call

    result = vdf.run(entries=entries, fetch=fetch_fixed, repo_root=tmp_path)
    assert result.overall == vdf.FAIL
    assert result.entries[0].status == vdf.FAIL


# ---------------------------------------------------------------------------
# pytest integration contract: NOT_RUN -> real skip, FAIL -> real fail,
# PASS -> normal pass. Uses an injected fetch (offline) standing in for "no
# network reached upstream" so this is verifiable without a live connection,
# while still proving the skip/fail plumbing is correct (not silently PASSED).
# ---------------------------------------------------------------------------

def _pytest_outcome_for(result: vdf.RunResult) -> None:
    if result.overall == vdf.NOT_RUN:
        pytest.skip("provenance diff freshness: upstream unreachable (NOT_RUN)")
    if result.overall == vdf.FAIL:
        failing = [e.label for e in result.entries if e.status == vdf.FAIL]
        pytest.fail(f"stale/mismatched provenance diff(s): {failing}")
    # PASS: fall through as a normal pass.


def test_not_run_result_triggers_real_pytest_skip(tmp_path: Path):
    (tmp_path / "overlay_a.py").write_bytes(b"x")
    entries = [("overlay_a.py", "diff_a.patch", "org/repo", "deadbeef", "pkg/a.py", "diff")]
    result = vdf.run(entries=entries, fetch=lambda *a: None, repo_root=tmp_path)
    with pytest.raises(pytest.skip.Exception):
        _pytest_outcome_for(result)


def test_fail_result_triggers_real_pytest_fail(tmp_path: Path):
    base = b"a\nb\n"
    overlay = tmp_path / "overlay_a.py"
    overlay.write_bytes(b"a\nCHANGED\n")
    (tmp_path / "diff_a.patch").write_text("--- a/pkg/a.py\n+++ b/overlay_a.py\n@@ stale @@\n")
    entries = [("overlay_a.py", "diff_a.patch", "org/repo", "deadbeef", "pkg/a.py", "diff")]
    result = vdf.run(entries=entries, fetch=lambda *a: base, repo_root=tmp_path)
    with pytest.raises(pytest.fail.Exception):
        _pytest_outcome_for(result)


def test_pass_result_does_not_skip_or_fail(tmp_path: Path):
    base = b"a\nb\n"
    overlay = tmp_path / "overlay_a.py"
    overlay.write_bytes(b"a\nCHANGED\n")
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".py") as bf:
        bf.write(base); bf.flush()
        proc = subprocess.run(
            ["diff", "-u", "--label=a/pkg/a.py", "--label=b/overlay_a.py", bf.name, str(overlay)],
            capture_output=True, text=True)
    (tmp_path / "diff_a.patch").write_text(proc.stdout)
    entries = [("overlay_a.py", "diff_a.patch", "org/repo", "deadbeef", "pkg/a.py", "diff")]
    result = vdf.run(entries=entries, fetch=lambda *a: base, repo_root=tmp_path)
    assert result.overall == vdf.PASS
    _pytest_outcome_for(result)  # must not raise


# ---------------------------------------------------------------------------
# Sanity: the real ENTRIES table used by CI still points at files that exist
# in this repo (catches a typo'd path without ever touching the network).
# ---------------------------------------------------------------------------

def test_real_entries_overlay_and_diff_paths_exist():
    root = Path(__file__).resolve().parent.parent
    for overlay, diff_file, _repo, _rev, _upath, _mode in vdf.ENTRIES:
        assert (root / overlay).exists(), f"missing overlay: {overlay}"
        assert (root / diff_file).exists(), f"missing diff: {diff_file}"
