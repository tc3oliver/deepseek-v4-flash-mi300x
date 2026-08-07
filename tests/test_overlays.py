"""Overlay integrity tests — reuse the exact logic the container runs at startup.

These are the CI counterpart to ``scripts/validate_overlays.py`` (host mode):
every pinned artifact matches its SHA256, every overlay source is pinned, and the
embedded source->target MAPPING matches the bind-mounts declared in compose.yaml.
"""
from pathlib import Path

import validate_overlays as vo

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pins():
    return vo.parse_sha256sums(REPO_ROOT / "SHA256SUMS")


def test_sha256sums_parses():
    pins = _pins()
    # 3 config + 10 overlays + 10 diffs + 2 CSVs + integrity script (pinned) ...
    assert len(pins) >= 25


def test_every_overlay_source_is_pinned():
    pins = _pins()
    for src, _tgt in vo.MAPPING:
        assert src in pins, f"overlay source not pinned in SHA256SUMS: {src}"


def test_mapping_matches_compose_mounts():
    errors = vo._cross_check_compose(REPO_ROOT)
    assert errors == [], "MAPPING / compose.yaml drift:\n  - " + "\n  - ".join(errors)


def test_ci_integrity_checks_pass():
    """The full host-mode gate: SHA256 of every pinned file + mapping sanity."""
    errors = vo.run_ci_checks(_pins())
    assert errors == [], "CI integrity checks failed:\n  - " + "\n  - ".join(errors)
