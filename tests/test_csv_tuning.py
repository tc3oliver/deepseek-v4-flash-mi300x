"""Validation for the AITER A8W8 blockscale tuning tables.

Checks schema, numeric parseability, duplicate-shape keys, and the production
invariant that the mounted MI300X table is gfx942-only (gfx950 rows live in a
separate reference file). These run in CI on every push/PR.
"""
import csv
from pathlib import Path

import pytest

TUNING = Path(__file__).resolve().parent.parent / "tuning"

# Files mounted into the container by compose.yaml (production tables).
PRODUCTION_CSVS = [
    TUNING / "dsv4-a8w8-blockscale-tuned-gemm.mi300x.prefill-m2688.csv",
    TUNING / "dsv4-mi300x-a8w8-blockscale-bpreshuffle-ck.prefill-m2688.csv",
]
# Reference-only (NOT mounted): preserves gfx950 rows carried over from stock AITER.
REFERENCE_CSVS = [
    TUNING / "dsv4-a8w8-blockscale-tuned-gemm.gfx950.reference.csv",
]

REQUIRED_COLS = [
    "gfx", "cu_num", "M", "N", "K", "libtype", "kernelId",
    "splitK", "us", "kernelName", "tflops", "bw", "errRatio",
]
NUMERIC_COLS = ["cu_num", "M", "N", "K", "kernelId", "splitK", "us", "tflops", "bw", "errRatio"]
KEY_COLS = ["gfx", "M", "N", "K"]


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("p", PRODUCTION_CSVS + REFERENCE_CSVS)
def test_schema_and_numeric(p: Path):
    rows = _rows(p)
    assert rows, f"{p.name}: no data rows"
    for r in rows:
        assert list(r.keys()) == REQUIRED_COLS, f"{p.name}: header mismatch -> {list(r.keys())}"
        for c in NUMERIC_COLS:
            assert r[c].strip() != "", f"{p.name}: empty numeric field {c}"
            float(r[c])  # raises ValueError on unparseable


@pytest.mark.parametrize("p", PRODUCTION_CSVS + REFERENCE_CSVS)
def test_no_duplicate_shape_key(p: Path):
    seen: set[tuple[str, str, str, str]] = set()
    for r in _rows(p):
        key = tuple(r[c] for c in KEY_COLS)
        assert key not in seen, f"{p.name}: duplicate shape key {key}"
        seen.add(key)


def test_production_tuned_gemm_is_gfx942_only():
    archs = {r["gfx"] for r in _rows(PRODUCTION_CSVS[0])}
    assert archs == {"gfx942"}, (
        f"production tuned-gemm CSV must be gfx942-only (it is mounted into the "
        f"container); got {archs}"
    )


def test_gfx942_uses_304_cus():
    for p in PRODUCTION_CSVS:
        for r in _rows(p):
            if r["gfx"] == "gfx942":
                assert r["cu_num"] == "304", f"{p.name}: gfx942 cu_num must be 304"


def test_gfx950_reference_preserves_all_rows():
    """The gfx950 rows removed from production are fully retained for reference."""
    ref = _rows(REFERENCE_CSVS[0])
    assert ref, "gfx950 reference file is empty"
    assert all(r["gfx"] == "gfx950" for r in ref), "reference file contains non-gfx950 rows"
    assert len(ref) == 204, f"expected 204 gfx950 reference rows, got {len(ref)}"
