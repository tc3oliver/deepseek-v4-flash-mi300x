#!/usr/bin/env python3
"""Provenance diff-freshness validator.

Detects the failure mode: *a full-file overlay changed but its provenance diff
(``patches/diffs/NN-*.patch``) was not regenerated*. Without this, the diffs
documented in ``patches/README.md`` silently drift from the overlays that
actually run.

How it works: for every overlay with a recorded pinned upstream base, fetch the
upstream file at that exact revision from GitHub, regenerate the unified diff
with the same ``diff -u --label`` invocation documented in ``patches/README.md``,
and compare it to the stored diff. The ``--- a/...`` / ``+++ b/...`` header
labels are normalized away before comparison (they are cosmetic — e.g. the
triton overlay's stored label uses the in-package path while raw GitHub serves
the monorepo path), so only the actual change hunks must match.

Overlay 07 (``rocm_aiter_mla.dspark-causal.py``) is special: the README records
it as byte-identical to upstream at its commit, so it is verified by direct
equality rather than a regenerated diff.

Tri-state result per entry and overall (``PASS`` / ``FAIL`` / ``NOT_RUN``):
  - PASS     — upstream fetched successfully and the diff is fresh.
  - FAIL     — upstream fetched successfully but the diff is stale/mismatched
               (or only some entries could be fetched — a partial result is
               not treated as a silent pass; see ``aggregate()``).
  - NOT_RUN  — no upstream base could be fetched at all (e.g. no network).
               This is a distinct state, never conflated with PASS.

This module has no dependency on network I/O beyond the default ``fetch=_fetch``
argument — every function accepts an injectable ``fetch`` callable and/or raw
base bytes, so callers (including tests) can exercise the comparison and
aggregation logic entirely offline with synthetic fixtures. Real network
validation happens exactly once in CI, via this script's CLI entry point
(``.github/workflows/validate.yml``); ``tests/test_diff_freshness.py`` tests the
same logic with local fixtures and does not re-fetch upstream, so freshness is
network-checked once per CI run, not once per invocation site.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"

# (overlay_file, diff_file, repo, revision, upstream_github_path, mode)
# mode "diff" = regenerate diff(base, overlay) and compare (label-normalized)
# mode "verbatim" = overlay must be byte-identical to upstream@revision
ENTRIES: list[tuple[str, str, str, str, str, str]] = [
    ("patches/gpt_oss_triton_kernels_moe.pack128-fused-silu-fast-routing.py",
     "patches/diffs/01-gpt_oss_triton_kernels_moe.pack128-fused-silu-fast-routing.patch",
     "vllm-project/vllm", "cb8104839c141609d99f1254459ef3a4f1bd4263",
     "vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py", "diff"),
    ("patches/mxfp4.fused-silu.py",
     "patches/diffs/02-mxfp4.fused-silu.patch",
     "vllm-project/vllm", "cb8104839c141609d99f1254459ef3a4f1bd4263",
     "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py", "diff"),
    ("patches/triton-kernels-matmul-ogs-opt-flags.dsv4-mi300x.py",
     "patches/diffs/03-triton-kernels-matmul-ogs-opt-flags.dsv4-mi300x.patch",
     "ROCm/triton", "0f380657dbf3ee86eb57558ff71df24f03b5d4e7",
     "python/triton_kernels/triton_kernels/matmul_ogs_details/opt_flags.py", "diff"),
    ("patches/fused_compress_quant_cache.fnuz-shuffle.py",
     "patches/diffs/04-fused_compress_quant_cache.fnuz-shuffle.patch",
     "vllm-project/vllm", "cb8104839c141609d99f1254459ef3a4f1bd4263",
     "vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py", "diff"),
    ("patches/aiter_pa_mqa_logits.i64.py",
     "patches/diffs/05-aiter_pa_mqa_logits.i64.patch",
     "ROCm/aiter", "4db400a90c1c1c558f3dbb40b0e6728825bbcc2b",
     "aiter/ops/triton/gluon/pa_mqa_logits.py", "diff"),
    ("patches/rocm_aiter_mla_sparse.prefill-bh64.py",
     "patches/diffs/06-rocm_aiter_mla_sparse.prefill-bh64.patch",
     "vllm-project/vllm", "cb8104839c141609d99f1254459ef3a4f1bd4263",
     "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py", "diff"),
    ("patches/rocm_aiter_mla.dspark-causal.py",
     "patches/diffs/07-rocm_aiter_mla.dspark-causal.patch",
     "vllm-project/vllm", "77469c9057bec3212a64877dbbf3b9c48c22d786",
     "vllm/v1/attention/backends/mla/rocm_aiter_mla.py", "verbatim"),
    ("patches/dspark-speculator.independent-draft-gumbel.py",
     "patches/diffs/08-dspark-speculator.independent-draft-gumbel.patch",
     "vllm-project/vllm", "cb8104839c141609d99f1254459ef3a4f1bd4263",
     "vllm/v1/worker/gpu/spec_decode/dspark/speculator.py", "diff"),
    ("patches/spec-decode-utils.independent-draft-gumbel.py",
     "patches/diffs/09-spec-decode-utils.independent-draft-gumbel.patch",
     "vllm-project/vllm", "cb8104839c141609d99f1254459ef3a4f1bd4263",
     "vllm/v1/worker/gpu/spec_decode/utils.py", "diff"),
    ("patches/kv_offload_cpu_gpu_worker.load-war.py",
     "patches/diffs/10-kv_offload_cpu_gpu_worker.load-war.patch",
     "vllm-project/vllm", "cb8104839c141609d99f1254459ef3a4f1bd4263",
     "vllm/v1/kv_offload/cpu/gpu_worker.py", "diff"),
]

RAW = "https://raw.githubusercontent.com/{repo}/{rev}/{path}"


@dataclass
class EntryResult:
    label: str
    status: str
    detail: str


@dataclass
class RunResult:
    overall: str
    entries: list[EntryResult]


def _normalize(diff_text: str) -> list[str]:
    """Strip cosmetic ``--- a/..`` / ``+++ b/..`` label lines so only the
    actual change hunks are compared."""
    return [ln for ln in diff_text.splitlines() if not ln.startswith("--- ") and not ln.startswith("+++ ")]


def _fetch(repo: str, rev: str, path: str) -> bytes | None:
    """Real network fetch. The only I/O boundary in this module — every other
    function takes bytes/callables so it can be tested without a network."""
    url = RAW.format(repo=repo, rev=rev, path=path)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except Exception as e:  # network/DNS/HTTP error
        print(f"    (fetch failed: {type(e).__name__})", file=sys.stderr)
        return None


def check_entry(
    *, label: str, overlay_path: Path, diff_path: Path, upath: str, mode: str,
    base_bytes: bytes | None,
) -> EntryResult:
    """Pure comparison logic for one overlay/diff pair, given already-fetched
    upstream bytes (or ``None`` to represent a failed/unavailable fetch).
    Does local filesystem + subprocess I/O only — no network."""
    if base_bytes is None:
        return EntryResult(label, NOT_RUN, "no network / upstream base unavailable")

    if mode == "verbatim":
        ok = hashlib.sha256(overlay_path.read_bytes()).hexdigest() == hashlib.sha256(base_bytes).hexdigest()
        return EntryResult(
            label, PASS if ok else FAIL,
            "verbatim == upstream" if ok else "overlay drifted from upstream",
        )

    # mode == "diff": regenerate the diff the same way patches/README.md documents.
    suffix = Path(upath).suffix or ".tmp"
    with tempfile.NamedTemporaryFile(suffix=suffix) as base_file:
        base_file.write(base_bytes)
        base_file.flush()
        proc = subprocess.run(
            ["diff", "-u",
             f"--label=a/{upath}",
             f"--label=b/{overlay_path.name}",
             base_file.name, str(overlay_path)],
            capture_output=True, text=True)
    # diff exits 1 when files differ (expected for overlays); stdout is the diff.
    regen = _normalize(proc.stdout)
    stored = _normalize(diff_path.read_text())
    if regen == stored:
        return EntryResult(label, PASS, "matches stored diff")
    return EntryResult(label, FAIL, f"stale: stored {len(stored)}L vs regen {len(regen)}L hunk lines")


def aggregate(entries: list[EntryResult]) -> str:
    """Roll per-entry statuses up to one overall PASS/FAIL/NOT_RUN.

    - All entries PASS                  -> PASS
    - Any entry FAIL                    -> FAIL
    - All entries NOT_RUN (or no entries) -> NOT_RUN (completely unverifiable)
    - A PASS/NOT_RUN mix (no FAIL)      -> FAIL (partial verification is not a
      silent pass — some diffs could not be confirmed fresh)
    """
    if not entries:
        return NOT_RUN
    statuses = {e.status for e in entries}
    if FAIL in statuses:
        return FAIL
    if statuses == {NOT_RUN}:
        return NOT_RUN
    if NOT_RUN in statuses:
        return FAIL
    return PASS


def run(
    entries: list[tuple[str, str, str, str, str, str]] = ENTRIES,
    fetch=_fetch,
    repo_root: Path = REPO_ROOT,
    stream=sys.stdout,
) -> RunResult:
    print(f"[validate-diff-freshness] checking {len(entries)} provenance diffs vs pinned upstream bases", file=stream)
    results: list[EntryResult] = []
    for overlay, diff_file, repo, rev, upath, mode in entries:
        label = os.path.basename(diff_file)
        overlay_path = repo_root / overlay
        diff_path = repo_root / diff_file
        base = fetch(repo, rev, upath)
        results.append(check_entry(
            label=label, overlay_path=overlay_path, diff_path=diff_path,
            upath=upath, mode=mode, base_bytes=base,
        ))

    overall = aggregate(results)
    width = max((len(r.label) for r in results), default=0)
    for r in results:
        print(f"  {r.label:<{width}}  {r.status:<8}  {r.detail}", file=stream)

    print(file=stream)
    if overall == NOT_RUN:
        print("[validate-diff-freshness] NOT_RUN — no upstream base could be fetched (no network).", file=stream)
    elif overall == FAIL:
        failing = [r.label for r in results if r.status == FAIL]
        not_run = [r.label for r in results if r.status == NOT_RUN]
        print(f"[validate-diff-freshness] FAIL — stale/mismatched: {failing or 'none'}; "
              f"unverifiable: {not_run or 'none'}. Regenerate via: "
              f"diff -u --label a/<upath> --label b/<overlay> <base> <overlay>", file=stream)
    else:
        print(f"[validate-diff-freshness] PASS — all {len(results)} fetched provenance diffs are fresh.", file=stream)

    return RunResult(overall=overall, entries=results)


def main() -> int:
    result = run()
    # NOT_RUN does not fail the CLI (a total network outage is an environment
    # condition, not a code-correctness signal) but is never printed/exit-coded
    # as PASS — see RunResult.overall / EntryResult.status for the actual
    # tri-state, which callers (e.g. tests/test_diff_freshness.py) inspect
    # directly rather than relying on the collapsed exit code.
    return {PASS: 0, FAIL: 1, NOT_RUN: 0}[result.overall]


if __name__ == "__main__":
    raise SystemExit(main())
