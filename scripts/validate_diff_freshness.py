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

Requires network (GitHub raw). CI has it; if no base can be fetched the run is
reported as NOT RUN rather than falsely PASS/FAIL. Exit 0 only when every
fetchable overlay is fresh; exit 1 on any stale diff.

Reused by tests (``tests/test_overlays.py``) so CI and the test suite share one
implementation.
"""
from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

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


def _normalize(diff_text: str) -> list[str]:
    """Strip cosmetic ``--- a/..`` / ``+++ b/..`` label lines so only the
    actual change hunks are compared."""
    return [ln for ln in diff_text.splitlines() if not ln.startswith("--- ") and not ln.startswith("+++ ")]


def _fetch(repo: str, rev: str, path: str) -> bytes | None:
    url = RAW.format(repo=repo, rev=rev, path=path)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except Exception as e:  # network/DNS/HTTP error
        print(f"    (fetch failed: {type(e).__name__})", file=sys.stderr)
        return None


def run() -> int:
    print("[validate-diff-freshness] checking 10 provenance diffs vs pinned upstream bases")
    results: list[tuple[str, str]] = []  # (label, status)
    fetched = 0
    skipped = 0
    failures = 0

    for overlay, diff_file, repo, rev, upath, mode in ENTRIES:
        label = f"{os.path.basename(diff_file)}"
        overlay_path = REPO_ROOT / overlay
        diff_path = REPO_ROOT / diff_file
        base = _fetch(repo, rev, upath)
        if base is None:
            results.append((label, "SKIP (no network)"))
            skipped += 1
            continue
        fetched += 1
        base_file = REPO_ROOT / ".tmp_freshness_base.py"
        base_file.write_bytes(base)
        try:
            if mode == "verbatim":
                ok = hashlib.sha256(overlay_path.read_bytes()).hexdigest() == hashlib.sha256(base).hexdigest()
                status = "PASS (verbatim == upstream)" if ok else "FAIL (overlay drifted from upstream)"
                if not ok:
                    failures += 1
            else:
                # Regenerate the diff the same way patches/README.md documents.
                proc = subprocess.run(
                    ["diff", "-u",
                     f"--label=a/{upath}",
                     f"--label=b/{os.path.basename(overlay)}",
                     str(base_file), str(overlay_path)],
                    capture_output=True, text=True)
                # diff exits 1 when files differ (expected for overlays); stdout is the diff.
                regen = _normalize(proc.stdout)
                stored = _normalize(diff_path.read_text())
                if regen == stored:
                    status = "PASS"
                else:
                    status = f"FAIL (stale: stored {len(stored)}L vs regen {len(regen)}L hunk lines)"
                    failures += 1
            results.append((label, status))
        finally:
            base_file.unlink(missing_ok=True)

    width = max(len(l) for l, _ in results)
    for label, status in results:
        print(f"  {label:<{width}}  {status}")

    print()
    if fetched == 0:
        print("[validate-diff-freshness] NOT RUN — no upstream base could be fetched (no network).")
        return 0  # NOT RUN, not a false PASS
    if skipped > 0:
        print(f"[validate-diff-freshness] WARNING: {skipped} entry/entries skipped (partial network).")
        failures += skipped
    if failures:
        print(f"[validate-diff-freshness] FAILED — {failures} stale/missing diff(s). "
              f"Regenerate via: diff -u --label a/<upath> --label b/<overlay> <base> <overlay>")
        return 1
    print(f"[validate-diff-freshness] OK — all {fetched} fetched provenance diffs are fresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
