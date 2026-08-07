#!/usr/bin/env python3
"""Overlay integrity guard — shared by CI and container startup.

Prevents the silent-failure mode where a missed bind-mount (for example, after a
Python version bump that changes site-packages paths) lets stock, un-patched vLLM
run while ``/health`` still returns 200. With stock code, every correctness fix in
this repo (MXFP4 routing mask, FNUZ FP8, causal verify, CPU-KV fence, ...) silently
disappears and the server reports healthy.

Two layers:

  1. Content pin — every artifact listed in ``SHA256SUMS`` hashes to its pinned value.
  2. Overlay canary — every overlay source is actually present at its in-container
     runtime target with the pinned bytes.

Modes (auto-detected from whether the in-container vLLM install path exists):

  - **runtime** (inside the container): for each overlay, the runtime *target* must
    exist and hash-match its pinned *source*. A missing target is a hard failure
    (the bind-mount silently missed -> stock vLLM would run).
  - **ci/host** (repo checkout / GitHub Actions): verify ``SHA256SUMS`` for every
    pinned repo artifact, confirm every mapping source exists and is pinned, and
    cross-check the embedded mapping against the bind-mounts declared in
    ``compose.yaml``.

Exit code 0 = OK; non-zero = fail (container startup must abort via ``set -e``).

This module is importable (``from validate_overlays import ...``) so the test
suite reuses exactly the same logic as runtime/CI.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Single source of truth: overlay source (repo-relative) -> runtime target path.
# This MUST match the ``volumes:`` bind-mounts of the ``inference`` service in
# compose.yaml. ``test_overlays.py::test_mapping_matches_compose`` enforces that.
# Only file bind-mounts whose source is pinned in SHA256SUMS are listed here.
# ---------------------------------------------------------------------------
MAPPING: list[tuple[str, str]] = [
    ("vllm-entrypoint.sh", "/opt/vllm-entrypoint.sh"),
    (
        "tuning/dsv4-mi300x-a8w8-blockscale-bpreshuffle-ck.prefill-m2688.csv",
        "/opt/aiter-configs/dsv4-mi300x-a8w8-blockscale-bpreshuffle.csv",
    ),
    (
        "tuning/dsv4-a8w8-blockscale-tuned-gemm.mi300x.prefill-m2688.csv",
        "/usr/local/lib/python3.12/dist-packages/aiter/configs/model_configs/"
        "dsv4_a8w8_blockscale_tuned_gemm.csv",
    ),
    (
        "patches/gpt_oss_triton_kernels_moe.pack128-fused-silu-fast-routing.py",
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
        "fused_moe/experts/gpt_oss_triton_kernels_moe.py",
    ),
    (
        "patches/mxfp4.fused-silu.py",
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
        "fused_moe/oracle/mxfp4.py",
    ),
    (
        "patches/triton-kernels-matmul-ogs-opt-flags.dsv4-mi300x.py",
        "/usr/local/lib/python3.12/dist-packages/vllm/third_party/triton_kernels/"
        "matmul_ogs_details/opt_flags.py",
    ),
    (
        "patches/fused_compress_quant_cache.fnuz-shuffle.py",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/common/"
        "ops/fused_compress_quant_cache.py",
    ),
    (
        "patches/aiter_pa_mqa_logits.i64.py",
        "/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/gluon/"
        "pa_mqa_logits.py",
    ),
    (
        "patches/rocm_aiter_mla_sparse.prefill-bh64.py",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/"
        "rocm_aiter_mla_sparse.py",
    ),
    (
        "patches/rocm_aiter_mla.dspark-causal.py",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/"
        "rocm_aiter_mla.py",
    ),
    (
        "patches/dspark-speculator.independent-draft-gumbel.py",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/"
        "dspark/speculator.py",
    ),
    (
        "patches/spec-decode-utils.independent-draft-gumbel.py",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/"
        "utils.py",
    ),
    (
        "patches/kv_offload_cpu_gpu_worker.load-war.py",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/"
        "gpu_worker.py",
    ),
    # The integrity guard also verifies its own mount (so a drifted/missed
    # validate_overlays.py mount is caught, not silently skipped).
    ("scripts/validate_overlays.py", "/opt/validate_overlays.py"),
]

# Marker path that exists only inside the running container.
_CONTAINER_MARKER = "/usr/local/lib/python3.12/dist-packages/vllm"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sha256sums(path: str | Path) -> dict[str, str]:
    """Parse a SHA256SUMS file into ``{repo-relative path: hex digest}``."""
    pins: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # Format: "<64 hex>  <path>" (two spaces, per GNU coreutils).
            parts = line.split(None, 1)
            if len(parts) != 2 or len(parts[0]) != 64:
                raise ValueError(f"{path}:{lineno}: malformed SHA256SUMS line")
            pins[parts[1].strip()] = parts[0].strip()
    return pins


def _repo_root() -> Path:
    # In CI/host, resolve relative to this file's repo location. In-container,
    # repo-relative paths do not exist, so this is only used in ci mode.
    return Path(__file__).resolve().parent.parent


def is_in_container() -> bool:
    return os.path.isdir(_CONTAINER_MARKER)


def run_runtime_checks(pins: dict[str, str]) -> list[str]:
    """Inside the container: every overlay target must exist and match its pin."""
    errors: list[str] = []
    for src, tgt in MAPPING:
        expected = pins.get(src)
        if expected is None:
            errors.append(f"overlay source not pinned in SHA256SUMS: {src}")
            continue
        if not os.path.exists(tgt):
            # The silent-failure case: bind-mount missed -> stock vLLM would run.
            errors.append(
                f"OVERLAY TARGET MISSING: {tgt} (source {src}). "
                f"The bind-mount did not apply; stock un-patched vLLM would run."
            )
            continue
        actual = sha256_file(tgt)
        if actual != expected:
            errors.append(
                f"OVERLAY HASH MISMATCH at {tgt} (source {src}): "
                f"expected {expected}, got {actual}"
            )
    return errors


def run_ci_checks(pins: dict[str, str]) -> list[str]:
    """On the host / CI: full SHA256SUMS verification + mapping sanity."""
    root = _repo_root()
    errors: list[str] = []

    # 1. Every pinned artifact exists and matches its hash.
    for rel, expected in sorted(pins.items()):
        p = root / rel
        if not p.exists():
            errors.append(f"pinned file missing from repo: {rel}")
            continue
        actual = sha256_file(p)
        if actual != expected:
            errors.append(
                f"SHA256 MISMATCH for {rel}: expected {expected}, got {actual}"
            )

    # 2. Every mapping source exists and is pinned.
    for src, _tgt in MAPPING:
        if not (root / src).exists():
            errors.append(f"overlay source missing: {src}")
        if src not in pins:
            errors.append(f"overlay source not pinned in SHA256SUMS: {src}")

    # 3. Cross-check the mapping against compose.yaml bind-mounts (best-effort).
    errors.extend(_cross_check_compose(root))

    return errors


def _cross_check_compose(root: Path) -> list[str]:
    """Ensure MAPPING exactly matches the inference service file bind-mounts."""
    errors: list[str] = []
    compose = root / "compose.yaml"
    if not compose.exists():
        return errors  # nothing to cross-check
    try:
        import yaml  # type: ignore
    except ImportError:
        # PyYAML is a vLLM dependency in-container and available in CI; if it is
        # somehow absent, skip the cross-check rather than failing opaquely.
        return errors
    doc = yaml.safe_load(compose.read_text())
    services = (doc or {}).get("services") or {}
    inference = services.get("inference") or {}
    mounts: dict[str, str] = {}
    for entry in inference.get("volumes") or []:
        if not isinstance(entry, str) or ":" not in entry:
            continue
        src, rest = entry.split(":", 1)
        tgt = rest.split(":")[0]  # strip the mode (":ro")
        if src.startswith("./"):
            src = src[2:]
        mounts[src] = tgt
    expected_pairs = set(MAPPING)
    actual_pairs = set(
        (s, t) for s, t in mounts.items() if s.endswith((".py", ".csv", ".sh"))
    )
    missing = expected_pairs - actual_pairs
    extra = actual_pairs - expected_pairs
    for src, tgt in missing:
        errors.append(f"MAPPING NOT IN compose.yaml: {src} -> {tgt}")
    for src, tgt in extra:
        errors.append(f"compose.yaml mount not in validator MAPPING: {src} -> {tgt}")
    return errors


def main() -> int:
    root = _repo_root()
    # SHA256SUMS location: repo root on host; /opt/SHA256SUMS in-container.
    candidates = [Path("/opt/SHA256SUMS"), root / "SHA256SUMS"]
    sums_path = next((p for p in candidates if p.exists()), None)
    if sums_path is None:
        print("FATAL: SHA256SUMS not found (/opt/SHA256SUMS nor repo root).",
              file=sys.stderr)
        return 2

    pins = parse_sha256sums(sums_path)
    in_container = is_in_container()
    mode = "runtime" if in_container else "ci/host"
    print(f"[validate-overlays] mode={mode}  pins={len(pins)}  overlays={len(MAPPING)}")

    errors = run_runtime_checks(pins) if in_container else run_ci_checks(pins)

    if errors:
        print(f"\n[validate-overlays] FAILED ({len(errors)} problem(s)):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("[validate-overlays] OK — all overlays present and pinned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
