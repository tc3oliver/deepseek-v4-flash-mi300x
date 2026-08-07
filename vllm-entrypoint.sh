#!/bin/sh
set -eu

# Fail-fast overlay integrity guard (shared with CI via scripts/validate_overlays.py).
# Aborts startup if any pinned artifact drifted or any overlay did not bind-mount at
# its runtime target — prevents stock, un-patched vLLM from running while /health is
# still 200.
python3 /opt/validate_overlays.py

# A dead EngineCore cannot unlink its CPU-KV mmap. This host serves one vLLM instance.
find /dev/shm -maxdepth 1 -type f -name 'vllm_offload_*.mmap' -delete

exec vllm serve "$@"
