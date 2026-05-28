#!/usr/bin/env bash
# Compatibility wrapper for the current BF16 gate.
#
# The BF16 smoke already passed with fast_ram. Do not go straight to 64 batches:
# run the 4-batch prune + HF eval gate first, then launch 64 only after review.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

exec bash scripts/run_bf16_gate_4_then_64.sh "$@"
