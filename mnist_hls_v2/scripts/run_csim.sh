#!/usr/bin/env bash
# Full functional check with plain g++ -- no Xilinx tools needed.
#
# The kernel has no ap_int dependency, so the entire 10000-image test set runs
# in under a second. Use this for every edit; save csynth for when the
# behaviour is already known to be correct.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${TMPDIR:-/tmp}/mlp_tb_$$"

g++ -O2 -Wall -Wno-unknown-pragmas -Wno-unused-label -std=c++14 \
    -I"$root/src" \
    "$root/test/tb_mlp.cpp" "$root/src/mlp_top.cpp" -o "$out"

trap 'rm -f "$out"' EXIT
MLP_TEST_DATA_DIR="$root/test/data" "$out" "$@"
