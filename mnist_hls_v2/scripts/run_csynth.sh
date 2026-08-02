#!/usr/bin/env bash
# C synthesis for a named variant, into results/<name>/.
#
# Usage: run_csynth.sh <name> [clock_ns]
#
# The config is generated rather than checked in so the clock target can be
# swept without hand-editing, and so every path stays relative to the repo --
# v1's hls_config.cfg hard-coded /media/ntk/sda4/yuechi/... which breaks the
# moment the tree moves.
set -euo pipefail

name="${1:?usage: run_csynth.sh <name> [clock_ns]}"
clock="${2:-4}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/results/$name"
mkdir -p "$out"

cat > "$out/hls_config.cfg" <<EOF
part=xck24-ubva530-2LV-c

[hls]
flow_target=vivado
package.output.format=ip_catalog
package.output.syn=false
syn.top=MultilayerPerceptron
syn.file=$root/src/mlp_top.cpp
tb.file=$root/test/tb_mlp.cpp
syn.cflags=-I$root/src -std=c++14
tb.cflags=-I$root/src -std=c++14 -DMLP_TEST_DATA_DIR='"$root/test/data"'
# Bind every `*` to a DSP. Without this HLS puts all 394 multipliers on LUT
# fabric (51k LUTs, 0 DSPs); the lanes that must stay off DSPs are expressed
# in the source as shift-adds instead. v1 steers per-variable with BIND_OP
# instead, which works but which Vitis 2025.1's pragma lint refuses to let
# this tree compile -- see the note in src/mlp.hpp.
syn.op=mul -impl dsp
clock=${clock}ns
EOF

# settings64.sh reads unset variables, so relax nounset just for it.
set +u
source "${XILINX_ROOT:-/media/ntk/sda4/Xilinx}/2025.1/Vitis/settings64.sh"
set -u

cd "$out"
v++ -c --mode hls --config "$out/hls_config.cfg" --work_dir "$out/hls" 2>&1 | tee "$out/csynth.log"

report="$out/hls/hls/syn/report/MultilayerPerceptron_csynth.rpt"
if [[ -f "$report" ]]; then
    cp "$report" "$out/MultilayerPerceptron_csynth.rpt"
    echo
    echo "=== $name @ ${clock}ns ==="
    sed -n '/== Performance Estimates/,/Detail/p' "$report" | grep -E '^\s*\|\s*[0-9]' | head -3
    sed -n '/Utilization Estimates/,/Detail/p' "$report" | grep -E 'Total|Available|Utilization'
else
    echo "ERROR: no synthesis report at $report" >&2
    exit 1
fi
