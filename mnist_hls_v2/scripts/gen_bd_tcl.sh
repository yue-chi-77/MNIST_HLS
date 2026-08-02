#!/usr/bin/env bash
# Emit a copy of the v1 block design tcl with the PL0 clock retargeted.
#
# The shipped design ran PL0 at 100 MHz, which is why Vivado stopped optimising
# at 9 ns: it had 10 ns to hit. Everything about this sweep is that one number.
#
# Usage: gen_bd_tcl.sh <freq_mhz> > design_<freq>.tcl
set -euo pipefail

freq="${1:?usage: gen_bd_tcl.sh <freq_mhz>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/../reports/vivado_2025.1_KD240/design_1.tcl"

# The PS reports the achievable IOPLL division, not the request; 1.5 GHz VCO
# over an integer divider is what the tool would have written itself.
act=$(awk -v f="$freq" 'BEGIN { printf "%.6f", 1499.985004 / int(1499.985004 / f + 0.5) }')

sed -e "s/CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ {99.999001}/CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ {$act}/" \
    -e "s/CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100}/CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {$freq}/" \
    "$src"
