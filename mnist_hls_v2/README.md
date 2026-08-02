# mnist_hls_v2 — KD240 MNIST MLP

A rewrite of this repository's KV260 MNIST HLS project targeting the KD240
(`xck24-ubva530-2LV-c`). It lives alongside the original, which is untouched:
weights and test vectors are derived from `../src/mnist.h` and `../pynq/`.

Network: 784 → 128 → 256 → 10, int8 weights, power-of-two rescale, argmax on
chip. One activation is consumed per cycle per layer, so latency is
784 + 128 + 256 cycles plus overhead.

## The headline result

v1's shipped bitstream was implemented with the PL clock constrained to
**100 MHz**. Vivado therefore stopped optimising the moment it reached 9.0 ns,
and the board could only be overclocked to ~166 MHz before results broke.

Re-implementing the **identical** design at a 250 MHz constraint — one number
in the block design, no RTL or HLS change — closes timing with margin:

| | v1 as shipped | re-constrained |
|---|---|---|
| PL0 constraint | 100 MHz | **250 MHz** |
| WNS | +0.996 ns | **+0.010 ns (MET)** |
| critical path | 9.004 ns | **3.044 ns** |
| — of which routing | 6.647 ns (83%) | 1.761 ns (58%) |
| achievable clock | ~111 MHz | **250 MHz** |
| LUT (post-implementation) | — | 26,052 (36.9%) |
| DSP | — | 256 (71.1%) |
| BRAM tile | — | 31.5 (14.6%) |

The critical path is the same net in both builds — a fanout-256 broadcast from
the `data1` BRAM to layer 2's multipliers, with **zero logic levels**. It was
never a logic-depth problem. Given a constraint worth working for, the placer
put those DSPs next to the BRAM and the same net got 3.8x faster.

### Measured on hardware

Run on the KD240 (`kria`, 120.126.83.228) over the full 10000-image test set,
every prediction compared against the host reference:

| PL clock | wrong predictions | batch | throughput |
|---|---|---|---|
| 250 MHz (constraint) | **0** | 59.28 ms | 168,698 fps |
| 300 MHz | **0** | 49.92 ms | 200,333 fps |
| 375 MHz | **0** | 40.30 ms | **248,122 fps** |
| 500 MHz | 8,977 | — | past the limit |

375 MHz is the real ceiling; 500 MHz breaks badly rather than degrading.

Two numbers matter, and they are not the same thing:

* **168,698 fps at 250 MHz** is what Vivado signs off. Timing is met at the
  slow corner, so it holds across temperature, voltage and silicon lot.
* **248,122 fps at 375 MHz** is overclocking. It is bit-exact on this chip at
  room temperature today, and it is what the article's KV260 build did too
  (231,630 fps), but nothing guarantees it on another board or a hot day.

For reference the shipped 100 MHz build could only be pushed to ~166 MHz,
roughly 110,000 fps. The 375 MHz figure is a **2.26x** improvement, and the
signed-off 250 MHz figure alone is 1.5x.

Measured cycles/image is 1482-1511 against 1411 from synthesis. The gap is a
fixed ~2.7 ms of host overhead per batch (cache flush/invalidate plus Python
polling on `AP_DONE`), not accelerator time -- accelerator-only throughput at
375 MHz is ~265,700 fps.

### Why not faster

PL0 comes from the PS IOPLL, which only produces 1500/N MHz: 250, 300, 375.
A 275 MHz request silently snaps back to 250.

| constraint | WNS | verdict |
|---|---|---|
| 100 MHz | +0.996 ns | met, but leaves 2.25x on the table |
| 250 MHz | +0.010 ns | **met — use this** |
| 300 MHz | −0.276 ns | fails; ceiling is ~277 MHz |

Reaching 277 MHz would need an MMCM in the PL rather than the PS clock, for an
11% gain over 250 MHz.

## What changed from v1

- **No `ap_int` dependency.** v1 used `hlslib::DataPack`, which pulls in
  `ap_int.h` and confines the kernel to a Vitis environment. A plain 2-D array
  with `ARRAY_RESHAPE complete dim=2` gives the same one-wide-word-per-row
  memory. The whole kernel now builds with `g++`, so a full 10000-image check
  takes 0.66 s (`scripts/run_csim.sh`) instead of a csynth round-trip.
- **Template dimensions.** `Dense<K, M, Relu, DspLanes>` replaces five runtime
  ints (`K, N, M, D, W`) plus `LOOP_TRIPCOUNT` hints. `N` and `D` were always
  1 and `W` was always `M`.
- **No padding.** v1 padded every weight matrix to width 256 because `Vec_t`
  was fixed at 256 — layer 3 (256x10) stored 96% zeros. Weights drop from
  299,008 to 135,680 bytes (55%).
- **A testbench that can fail.** v1's ran one image, printed mismatches, and
  returned 0, so C simulation passed unconditionally. This one runs all 10000
  against a numpy golden reference, requires bit-exactness, and returns
  non-zero on failure.
- **Verified accumulator width.** v1 narrowed the accumulator from Vitis AI's
  32 bits to 16 with nothing checking that 16 bits suffice. Simulation now
  recomputes every layer in 32 bits and compares. Measured peak is **28,698
  against an int16 limit of 32,767 — 12% headroom**. It does not overflow on
  this data, but it is close, and a rescaled input or retrained model could
  break it silently.

## Two v1 findings worth knowing

**`MultiplyFabric` in v1 does nothing.** It carries
`#pragma HLS BIND_OP op=mul impl=fabric`, but Vitis 2025.1 rejects both
`mul + fabric` and `mul + dsp` as invalid for xck24. v1 only synthesised
cleanly because its pragma checker aborted on `DataPack.h`'s missing
`<cstddef>` before reaching the directive. The DSP/LUT split in v1 is HLS's
automatic allocation, not the pragma.

**The working control is global, not per-variable**: `syn.op=mul -impl dsp` in
the build config. Without it HLS puts all 394 multipliers on fabric.

## DSP budget

KD240 has 360 DSPs; a fully-parallel 784-128-256-10 network wants
128 + 256 + 10 = 394 multipliers, so 34 cannot have one. `kDspBudget` in
`src/mlp_top.cpp` sets how many lanes get a real `*` (and therefore a DSP);
the rest use `MulShiftAdd`, which is not a multiply operation and so is never
a DSP candidate.

Measured (HLS estimates, 4 ns target):

| variant | DSP | LUT | cycles/image | fits? |
|---|---|---|---|---|
| v1 | 256 | 37,002 | 1411 | yes |
| no `syn.op` | 0 | 51,325 | 1407 | yes |
| all lanes on DSP | 394 | 35,661 | 1410 | **no** — over 360 |
| `kDspBudget=354` | 354 | 49,005 | 1412 | yes |

The shift-add lanes cost ~325 LUT each, well above HLS's own fabric
multiplier, so trading fabric lanes for DSPs is currently a LUT *regression*.
v1's automatic split remains the best LUT result that fits. Genuinely reducing
LUT needs DSP48E2 int8 pair-packing (two products sharing one operand in one
DSP, 394 → ~197 DSPs and no fabric multipliers) — not implemented here.

## Layout

```
src/       mlp.hpp (dense layer), mlp_top.cpp (network), weights.hpp (generated)
test/      tb_mlp.cpp, data/ (golden vectors, generated)
scripts/   gen_weights.py, run_csim.sh, run_csynth.sh, gen_bd_tcl.sh
vivado/    build_impl.tcl  (design_1_<freq>.tcl and ip_repo/ are generated)
pynq/      bitstream, notebook, board test scripts
reports/   the synthesis and timing reports the numbers above come from
```

`results/` holds the raw Vivado and Vitis working directories (~1.5 GB) and is
git-ignored; `reports/` carries the curated output. To rebuild the IP catalog
entry `build_impl.tcl` expects:

```bash
unzip ../reports/vitis_2025.1_KD240/MultilayerPerceptron_KD240.zip \
      -d vivado/ip_repo/MultilayerPerceptron
```

## Usage

```bash
scripts/gen_weights.py             # regenerate weights.hpp + golden vectors
scripts/run_csim.sh                # full 10000-image functional check (0.7 s)
scripts/run_csynth.sh <name> [ns]  # C synthesis into results/<name>/
scripts/gen_bd_tcl.sh 250 > vivado/design_1_250mhz.tcl
cd vivado && vivado -mode batch -source build_impl.tcl \
    -tclargs design_1_250mhz.tcl ../results/impl_250
# set XILINX_ROOT if Vitis is not under /media/ntk/sda4/Xilinx
```

`build_impl.tcl` prints `RESULT_WNS_NS` and `RESULT_FMAX_MHZ` on completion.

## On the board

Files live in `/root/jupyter_notebooks/yuechi_MNIST_KD240/v2_250mhz/` on
`kria`; the v1 files one level up are untouched.

```
mnist_kd240.bit / .hwh     the 250 MHz build
golden_pred_i8.bin         host reference, 10000 predictions
run_board_test.py          correctness + throughput + ceiling
sweep.py                   clock sweep only
```

PYNQ needs `XILINX_XRT` and `BOARD` set or it reports "No Devices Found":

```bash
sudo XILINX_XRT=/usr BOARD=KD240 \
    /usr/local/share/pynq-venv/bin/python run_board_test.py
```

Note the PL clock **persists across sessions** -- it is not reset by loading a
bitstream. Set it explicitly rather than assuming the design frequency.

## Not yet done

- The v2 kernel has been through csynth only, not Vivado implementation. The
  bitstream measured above is v1's IP re-implemented at 250 MHz.
- DSP48E2 int8 pair-packing, the one route that would genuinely cut LUTs.
