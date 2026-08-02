#!/usr/bin/env python3
"""Measure the 250 MHz KD240 build on hardware.

Run on the board, as root, inside the PYNQ virtualenv:

    sudo /usr/local/share/pynq-venv/bin/python run_board_test.py

Three things are checked, in this order:

1. Correctness against a host-computed golden reference, image by image.
   Accuracy on its own is not a correctness test: a design pushed past its
   timing limit still scores plausibly while individual predictions are wrong.
2. Throughput, timing the accelerator only. v1's notebook copied 7.84 MB into
   the DMA buffer inside the timed region, charging numpy's memcpy to the FPGA.
3. The real overclock ceiling, by stepping the PL clock and re-running (1).
"""

import os
import sys
import time

import numpy as np
from pynq import Overlay, allocate, ps

HERE = os.path.dirname(os.path.abspath(__file__))
N_IMAGES = 10000
PIXELS = 784
POLL_TIMEOUT_S = 10.0
REPEATS = 10

# The PS IOPLL emits 1500/N MHz, so these are the only values it can produce
# around the design target -- anything else silently snaps to one of them.
CLOCK_STEPS = [299.997, 374.996]


def load_inputs():
    parent = os.path.join(HERE, "..")
    x = (np.load(os.path.join(parent, "x_test.npy")) // 32).astype(np.int8)
    x = x.reshape(N_IMAGES, PIXELS)
    y = np.load(os.path.join(parent, "y_test.npy")).astype(np.uint8)
    with open(os.path.join(HERE, "golden_pred_i8.bin"), "rb") as handle:
        golden = np.frombuffer(handle.read(), dtype=np.int8)
    if golden.size != N_IMAGES:
        raise ValueError("golden has %d entries, expected %d" % (golden.size, N_IMAGES))
    return x, y, golden


def main():
    x, y, golden = load_inputs()

    overlay = Overlay(os.path.join(HERE, "mnist_kd240.bit"))
    regs = overlay.MultilayerPerceptron_0.register_map

    design_mhz = ps.Clocks.fclk0_mhz
    print("PL clock  : %.3f MHz" % design_mhz)
    print("CPU clock : %.2f MHz" % ps.Clocks.cpu_mhz)

    in_buf = allocate(shape=(N_IMAGES * PIXELS,), dtype=np.int8)
    out_buf = allocate(shape=(N_IMAGES,), dtype=np.int8)
    regs.im_1.im = in_buf.device_address
    regs.out_r_1.out_r = out_buf.device_address

    # Setup, not inference: loaded once, deliberately outside the timing.
    in_buf[:] = x.reshape(-1)
    in_buf.flush()

    def run():
        out_buf[:] = 0
        out_buf.flush()
        regs.CTRL.AP_START = 1
        deadline = time.time() + POLL_TIMEOUT_S
        while regs.CTRL.AP_DONE == 0:
            if time.time() > deadline:
                raise TimeoutError("AP_DONE never asserted")
        out_buf.invalidate()
        return np.array(out_buf)

    def timed_batch():
        start = time.time()
        result = run()
        return time.time() - start, result

    print("\n=== correctness at design frequency ===")
    result = run()
    wrong = int((result != golden).sum())
    print("mismatches vs golden : %d" % wrong)
    print("accuracy (hardware)  : %.4f" % (result == y).mean())
    print("accuracy (golden)    : %.4f" % (golden == y).mean())
    if wrong:
        print("FAIL: hardware disagrees with the reference at its design clock")
        return 1

    print("\n=== throughput ===")
    times = [timed_batch()[0] for _ in range(REPEATS)]
    mean = sum(times) / len(times)
    print("batch      : %.2f ms mean, %.2f ms best (%d runs of %d images)"
          % (mean * 1e3, min(times) * 1e3, REPEATS, N_IMAGES))
    print("per image  : %.3f us" % (mean / N_IMAGES * 1e6))
    print("throughput : {:,.0f} fps".format(N_IMAGES / mean))
    print("cycles/img : %.0f (at %.0f MHz)"
          % (mean / N_IMAGES * design_mhz * 1e6, design_mhz))

    print("\n=== overclock ceiling ===")
    print("Raising the PL clock and re-checking every prediction.")
    highest_good = design_mhz
    best_fps = N_IMAGES / mean
    try:
        for target in CLOCK_STEPS:
            if target <= design_mhz + 0.1:
                continue
            try:
                ps.Clocks.fclk0_mhz = target
            except Exception as exc:
                print("  %7.1f MHz : cannot set (%s)" % (target, exc))
                break
            actual = ps.Clocks.fclk0_mhz
            try:
                elapsed, result = timed_batch()
            except TimeoutError:
                print("  %7.1f MHz : HUNG, no AP_DONE" % actual)
                break
            wrong = int((result != golden).sum())
            if wrong:
                print("  %7.1f MHz : %d wrong predictions -- past the limit"
                      % (actual, wrong))
                break
            print("  {:7.1f} MHz : ok, {:,.0f} fps".format(actual, N_IMAGES / elapsed))
            highest_good, best_fps = actual, N_IMAGES / elapsed
    finally:
        ps.Clocks.fclk0_mhz = design_mhz
        print("restored PL clock to %.3f MHz" % ps.Clocks.fclk0_mhz)

    print("\nhighest bit-exact clock : %.1f MHz" % highest_good)
    print("throughput there        : {:,.0f} fps".format(best_fps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
