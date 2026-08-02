#!/usr/bin/env python3
"""Sweep the PL clock and find the highest bit-exact frequency.

The PS IOPLL emits 1500/N MHz, so these are the only reachable steps. The clock
persists across sessions, so it is set explicitly rather than assumed.

Correctness is judged by comparing every one of the 10000 predictions against a
host-computed reference. Accuracy is not a usable test here: a design past its
timing limit still scores ~0.97 while individual predictions are wrong.
"""

import os
import time

import numpy as np
from pynq import Overlay, allocate, ps

HERE = os.path.dirname(os.path.abspath(__file__))
N_IMAGES, PIXELS = 10000, 784
POLL_TIMEOUT_S = 10.0
STEPS = [249.997, 299.997, 374.996, 499.995]   # 1500/6, /5, /4, /3
DESIGN_MHZ = 249.997

parent = os.path.join(HERE, "..")
x = (np.load(os.path.join(parent, "x_test.npy")) // 32).astype(np.int8).reshape(N_IMAGES, PIXELS)
y = np.load(os.path.join(parent, "y_test.npy")).astype(np.uint8)
golden = np.frombuffer(open(os.path.join(HERE, "golden_pred_i8.bin"), "rb").read(), dtype=np.int8)

overlay = Overlay(os.path.join(HERE, "mnist_kd240.bit"))
regs = overlay.MultilayerPerceptron_0.register_map

in_buf = allocate(shape=(N_IMAGES * PIXELS,), dtype=np.int8)
out_buf = allocate(shape=(N_IMAGES,), dtype=np.int8)
regs.im_1.im = in_buf.device_address
regs.out_r_1.out_r = out_buf.device_address
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


print("CPU %.0f MHz\n" % ps.Clocks.cpu_mhz)
print("%9s  %8s  %10s  %11s  %s" % ("PL clock", "wrong", "batch ms", "fps", "cycles/img"))
print("-" * 62)

best = None
for target in STEPS:
    ps.Clocks.fclk0_mhz = target
    mhz = ps.Clocks.fclk0_mhz
    try:
        run()                                   # warm up, discard
        times = []
        for _ in range(5):
            t0 = time.time()
            result = run()
            times.append(time.time() - t0)
    except TimeoutError:
        print("%7.1f MHz  %8s" % (mhz, "HUNG"))
        break
    mean = sum(times) / len(times)
    wrong = int((result != golden).sum())
    print("%7.1f MHz  %8d  %10.2f  %11s  %10.0f" % (
        mhz, wrong, mean * 1e3, "{:,.0f}".format(N_IMAGES / mean),
        mean / N_IMAGES * mhz * 1e6))
    if wrong:
        print("           -> past the limit, stopping")
        break
    best = (mhz, N_IMAGES / mean)

ps.Clocks.fclk0_mhz = DESIGN_MHZ
print("\nrestored PL clock to %.3f MHz (design constraint)" % ps.Clocks.fclk0_mhz)
if best:
    print("highest bit-exact: %.1f MHz, {:,.0f} fps".format(best[1]) % best[0])
