#pragma once

#include <stdint.h>

// Quantised fully-connected layer, fully unrolled across the output dimension.
//
// One input activation is consumed per cycle and broadcast to all M
// multipliers, so a layer takes K cycles.
//
// v1 passed K/M/N/D/W as runtime ints with LOOP_TRIPCOUNT hints and relied on
// Vitis constant-propagating them into three specialised instances. Here the
// dimensions are template parameters, so the specialisation is guaranteed by
// the language rather than by the tool's mood, and N/D/W -- which were always
// 1, 1 and M -- are gone.

namespace mlp {

#ifndef __SYNTHESIS__
// v1 narrowed the accumulator from Vitis AI's 32 bits to 16 so the 256-wide
// internal bus would fit. Nothing checked that 16 bits are actually enough --
// layer 1 sums 784 int8 products, whose worst case is far outside int16. In
// simulation every layer is recomputed in 32 bits and compared, so the
// testbench proves the width is sufficient for this data instead of assuming.
extern long mlp_acc_peak;      // largest |accumulator| observed
extern long mlp_acc_overflows; // times the int16 accumulator disagreed with int32
#endif

// int8 x int8 -> int16.
//
// v1 wraps this in `#pragma HLS BIND_OP op=mul impl=fabric` to keep layers 1
// and 3 off the DSPs, and that pragma works: synthesising v1 with and without
// it gives 256 vs 394 DSPs, a difference of exactly layer 1's 128 plus layer
// 3's 10. It is load-bearing -- without it the design wants 394 DSPs and does
// not fit in KD240's 360.
//
// v2 cannot use it. Vitis 2025.1's pragma lint wrongly reports
// `mul + fabric is invalid combination` and hard-fails the build; v1 escapes
// the false alarm only because that lint pass aborts earlier on DataPack.h's
// unresolvable <cstddef>. Dropping the ap_int dependency lets the lint run,
// so v2 falls back to the global `syn.op=mul -impl dsp` (every `*` takes a
// DSP) plus MulShiftAdd below for the lanes that must not have one. That
// fallback is measurably worse than what v1 gets.
inline int16_t Mul(int8_t a, int8_t b) {
#pragma HLS INLINE
    return (int16_t)((int16_t)a * (int16_t)b);
}

// Same product, built from shifts and adds instead of the `*` operator.
//
// With `syn.op=mul -impl dsp` in the build config every `*` claims a DSP, and
// the network wants 34 more than KD240 has. v1 steers the surplus back to
// fabric with a per-variable BIND_OP; this tree cannot, because the pragma
// lint hard-fails on it (see Mul above). An expression that is not a multiply
// at all is never a DSP candidate in the first place, so the shortfall is
// spelled out here instead -- at ~325 LUT per lane, which is worse than the
// fabric multiplier BIND_OP would have given us.
//
// b is signed, so bit 7 carries negative weight.
inline int16_t MulShiftAdd(int8_t a, int8_t b) {
#pragma HLS INLINE
    const int16_t wide = (int16_t)a;
    int16_t sum = 0;
    for (int s = 0; s < 7; ++s) {
#pragma HLS UNROLL
        if ((b >> s) & 1) sum = (int16_t)(sum + (int16_t)(wide << s));
    }
    if ((b >> 7) & 1) sum = (int16_t)(sum - (int16_t)(wide << 7));
    return sum;
}

// K inputs, M outputs. `scale` is a right shift, not a division: the Vitis AI
// quantiser emits power-of-two scale factors precisely so this is a free bit
// shift in hardware.
// DspLanes output channels use a real multiply (and therefore a DSP); the
// remaining M - DspLanes are built from shifts and adds on LUT fabric.
template <int K, int M, bool Relu, int DspLanes>
void Dense(const int8_t a[K],
           const int8_t w[K][M],
           const int8_t bias[M],
           int8_t scale,
           int8_t out[M]) {
#pragma HLS INLINE off
    // One wide word per input index, so a whole row of weights arrives in a
    // single cycle. This is the layout v1 got from hlslib::DataPack; reshaping
    // a plain 2-D array reaches it without the ap_int dependency -- and
    // without tripping the front-end crash that AGGREGATE-on-struct causes.
#pragma HLS ARRAY_RESHAPE variable=w complete dim=2

    int16_t acc[M];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

#ifndef __SYNTHESIS__
    int32_t acc_wide[M] = {};
#endif

Accumulate:
    for (int k = 0; k < K; ++k) {
#pragma HLS PIPELINE II=1
        const int8_t a_k = a[k];

        for (int i = 0; i < M; ++i) {
#pragma HLS UNROLL
            // Fully unrolled, so `i` is constant per instance and this picks
            // an implementation at compile time.
            const int16_t product = (i < DspLanes) ? Mul(a_k, w[k][i])
                                                   : MulShiftAdd(a_k, w[k][i]);
            acc[i] = (k == 0) ? product : (int16_t)(acc[i] + product);
#ifndef __SYNTHESIS__
            acc_wide[i] = (k == 0) ? (int32_t)product : acc_wide[i] + product;
#endif
        }
    }

#ifndef __SYNTHESIS__
    for (int i = 0; i < M; ++i) {
        const long magnitude = acc_wide[i] < 0 ? -(long)acc_wide[i] : (long)acc_wide[i];
        if (magnitude > mlp_acc_peak) mlp_acc_peak = magnitude;
        if (acc_wide[i] != (int32_t)acc[i]) ++mlp_acc_overflows;
    }
#endif

Activate:
    for (int i = 0; i < M; ++i) {
#pragma HLS UNROLL
        const int16_t scaled = (int16_t)((acc[i] + bias[i]) >> scale);
        out[i] = Relu ? (scaled < 0 ? (int8_t)0 : (int8_t)scaled) : (int8_t)scaled;
    }
}

}  // namespace mlp
