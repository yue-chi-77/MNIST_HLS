#include "mlp_top.hpp"

#include "mlp.hpp"
#include "weights.hpp"

// 784 -> 128 -> 256 -> 10, matching v1 exactly.
//
// The external interface (m_axi depths, s_axilite register layout) is kept
// identical to v1 on purpose: the existing Vivado block design and the PYNQ
// register map stay valid, so this is a drop-in bitstream replacement.

namespace {

constexpr int kIn = 784;
constexpr int kL1 = 128;
constexpr int kL2 = 256;
constexpr int kOut = 10;
constexpr int kImages = 10000;

// DSP budget. KD240 has 360; the network wants 128+256+10 = 394 multipliers,
// so 40 are given shift-add implementations on fabric to leave a placement
// margin. v1 put 138 on fabric -- an entire layer at a time -- because its
// only control was per layer.
constexpr int kDspBudget = 354;
constexpr int kL2Dsp = kL2;   // the widest layer keeps all of its DSPs
constexpr int kL3Dsp = kOut;  // only 10, cheap to satisfy
constexpr int kL1Dsp = kDspBudget - kL2Dsp - kL3Dsp;

static_assert(kL1Dsp >= 0 && kL1Dsp <= kL1, "layer 1 DSP lanes out of range");
static_assert(kL1Dsp + kL2Dsp + kL3Dsp <= 360, "over the KD240 DSP budget");

constexpr int kBias1 = 0;
constexpr int kBias2 = kL1;
constexpr int kBias3 = kL1 + kL2;


static_assert(sizeof(weights1) == kIn * kL1, "weights1 has the wrong shape");
static_assert(sizeof(weights2) == kL1 * kL2, "weights2 has the wrong shape");
static_assert(sizeof(weights3) == kL2 * kOut, "weights3 has the wrong shape");

int8_t ArgMax(const int8_t scores[kOut]) {
#pragma HLS INLINE off
    int8_t best = 0;
    int8_t best_value = -128;
    for (int j = 0; j < kOut; ++j) {
#pragma HLS PIPELINE II=1
        if (scores[j] > best_value) {
            best = (int8_t)j;
            best_value = scores[j];
        }
    }
    return best;
}

}  // namespace

void MultilayerPerceptron(const int8_t im[], int8_t out[]) {
#pragma HLS INTERFACE m_axi port=im bundle=gmem0 offset=slave depth=7840000
#pragma HLS INTERFACE m_axi port=out bundle=gmem1 offset=slave depth=10000
#pragma HLS INTERFACE s_axilite port=im bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int8_t act1[kL1];
    int8_t act2[kL2];
    int8_t act3[kOut];

    int n_images = kImages;
#ifndef __SYNTHESIS__
    n_images = mlp_tb_image_count;
#endif

Images:
    for (int n = 0; n < n_images; ++n) {
        mlp::Dense<kIn, kL1, true, kL1Dsp>(
            im + kIn * n,
            weights1,
            bias + kBias1, scales[0], act1);

        mlp::Dense<kL1, kL2, true, kL2Dsp>(
            act1,
            weights2,
            bias + kBias2, scales[1], act2);

        mlp::Dense<kL2, kOut, false, kL3Dsp>(
            act2,
            weights3,
            bias + kBias3, scales[2], act3);

        out[n] = ArgMax(act3);
    }
}
