#pragma once

#include <stdint.h>

// Classifies a batch of MNIST images held in DRAM.
//
//   im  : kImages * 784 int8 pixels, image-major
//   out : kImages int8 predicted digits
//
// The batch size is fixed at 10000 for synthesis so the host pays the PCIe/AXI
// setup cost once for the whole test set rather than per image.
void MultilayerPerceptron(const int8_t im[], int8_t out[]);

#ifndef __SYNTHESIS__
// Lets the testbench run a short batch without resynthesising. Ignored by HLS.
extern int mlp_tb_image_count;
#endif
