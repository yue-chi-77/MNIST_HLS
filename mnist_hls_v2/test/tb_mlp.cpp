// Functional testbench for MultilayerPerceptron.
//
// v1's testbench ran one image, printed any mismatch, and returned 0 -- so C
// simulation passed unconditionally. This one runs the whole 10000-image test
// set, requires a bit-exact match against a numpy golden reference, checks
// that the int16 accumulator never overflowed, and returns non-zero on any
// failure so csim actually gates.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mlp.hpp"
#include "mlp_top.hpp"

#ifndef MLP_TEST_DATA_DIR
#define MLP_TEST_DATA_DIR "test/data"
#endif

namespace mlp {
long mlp_acc_peak = 0;
long mlp_acc_overflows = 0;
}  // namespace mlp

int mlp_tb_image_count = 0;

namespace {

constexpr int kPixels = 784;
constexpr int kMaxImages = 10000;
constexpr long kInt16Max = 32767;

std::vector<char> ReadAll(const std::string &path) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "FAIL: cannot open %s\n", path.c_str());
        std::exit(2);
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> buffer(static_cast<size_t>(size));
    if (std::fread(buffer.data(), 1, buffer.size(), f) != buffer.size()) {
        std::fprintf(stderr, "FAIL: short read on %s\n", path.c_str());
        std::exit(2);
    }
    std::fclose(f);
    return buffer;
}

}  // namespace

int main(int argc, char **argv) {
    const std::string dir =
        (argc > 1) ? argv[1] : (std::getenv("MLP_TEST_DATA_DIR")
                                    ? std::getenv("MLP_TEST_DATA_DIR")
                                    : MLP_TEST_DATA_DIR);

    const std::vector<char> images = ReadAll(dir + "/x_test_i8.bin");
    const std::vector<char> labels = ReadAll(dir + "/y_test_u8.bin");
    const std::vector<char> golden = ReadAll(dir + "/golden_pred_i8.bin");

    const int n_images = static_cast<int>(images.size() / kPixels);
    if (n_images != static_cast<int>(labels.size()) ||
        n_images != static_cast<int>(golden.size()) || n_images > kMaxImages) {
        std::fprintf(stderr, "FAIL: inconsistent test data (%d images)\n", n_images);
        return 2;
    }

    // Under csim the top function honours this instead of the synthesis-time
    // 10000, so a quick run can be requested with MLP_TB_IMAGES.
    const char *limit = std::getenv("MLP_TB_IMAGES");
    mlp_tb_image_count = limit ? std::atoi(limit) : n_images;
    if (mlp_tb_image_count <= 0 || mlp_tb_image_count > n_images) {
        mlp_tb_image_count = n_images;
    }

    std::vector<int8_t> predictions(static_cast<size_t>(mlp_tb_image_count), 0);
    MultilayerPerceptron(reinterpret_cast<const int8_t *>(images.data()),
                         predictions.data());

    int mismatches = 0;
    int correct = 0;
    for (int i = 0; i < mlp_tb_image_count; ++i) {
        if (predictions[i] != static_cast<int8_t>(golden[i])) {
            if (mismatches < 10) {
                std::printf("  mismatch at image %d: got %d, golden %d\n", i,
                            predictions[i], static_cast<int>(golden[i]));
            }
            ++mismatches;
        }
        if (predictions[i] == static_cast<int8_t>(labels[i])) ++correct;
    }

    const double accuracy = static_cast<double>(correct) / mlp_tb_image_count;
    std::printf("images            : %d\n", mlp_tb_image_count);
    std::printf("accuracy          : %.4f\n", accuracy);
    std::printf("golden mismatches : %d\n", mismatches);
    std::printf("peak |accumulator|: %ld (int16 limit %ld, %.0f%% headroom used)\n",
                mlp::mlp_acc_peak, kInt16Max,
                100.0 * mlp::mlp_acc_peak / kInt16Max);
    std::printf("accumulator overflows: %ld\n", mlp::mlp_acc_overflows);

    bool ok = true;
    if (mismatches != 0) {
        std::printf("FAIL: output differs from the golden reference\n");
        ok = false;
    }
    if (mlp::mlp_acc_overflows != 0) {
        std::printf("FAIL: the int16 accumulator overflowed\n");
        ok = false;
    }
    if (ok) std::printf("PASS\n");
    return ok ? 0 : 1;
}
