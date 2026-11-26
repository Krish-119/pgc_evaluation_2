#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <cassert>

#include "../src/cuda_utils.h"
#include "../src/cpu_softmax.h"
#include "../src/separate_kernels.cuh"
#include "../src/fused_kernel.cuh"

struct TestCase {
    std::string name;
    int rows;
    int cols;
    int block_size;
    std::string description;
};

bool execute_test_case(
    const TestCase& tc,
    const std::vector<float>& h_input,
    std::ostream& report_out)
{
    int rows = tc.rows;
    int cols = tc.cols;
    int block_size = tc.block_size;
    size_t total = static_cast<size_t>(rows) * cols;
    size_t mat_bytes = total * sizeof(float);
    size_t row_bytes = rows * sizeof(float);

    std::vector<float> h_cpu(total, 0.0f);
    std::vector<float> h_sep(total, 0.0f);
    std::vector<float> h_fused(total, 0.0f);

    // Run CPU Reference (double-precision accumulation)
    cpu_softmax(h_input.data(), h_cpu.data(), rows, cols);

    // Allocate GPU Device Memory
    float *d_in, *d_out_sep, *d_out_fused, *d_row_max, *d_row_sum;
    CUDA_CHECK(cudaMalloc(&d_in, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_sep, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_fused, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_row_max, row_bytes));
    CUDA_CHECK(cudaMalloc(&d_row_sum, row_bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_input.data(), mat_bytes, cudaMemcpyHostToDevice));

    // Run Separate Pipeline
    separate_kernels::launch_separate_pipeline_clean(
        d_in, d_out_sep, d_row_max, d_row_sum, rows, cols, block_size);

    // Run Fused Kernel (Auto-selects SharedMemCached or StreamingFallback)
    fused_kernel::FusedImplementationType impl_used;
    fused_kernel::launch_fused_softmax(
        d_in, d_out_fused, rows, cols, block_size, &impl_used);

    CUDA_CHECK(cudaMemcpy(h_sep.data(), d_out_sep, mat_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fused.data(), d_out_fused, mat_bytes, cudaMemcpyDeviceToHost));

    // Strict 1.0e-5f tolerance validation
    auto val_sep = validate_softmax_output(h_cpu.data(), h_sep.data(), rows, cols, 1.0e-5f);
    auto val_fused = validate_softmax_output(h_cpu.data(), h_fused.data(), rows, cols, 1.0e-5f);

    bool passed = val_sep.passed && val_fused.passed;

    report_out << "------------------------------------------------------------------------\n";
    report_out << "TEST: " << tc.name << " [" << rows << "x" << cols << ", BlockSize=" << block_size 
               << ", Impl=" << ((impl_used == fused_kernel::SHARED_MEM_CACHED) ? "SharedMemCached" : "StreamingFallback") << "]\n";
    report_out << "Description: " << tc.description << "\n";
    report_out << "Result: " << (passed ? ">>> PASS <<<" : "*** FAIL ***") << "\n";
    report_out << "  Separate Max Abs Error : " << std::scientific << val_sep.max_abs_error << "\n";
    report_out << "  Fused Max Abs Error    : " << std::scientific << val_fused.max_abs_error << "\n";
    report_out << "  Fused Max Row-Sum Dev  : " << std::scientific << val_fused.max_row_sum_error << "\n";
    report_out << "  Mismatched Elements    : " << val_fused.mismatched_elements << " / " << total << "\n";

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out_sep));
    CUDA_CHECK(cudaFree(d_out_fused));
    CUDA_CHECK(cudaFree(d_row_max));
    CUDA_CHECK(cudaFree(d_row_sum));

    return passed;
}

int main() {
    GpuProperties gpu = get_gpu_properties(0);
    std::cout << "\n========================================================================\n";
    std::cout << "     AUTONOMOUS CORRECTNESS & NUMERICAL STABILITY TEST SUITE\n";
    std::cout << "========================================================================\n";
    std::cout << "GPU: " << gpu.name << " (Compute " << gpu.major << "." << gpu.minor << ")\n";
    std::cout << "Shared Memory per Block: " << (gpu.shared_mem_per_block / 1024) << " KB\n\n";

    std::ofstream report_file("results/correctness_results.txt");
    report_file << "========================================================================\n";
    report_file << "CUDA SOFTMAX - CORRECTNESS AND NUMERICAL STABILITY TEST REPORT\n";
    report_file << "Target GPU: " << gpu.name << " | Tolerance: 1.0e-5\n";
    report_file << "========================================================================\n\n";

    int total_tests = 0;
    int passed_tests = 0;

    // Test 1: Single element (1x1)
    {
        TestCase tc = {"Test_01_SingleElement", 1, 1, 64, "Minimal 1x1 matrix; output must be exactly 1.0"};
        std::vector<float> input = {42.0f};
        total_tests++;
        if (execute_test_case(tc, input, report_file)) passed_tests++;
    }

    // Test 2: Demonstration Matrix (2x4)
    {
        TestCase tc = {"Test_02_Demonstration", 2, 4, 256, "Academic theme mandatory demonstration matrix"};
        std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 1.0f, 0.0f, 3.0f};
        total_tests++;
        if (execute_test_case(tc, input, report_file)) passed_tests++;
    }

    // Test 3: All-Zero Rows (4x8)
    {
        TestCase tc = {"Test_03_AllZeros", 4, 8, 128, "All zeros: row output should be perfectly uniform 1/8 = 0.125"};
        std::vector<float> input(4 * 8, 0.0f);
        total_tests++;
        if (execute_test_case(tc, input, report_file)) passed_tests++;
    }

    // Test 4: Identical Non-Zero Values (4x16)
    {
        TestCase tc = {"Test_04_IdenticalElements", 4, 16, 128, "Identical 7.5f values: output uniform 1/16 = 0.0625"};
        std::vector<float> input(4 * 16, 7.5f);
        total_tests++;
        if (execute_test_case(tc, input, report_file)) passed_tests++;
    }

    // Test 5: Strictly Positive Values
    {
        TestCase tc = {"Test_05_StrictlyPositive", 8, 32, 256, "Strictly positive values range [10.0, 50.0]"};
        std::vector<float> input(8 * 32);
        for (size_t i = 0; i < input.size(); ++i) input[i] = 10.0f + (i % 40);
        total_tests++;
        if (execute_test_case(tc, input, report_file)) passed_tests++;
    }

    // Test 6: Strictly Negative Values
    {
        TestCase tc = {"Test_06_StrictlyNegative", 8, 32, 256, "Strictly negative values range [-50.0, -10.0]"};
        std::vector<float> input(8 * 32);
        for (size_t i = 0; i < input.size(); ++i) input[i] = -50.0f + (i % 40);
        total_tests++;
        if (execute_test_case(tc, input, report_file)) passed_tests++;
    }

    // Test 7: Extreme Large Values (+1000.0f) - Numerical Stability Overcoming Overflow
    {
        TestCase tc = {"Test_07_ExtremePositive_Stability", 4, 16, 128, "Extreme values +1000.0f: verifies safe max-subtraction prevents FP32 inf overflow"};
        std::vector<float> input(4 * 16, 0.0f);
        for (int r = 0; r < 4; ++r) {
            input[r * 16 + 0] = 1000.0f; // Dominant element
            input[r * 16 + 1] = 999.0f;
            input[r * 16 + 2] = 998.0f;
        }
        total_tests++;
        if (execute_test_case(tc, input, report_file)) passed_tests++;
    }

    // Test 8: Extreme Negative Values (-1000.0f) - Numerical Stability Underflow
    {
        TestCase tc = {"Test_08_ExtremeNegative_Stability", 4, 16, 128, "Extreme negative -1000.0f: verifies safe handling without zero-division NaN"};
        std::vector<float> input(4 * 16, -1000.0f);
        for (int r = 0; r < 4; ++r) {
            input[r * 16 + 0] = -950.0f; // Max element in row
        }
        total_tests++;
        if (execute_test_case(tc, input, report_file)) passed_tests++;
    }

    // Test 9: Non-Power-of-Two Dimensions (128x127)
    {
        TestCase tc = {"Test_09_NonPowerOfTwo_Cols", 128, 127, 256, "Cols=127 (non-power-of-two): tests boundary conditions and shared reductions"};
        std::vector<float> input(128 * 127);
        for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>((i % 17) - 8);
        total_tests++;
        if (execute_test_case(tc, input, report_file)) passed_tests++;
    }

    // Test 10: Irregular Matrix Dimensions (1024x1000)
    {
        TestCase tc = {"Test_10_Irregular_1024x1000", 1024, 1000, 256, "1024x1000 irregular large matrix with arbitrary values"};
        std::vector<float> input(1024 * 1000);
        for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>((i % 31) - 15) * 0.5f;
        total_tests++;
        if (execute_test_case(tc, input, report_file)) passed_tests++;
    }

    // Test 11: Rows Wider than CUDA Block Size (512x2048 with BlockSize=256)
    {
        TestCase tc = {"Test_11_RowsWiderThanBlockSize_2048", 512, 2048, 256, "Cols=2048 > BlockSize=256: tests multi-pass block-strided loop in single block"};
        std::vector<float> input(512 * 2048);
        for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>((i % 23) - 11) * 0.25f;
        total_tests++;
        if (execute_test_case(tc, input, report_file)) passed_tests++;
    }

    // Test 12: Very Long Rows (256x4096 with BlockSize=256)
    {
        TestCase tc = {"Test_12_VeryLongRows_4096", 256, 4096, 256, "Cols=4096: cached in dynamic shared memory (4096*4 = 16KB < 48KB limit)"};
        std::vector<float> input(256 * 4096);
        for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>((i % 19) - 9) * 0.1f;
        total_tests++;
        if (execute_test_case(tc, input, report_file)) passed_tests++;
    }

    // Test 13: Extreme Dimensions Exceeding Shared Memory Limit (Streaming Fallback)
    {
        TestCase tc = {"Test_13_StreamingFallback_16384", 64, 16384, 256, "Cols=16384 exceeds 48KB smem: verifies automatic streaming fallback"};
        std::vector<float> input(64 * 16384);
        for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>((i % 29) - 14) * 0.1f;
        total_tests++;
        if (execute_test_case(tc, input, report_file)) passed_tests++;
    }

    // Summary
    report_file << "\n========================================================================\n";
    report_file << "TEST SUITE SUMMARY: " << passed_tests << " / " << total_tests << " PASSED\n";
    report_file << "OVERALL STATUS: " << ((passed_tests == total_tests) ? "ALL TESTS PASSED [PASS]" : "FAILED") << "\n";
    report_file << "========================================================================\n";
    report_file.close();

    std::cout << "Test Suite Finished: " << passed_tests << " / " << total_tests << " Passed.\n";
    std::cout << "Overall Status: " << ((passed_tests == total_tests) ? "[PASS]" : "[FAIL]") << "\n";
    std::cout << "Report written to results/correctness_results.txt\n\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
