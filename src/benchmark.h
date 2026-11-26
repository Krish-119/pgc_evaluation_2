#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <vector>
#include <string>
#include <random>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

#include "cuda_utils.h"
#include "cpu_softmax.h"
#include "separate_kernels.cuh"
#include "fused_kernel.cuh"

struct BenchmarkRecord {
    std::string config_name;
    int rows;
    int cols;
    int block_size;
    std::string fused_implementation; // "SharedMemCached" or "StreamingFallback"
    int warmup_runs;
    int measured_runs;

    double cpu_time_ms;
    float separate_max_time_ms;
    float separate_sum_time_ms;
    float separate_norm_time_ms;
    float separate_total_time_ms; // Measured via clean pipeline timing
    float fused_time_ms;

    float speedup;
    float time_reduction_pct;

    // Logical Memory Traffic Accounting (Algorithmic Scalar Model)
    // Separate: (4RC+4R) + (4RC+8R) + (8RC+8R) = 16RC + 20R bytes
    // Fused Cached: 4RC (Read X) + 4RC (Write Y) = 8RC bytes
    // Fused Streaming Fallback: 12RC (Read X) + 4RC (Write Y) = 16RC bytes
    size_t separate_bytes_moved;
    size_t fused_bytes_moved;
    size_t bytes_saved;
    float memory_traffic_reduction_pct;
    float fused_bandwidth_gb_s; // Decimal GB/s (1e9 bytes/s)

    // Independent numerical error reporting vs CPU reference (tolerance 1e-5f)
    float separate_max_abs_error;
    float fused_max_abs_error;
    float fused_mean_abs_error;
    float fused_max_row_sum_error;
    bool separate_passed;
    bool fused_passed;
    bool passed;
};

inline void generate_matrix_uniform(float* mat, int rows, int cols, float min_val = -5.0f, float max_val = 5.0f, unsigned int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    int total = rows * cols;
    for (int i = 0; i < total; ++i) {
        mat[i] = dist(rng);
    }
}

inline BenchmarkRecord run_single_benchmark(
    int rows,
    int cols,
    int block_size = 256,
    int warmup = 10,
    int iterations = 100,
    const std::string& name = "Standard")
{
    BenchmarkRecord rec;
    rec.config_name = name;
    rec.rows = rows;
    rec.cols = cols;
    rec.block_size = block_size;
    rec.warmup_runs = warmup;
    rec.measured_runs = iterations;

    size_t num_elements = static_cast<size_t>(rows) * cols;
    size_t matrix_bytes = num_elements * sizeof(float);
    size_t row_aux_bytes = rows * sizeof(float);

    std::vector<float> h_input(num_elements);
    std::vector<float> h_cpu_output(num_elements);
    std::vector<float> h_sep_output(num_elements);
    std::vector<float> h_fused_output(num_elements);

    generate_matrix_uniform(h_input.data(), rows, cols, -5.0f, 5.0f, 1337);

    // CPU Reference Timing (double-precision accumulation)
    rec.cpu_time_ms = cpu_softmax(h_input.data(), h_cpu_output.data(), rows, cols);

    // Allocate GPU Device Memory
    float *d_input = nullptr, *d_output_sep = nullptr, *d_output_fused = nullptr;
    float *d_row_max = nullptr, *d_row_sum = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_output_sep, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_output_fused, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_row_max, row_aux_bytes));
    CUDA_CHECK(cudaMalloc(&d_row_sum, row_aux_bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), matrix_bytes, cudaMemcpyHostToDevice));

    // -------------------------------------------------------------
    // Separate Kernels: Warm-up and Benchmarking
    // Primary pipeline comparison uses clean timing around complete 3-kernel sequence.
    // -------------------------------------------------------------
    for (int i = 0; i < warmup; ++i) {
        separate_kernels::launch_separate_pipeline_clean(
            d_input, d_output_sep, d_row_max, d_row_sum, rows, cols, block_size);
    }

    double acc_sep_total = 0.0;
    for (int i = 0; i < iterations; ++i) {
        float t = separate_kernels::launch_separate_pipeline_clean(
            d_input, d_output_sep, d_row_max, d_row_sum, rows, cols, block_size);
        acc_sep_total += t;
    }
    rec.separate_total_time_ms = static_cast<float>(acc_sep_total / iterations);

    // Measure individual kernel execution times separately
    auto ind_times = separate_kernels::measure_separate_individual_kernels(
        d_input, d_output_sep, d_row_max, d_row_sum, rows, cols, block_size);
    rec.separate_max_time_ms = ind_times.max_kernel_ms;
    rec.separate_sum_time_ms = ind_times.sum_kernel_ms;
    rec.separate_norm_time_ms = ind_times.norm_kernel_ms;

    // -------------------------------------------------------------
    // Fused Kernel: Warm-up and Benchmarking
    // -------------------------------------------------------------
    fused_kernel::FusedImplementationType impl_used = fused_kernel::SHARED_MEM_CACHED;
    for (int i = 0; i < warmup; ++i) {
        fused_kernel::launch_fused_softmax(d_input, d_output_fused, rows, cols, block_size, &impl_used);
    }

    double acc_fused = 0.0;
    for (int i = 0; i < iterations; ++i) {
        float t = fused_kernel::launch_fused_softmax(d_input, d_output_fused, rows, cols, block_size, &impl_used);
        acc_fused += t;
    }
    rec.fused_time_ms = static_cast<float>(acc_fused / iterations);
    rec.fused_implementation = (impl_used == fused_kernel::SHARED_MEM_CACHED) ? "SharedMemCached" : "StreamingFallback";

    // -------------------------------------------------------------
    // Verify Correctness against CPU Reference (Strict 1e-5f Tolerance)
    // -------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_sep_output.data(), d_output_sep, matrix_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fused_output.data(), d_output_fused, matrix_bytes, cudaMemcpyDeviceToHost));

    auto val_sep = validate_softmax_output(h_cpu_output.data(), h_sep_output.data(), rows, cols, 1.0e-5f);
    auto val_fused = validate_softmax_output(h_cpu_output.data(), h_fused_output.data(), rows, cols, 1.0e-5f);

    rec.separate_passed = val_sep.passed;
    rec.fused_passed = val_fused.passed;
    rec.passed = val_sep.passed && val_fused.passed;

    rec.separate_max_abs_error = val_sep.max_abs_error;
    rec.fused_max_abs_error = val_fused.max_abs_error;
    rec.fused_mean_abs_error = val_fused.mean_abs_error;
    rec.fused_max_row_sum_error = val_fused.max_row_sum_error;

    // -------------------------------------------------------------
    // Derived Metrics & Algorithmic Memory Traffic (Issue 2 & 3)
    // -------------------------------------------------------------
    rec.speedup = rec.separate_total_time_ms / rec.fused_time_ms;
    rec.time_reduction_pct = ((rec.separate_total_time_ms - rec.fused_time_ms) / rec.separate_total_time_ms) * 100.0f;

    // Separate Kernels Logical Traffic:
    // K1: Read X (4RC) + Write d_row_max (4R) = 4RC + 4R
    // K2: Read X (4RC) + Read d_row_max (4R) + Write d_row_sum (4R) = 4RC + 8R
    // K3: Read X (4RC) + Read d_row_max (4R) + Read d_row_sum (4R) + Write Y (4RC) = 8RC + 8R
    // Total = 16RC + 20R bytes
    rec.separate_bytes_moved = static_cast<size_t>(16) * rows * cols + static_cast<size_t>(20) * rows;

    if (impl_used == fused_kernel::SHARED_MEM_CACHED) {
        // Shared-Memory Cached Fused Kernel:
        // Read X (cooperatively loaded into shared memory ONCE) = 4RC bytes
        // Write Y (normalized output written to global memory ONCE) = 4RC bytes
        // Zero intermediate global memory arrays = 0 bytes
        // Total = 8RC bytes (8 bytes per FP32 element)
        rec.fused_bytes_moved = static_cast<size_t>(8) * rows * cols;
    } else {
        // Fallback streaming kernel (multi-pass without full row caching)
        // 3 reads of X (12RC) + 1 write of Y (4RC) = 16RC bytes
        rec.fused_bytes_moved = static_cast<size_t>(16) * rows * cols;
    }

    if (rec.fused_bytes_moved < rec.separate_bytes_moved) {
        rec.bytes_saved = rec.separate_bytes_moved - rec.fused_bytes_moved;
        rec.memory_traffic_reduction_pct = (static_cast<float>(rec.bytes_saved) / rec.separate_bytes_moved) * 100.0f;
    } else {
        rec.bytes_saved = 0;
        rec.memory_traffic_reduction_pct = 0.0f;
    }

    // Effective Logical Bandwidth in Decimal GB/s (10^9 bytes per GB)
    double sec = rec.fused_time_ms / 1000.0;
    rec.fused_bandwidth_gb_s = static_cast<float>((rec.fused_bytes_moved / 1.0e9) / sec);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output_sep));
    CUDA_CHECK(cudaFree(d_output_fused));
    CUDA_CHECK(cudaFree(d_row_max));
    CUDA_CHECK(cudaFree(d_row_sum));

    return rec;
}

inline void export_benchmark_csv(const std::vector<BenchmarkRecord>& records, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open " << path << " for writing." << std::endl;
        return;
    }
    ofs << "Config,Rows,Cols,BlockSize,FusedImpl,CPUTimeMs,SeparateMaxMs,SeparateSumMs,SeparateNormMs,"
        << "SeparateTotalMs,FusedTimeMs,Speedup,TimeReductionPct,SeparateBytesMoved,FusedBytesMoved,"
        << "BytesSaved,TrafficReductionPct,FusedBandwidthGBs,SeparateMaxAbsError,FusedMaxAbsError,Passed\n";

    for (const auto& r : records) {
        ofs << r.config_name << ","
            << r.rows << ","
            << r.cols << ","
            << r.block_size << ","
            << r.fused_implementation << ","
            << std::fixed << std::setprecision(4)
            << r.cpu_time_ms << ","
            << r.separate_max_time_ms << ","
            << r.separate_sum_time_ms << ","
            << r.separate_norm_time_ms << ","
            << r.separate_total_time_ms << ","
            << r.fused_time_ms << ","
            << std::setprecision(2)
            << r.speedup << ","
            << r.time_reduction_pct << ","
            << r.separate_bytes_moved << ","
            << r.fused_bytes_moved << ","
            << r.bytes_saved << ","
            << r.memory_traffic_reduction_pct << ","
            << r.fused_bandwidth_gb_s << ","
            << std::scientific << std::setprecision(4)
            << r.separate_max_abs_error << ","
            << r.fused_max_abs_error << ","
            << (r.passed ? "PASS" : "FAIL") << "\n";
    }
    ofs.close();
    std::cout << "[INFO] Benchmark CSV exported to: " << path << std::endl;
}

inline void export_benchmark_json(const std::vector<BenchmarkRecord>& records, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open " << path << " for writing." << std::endl;
        return;
    }
    ofs << "{\n  \"benchmarks\": [\n";
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        ofs << "    {\n"
            << "      \"config\": \"" << r.config_name << "\",\n"
            << "      \"rows\": " << r.rows << ",\n"
            << "      \"cols\": " << r.cols << ",\n"
            << "      \"block_size\": " << r.block_size << ",\n"
            << "      \"fused_implementation\": \"" << r.fused_implementation << "\",\n"
            << "      \"cpu_time_ms\": " << r.cpu_time_ms << ",\n"
            << "      \"separate_max_ms\": " << r.separate_max_time_ms << ",\n"
            << "      \"separate_sum_ms\": " << r.separate_sum_time_ms << ",\n"
            << "      \"separate_norm_ms\": " << r.separate_norm_time_ms << ",\n"
            << "      \"separate_total_ms\": " << r.separate_total_time_ms << ",\n"
            << "      \"fused_time_ms\": " << r.fused_time_ms << ",\n"
            << "      \"speedup\": " << r.speedup << ",\n"
            << "      \"time_reduction_pct\": " << r.time_reduction_pct << ",\n"
            << "      \"separate_bytes_moved\": " << r.separate_bytes_moved << ",\n"
            << "      \"fused_bytes_moved\": " << r.fused_bytes_moved << ",\n"
            << "      \"bytes_saved\": " << r.bytes_saved << ",\n"
            << "      \"traffic_reduction_pct\": " << r.memory_traffic_reduction_pct << ",\n"
            << "      \"fused_bandwidth_gb_s\": " << r.fused_bandwidth_gb_s << ",\n"
            << "      \"separate_max_abs_error\": " << r.separate_max_abs_error << ",\n"
            << "      \"fused_max_abs_error\": " << r.fused_max_abs_error << ",\n"
            << "      \"passed\": " << (r.passed ? "true" : "false") << "\n"
            << "    }" << (i + 1 < records.size() ? ",\n" : "\n");
    }
    ofs << "  ]\n}\n";
    ofs.close();
    std::cout << "[INFO] Benchmark JSON exported to: " << path << std::endl;
}

#endif // BENCHMARK_H
