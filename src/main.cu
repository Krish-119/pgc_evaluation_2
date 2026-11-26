#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdlib>

#include "cuda_utils.h"
#include "cpu_softmax.h"
#include "separate_kernels.cuh"
#include "fused_kernel.cuh"
#include "benchmark.h"

void print_banner(const GpuProperties& gpu) {
    std::cout << "\n========================================================================\n";
    std::cout << "          SOFTMAX AND KERNEL FUSION USING CUDA - PGC LAB\n";
    std::cout << "========================================================================\n";
    std::cout << "  Active GPU Hardware : " << gpu.name << "\n";
    std::cout << "  Compute Capability  : " << gpu.major << "." << gpu.minor << "\n";
    std::cout << "  SM Multiprocessors  : " << gpu.sm_count << "\n";
    std::cout << "  Global Memory (VRAM): " << std::fixed << std::setprecision(2)
              << (gpu.total_global_mem_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    std::cout << "  Shared Memory/Block : " << (gpu.shared_mem_per_block / 1024) << " KB ("
              << gpu.shared_mem_per_block << " bytes)\n";
    std::cout << "========================================================================\n\n";
}

void run_demonstration_mode(const GpuProperties& gpu) {
    std::cout << "\n>>> [MODE 1] SOFTMAX DEMONSTRATION MODE <<<\n";
    std::cout << "Input Matrix: 2 rows x 4 columns (Mandatory Evaluation Case)\n";

    const int rows = 2;
    const int cols = 4;
    const int block_size = 256;

    std::vector<float> h_input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 1.0f, 0.0f, 3.0f
    };

    std::vector<float> h_cpu_out(rows * cols, 0.0f);
    std::vector<float> h_sep_out(rows * cols, 0.0f);
    std::vector<float> h_fused_out(rows * cols, 0.0f);

    // CPU Softmax (ground truth reference with double-precision accumulation)
    double cpu_time = cpu_softmax(h_input.data(), h_cpu_out.data(), rows, cols);

    // GPU Allocations
    float *d_in, *d_out_sep, *d_out_fused, *d_row_max, *d_row_sum;
    size_t mat_bytes = rows * cols * sizeof(float);
    size_t row_bytes = rows * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_in, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_sep, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_fused, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_row_max, row_bytes));
    CUDA_CHECK(cudaMalloc(&d_row_sum, row_bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_input.data(), mat_bytes, cudaMemcpyHostToDevice));

    // Separate Kernels Execution with individual timings
    auto sep_timing = separate_kernels::measure_separate_individual_kernels(
        d_in, d_out_sep, d_row_max, d_row_sum, rows, cols, block_size);

    // Fused Kernel Execution (Shared memory row cached)
    fused_kernel::FusedImplementationType impl_used = fused_kernel::SHARED_MEM_CACHED;
    float fused_time = fused_kernel::launch_fused_softmax(
        d_in, d_out_fused, rows, cols, block_size, &impl_used);

    CUDA_CHECK(cudaMemcpy(h_sep_out.data(), d_out_sep, mat_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fused_out.data(), d_out_fused, mat_bytes, cudaMemcpyDeviceToHost));

    // Print Matrices
    print_matrix_sample(h_input.data(), rows, cols, "Input Matrix X");
    print_matrix_sample(h_cpu_out.data(), rows, cols, "CPU Reference Softmax Output");
    print_matrix_sample(h_sep_out.data(), rows, cols, "CUDA Separate Kernels Output");
    print_matrix_sample(h_fused_out.data(), rows, cols, "CUDA Fused Kernel Output");

    // Independent validations using 1.0e-5f tolerance
    auto val_sep = validate_softmax_output(h_cpu_out.data(), h_sep_out.data(), rows, cols, 1.0e-5f);
    auto val_fused = validate_softmax_output(h_cpu_out.data(), h_fused_out.data(), rows, cols, 1.0e-5f);

    std::cout << "--- Execution & Verification Summary ---\n";
    std::cout << "  CPU Execution Time          : " << std::fixed << std::setprecision(4) << cpu_time << " ms\n";
    std::cout << "  Separate Kernel 1 (Row Max) : " << sep_timing.max_kernel_ms << " ms\n";
    std::cout << "  Separate Kernel 2 (Exp Sum) : " << sep_timing.sum_kernel_ms << " ms\n";
    std::cout << "  Separate Kernel 3 (Normalize): " << sep_timing.norm_kernel_ms << " ms\n";
    std::cout << "  Total Separate Pipeline Time: " << sep_timing.total_pipeline_ms << " ms\n";
    std::cout << "  Fused Kernel Execution Time : " << fused_time << " ms ("
              << ((impl_used == fused_kernel::SHARED_MEM_CACHED) ? "SharedMemCached" : "StreamingFallback") << ")\n";
    std::cout << "  Speedup (Separate / Fused)  : " << std::setprecision(2) << (sep_timing.total_pipeline_ms / fused_time) << "x\n";
    std::cout << "  Separate Output Status      : " << (val_sep.passed ? "[PASS]" : "[FAIL]") 
              << " (Max Abs Error: " << std::scientific << val_sep.max_abs_error << ")\n";
    std::cout << "  Fused Output Status         : " << (val_fused.passed ? "[PASS]" : "[FAIL]") 
              << " (Max Abs Error: " << std::scientific << val_fused.max_abs_error << ")\n";
    std::cout << "  Row-Sum Check (|Sum - 1.0|) : " << val_fused.max_row_sum_error << "\n";
    std::cout << "----------------------------------------\n";

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out_sep));
    CUDA_CHECK(cudaFree(d_out_fused));
    CUDA_CHECK(cudaFree(d_row_max));
    CUDA_CHECK(cudaFree(d_row_sum));
}

void run_custom_input_mode(const GpuProperties& gpu) {
    std::cout << "\n>>> [MODE 2] CUSTOM MATRIX INPUT <<<\n";
    int rows = 0, cols = 0, block_size = 256;
    std::cout << "Enter number of rows: ";
    if (!(std::cin >> rows) || rows <= 0) {
        std::cout << "[ERROR] Invalid row count. Aborting.\n";
        std::cin.clear(); std::cin.ignore(1000, '\n');
        return;
    }
    std::cout << "Enter number of columns: ";
    if (!(std::cin >> cols) || cols <= 0) {
        std::cout << "[ERROR] Invalid column count. Aborting.\n";
        std::cin.clear(); std::cin.ignore(1000, '\n');
        return;
    }
    std::cout << "Enter thread block size (default 256): ";
    std::cin >> block_size;
    if (block_size <= 0 || block_size > 1024) block_size = 256;

    std::vector<float> h_input(rows * cols);
    if (rows * cols <= 32) {
        std::cout << "Enter " << (rows * cols) << " floating-point values separated by spaces:\n";
        for (int i = 0; i < rows * cols; ++i) {
            std::cin >> h_input[i];
        }
    } else {
        std::cout << "[INFO] Generating reproducible uniform random values for " << rows << "x" << cols << " matrix...\n";
        generate_matrix_uniform(h_input.data(), rows, cols, -3.0f, 3.0f, 999);
    }

    std::vector<float> h_cpu_out(rows * cols);
    std::vector<float> h_sep_out(rows * cols);
    std::vector<float> h_fused_out(rows * cols);

    double cpu_time = cpu_softmax(h_input.data(), h_cpu_out.data(), rows, cols);

    float *d_in, *d_out_sep, *d_out_fused, *d_row_max, *d_row_sum;
    size_t mat_bytes = rows * cols * sizeof(float);
    size_t row_bytes = rows * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_in, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_sep, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_fused, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_row_max, row_bytes));
    CUDA_CHECK(cudaMalloc(&d_row_sum, row_bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_input.data(), mat_bytes, cudaMemcpyHostToDevice));

    auto sep_timing = separate_kernels::measure_separate_individual_kernels(
        d_in, d_out_sep, d_row_max, d_row_sum, rows, cols, block_size);
    fused_kernel::FusedImplementationType impl_used = fused_kernel::SHARED_MEM_CACHED;
    float fused_time = fused_kernel::launch_fused_softmax(
        d_in, d_out_fused, rows, cols, block_size, &impl_used);

    CUDA_CHECK(cudaMemcpy(h_sep_out.data(), d_out_sep, mat_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fused_out.data(), d_out_fused, mat_bytes, cudaMemcpyDeviceToHost));

    print_matrix_sample(h_input.data(), rows, cols, "Custom Input");
    print_matrix_sample(h_fused_out.data(), rows, cols, "Fused Softmax Output");

    auto val_fused = validate_softmax_output(h_cpu_out.data(), h_fused_out.data(), rows, cols, 1.0e-5f);
    std::cout << "\nResults for " << rows << "x" << cols << " matrix:\n";
    std::cout << "  Separate Time: " << sep_timing.total_pipeline_ms << " ms\n";
    std::cout << "  Fused Time   : " << fused_time << " ms (" 
              << ((impl_used == fused_kernel::SHARED_MEM_CACHED) ? "SharedMemCached" : "StreamingFallback") << ")\n";
    std::cout << "  Speedup      : " << (sep_timing.total_pipeline_ms / fused_time) << "x\n";
    std::cout << "  Status       : " << (val_fused.passed ? "PASS" : "FAIL") << "\n";

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out_sep));
    CUDA_CHECK(cudaFree(d_out_fused));
    CUDA_CHECK(cudaFree(d_row_max));
    CUDA_CHECK(cudaFree(d_row_sum));
}

void run_complete_benchmark_suite(const GpuProperties& gpu) {
    std::cout << "\n>>> [MODE 3] RUNNING COMPREHENSIVE BENCHMARK SUITE <<<\n";
    std::cout << "Benchmarking 10 Warmup + 100 Measured Iterations per configuration...\n";
    std::cout << "Measuring clean pipeline timing boundaries for both Separate and Fused implementations.\n\n";

    struct Config {
        std::string name;
        int rows;
        int cols;
        int block_size;
    };

    std::vector<Config> configs = {
        {"Config_1024x256",   1024,  256, 256},
        {"Config_2048x512",   2048,  512, 256},
        {"Config_4096x1024",  4096, 1024, 256},
        {"Config_1024x2048",  1024, 2048, 256},
        {"Config_1024x4096",  1024, 4096, 256},
        {"Config_128x127_NonPow2", 128, 127, 256},
        {"Config_1024x1000_Irreg", 1024, 1000, 256}
    };

    std::vector<BenchmarkRecord> records;

    std::cout << std::left 
              << std::setw(24) << "Configuration"
              << std::setw(12) << "Dimensions"
              << std::setw(14) << "Separate (ms)"
              << std::setw(12) << "Fused (ms)"
              << std::setw(10) << "Speedup"
              << std::setw(18) << "Saved Logical Traffic"
              << std::setw(8)  << "Status"
              << "\n";
    std::cout << std::string(98, '-') << "\n";

    for (const auto& cfg : configs) {
        auto rec = run_single_benchmark(cfg.rows, cfg.cols, cfg.block_size, 10, 100, cfg.name);
        records.push_back(rec);

        std::stringstream dims;
        dims << rec.rows << "x" << rec.cols;

        std::stringstream saved_bw;
        saved_bw << std::fixed << std::setprecision(1) << (rec.bytes_saved / 1.0e6) << " MB (" 
                 << std::setprecision(0) << rec.memory_traffic_reduction_pct << "%)";

        std::stringstream sp_str;
        sp_str << std::fixed << std::setprecision(2) << rec.speedup << "x";

        std::cout << std::left 
                  << std::setw(24) << rec.config_name
                  << std::setw(12) << dims.str()
                  << std::setw(14) << std::fixed << std::setprecision(4) << rec.separate_total_time_ms
                  << std::setw(12) << rec.fused_time_ms
                  << std::setw(10) << sp_str.str()
                  << std::setw(18) << saved_bw.str()
                  << std::setw(8)  << (rec.passed ? "PASS" : "FAIL")
                  << "\n";
    }
    std::cout << std::string(98, '-') << "\n\n";

    std::string csv_path = "results/benchmark_results.csv";
    std::string json_path = "results/benchmark_results.json";
    export_benchmark_csv(records, csv_path);
    export_benchmark_json(records, json_path);
}

void run_block_size_analysis(const GpuProperties& gpu) {
    std::cout << "\n>>> [MODE 5] BLOCK SIZE ANALYSIS (64, 128, 256, 512, 1024 THREADS) <<<\n";
    std::cout << "Evaluating fixed matrix dimensions: 2048 rows x 1024 columns\n\n";

    const int rows = 2048;
    const int cols = 1024;
    std::vector<int> block_sizes = {64, 128, 256, 512, 1024};
    std::vector<BenchmarkRecord> records;

    std::cout << std::left 
              << std::setw(14) << "Block Size"
              << std::setw(16) << "Separate (ms)"
              << std::setw(14) << "Fused (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(18) << "Eff Bandwidth"
              << std::setw(10) << "Status"
              << "\n";
    std::cout << std::string(84, '-') << "\n";

    for (int bsz : block_sizes) {
        std::stringstream name;
        name << "BlockSize_" << bsz;
        auto rec = run_single_benchmark(rows, cols, bsz, 10, 100, name.str());
        records.push_back(rec);

        std::stringstream bw;
        bw << std::fixed << std::setprecision(2) << rec.fused_bandwidth_gb_s << " GB/s";

        std::stringstream sp_str;
        sp_str << std::fixed << std::setprecision(2) << rec.speedup << "x";

        std::cout << std::left 
                  << std::setw(14) << bsz
                  << std::setw(16) << std::fixed << std::setprecision(4) << rec.separate_total_time_ms
                  << std::setw(14) << rec.fused_time_ms
                  << std::setw(12) << sp_str.str()
                  << std::setw(18) << bw.str()
                  << std::setw(10) << (rec.passed ? "PASS" : "FAIL")
                  << "\n";
    }
    std::cout << std::string(84, '-') << "\n\n";

    export_benchmark_csv(records, "results/block_size_results.csv");
}

int main(int argc, char* argv[]) {
    GpuProperties gpu = get_gpu_properties(0);

    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--demo") {
            print_banner(gpu);
            run_demonstration_mode(gpu);
            return 0;
        } else if (arg == "--benchmark") {
            print_banner(gpu);
            run_complete_benchmark_suite(gpu);
            return 0;
        } else if (arg == "--block-analysis") {
            print_banner(gpu);
            run_block_size_analysis(gpu);
            return 0;
        }
    }

    int choice = 0;
    while (true) {
        print_banner(gpu);
        std::cout << "CUDA SOFTMAX AND KERNEL FUSION - MAIN MENU\n";
        std::cout << "1. Run Softmax Demonstration\n";
        std::cout << "2. Enter Custom Matrix\n";
        std::cout << "3. Run Complete Benchmark\n";
        std::cout << "4. Run Correctness & Stability Tests\n";
        std::cout << "5. Run Block Size Analysis\n";
        std::cout << "6. Open Visualization Dashboard\n";
        std::cout << "7. Generate Academic Deliverables (PPTX & DOCX)\n";
        std::cout << "8. Exit\n";
        std::cout << "\nEnter selection [1-8]: ";

        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(1000, '\n');
            continue;
        }

        switch (choice) {
            case 1:
                run_demonstration_mode(gpu);
                break;
            case 2:
                run_custom_input_mode(gpu);
                break;
            case 3:
                run_complete_benchmark_suite(gpu);
                break;
            case 4:
                std::cout << "\n[INFO] Launching Correctness and Numerical Stability Suite...\n";
                system("build\\correctness_tests.exe");
                break;
            case 5:
                run_block_size_analysis(gpu);
                break;
            case 6:
                std::cout << "\n[INFO] Launching Local Visualization Dashboard...\n";
                system("scripts\\start_dashboard.bat");
                break;
            case 7:
                std::cout << "\n[INFO] Generating Academic Deliverables (PPTX, DOCX, Charts)...\n";
                system("python scripts\\generate_deliverables.py");
                break;
            case 8:
                std::cout << "\nExiting. Thank you!\n";
                return 0;
            default:
                std::cout << "\nInvalid choice. Please select 1-8.\n";
                break;
        }

        std::cout << "\nPress Enter to return to menu...";
        std::cin.ignore(1000, '\n');
        std::cin.get();
    }

    return 0;
}
