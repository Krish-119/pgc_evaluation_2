#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << err << " (" << cudaGetErrorString(err)   \
                      << ") \"" #call "\"" << std::endl;                      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUDA_CHECK_LAST()                                                     \
    do {                                                                      \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Kernel Launch Error at " << __FILE__ << ":"    \
                      << __LINE__ << " code=" << err << " ("                  \
                      << cudaGetErrorString(err) << ")" << std::endl;         \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

struct GpuProperties {
    std::string name;
    int major;
    int minor;
    int sm_count;
    size_t total_global_mem_bytes;
    int max_threads_per_block;
    size_t shared_mem_per_block;
};

inline GpuProperties get_gpu_properties(int device_id = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    GpuProperties gpu;
    gpu.name = prop.name;
    gpu.major = prop.major;
    gpu.minor = prop.minor;
    gpu.sm_count = prop.multiProcessorCount;
    gpu.total_global_mem_bytes = prop.totalGlobalMem;
    gpu.max_threads_per_block = prop.maxThreadsPerBlock;
    gpu.shared_mem_per_block = prop.sharedMemPerBlock;
    return gpu;
}

inline void print_matrix_sample(const float* mat, int rows, int cols, const std::string& name, int max_r = 4, int max_c = 6) {
    std::cout << "\n--- Matrix: " << name << " (" << rows << " x " << cols << ") ---\n";
    int r_limit = (rows < max_r) ? rows : max_r;
    int c_limit = (cols < max_c) ? cols : max_c;

    for (int r = 0; r < r_limit; ++r) {
        std::cout << "  Row " << std::setw(3) << r << ": [ ";
        for (int c = 0; c < c_limit; ++c) {
            std::cout << std::fixed << std::setprecision(6) << std::setw(10) << mat[r * cols + c] << " ";
        }
        if (cols > c_limit) std::cout << "... ";
        std::cout << "]\n";
    }
    if (rows > r_limit) {
        std::cout << "  ... (" << (rows - r_limit) << " more rows)\n";
    }
    std::cout << std::endl;
}

#endif // CUDA_UTILS_H
