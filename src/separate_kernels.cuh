#ifndef SEPARATE_KERNELS_CUH
#define SEPARATE_KERNELS_CUH

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include "cuda_utils.h"

namespace separate_kernels {

/**
 * @brief Kernel 1: Computes the maximum value of each row via parallel reduction.
 * 
 * Grid:   gridDim.x = rows (one block per row)
 * Block:  blockDim.x = threads_per_block
 * Shared: threads_per_block * sizeof(float)
 */
__global__ void kernel_row_max(const float* __restrict__ input, float* __restrict__ row_max_out, int rows, int cols) {
    extern __shared__ float s_data[];
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    float local_max = -FLT_MAX;
    for (int c = tid; c < cols; c += bdim) {
        float val = input[row * cols + c];
        if (val > local_max) {
            local_max = val;
        }
    }
    s_data[tid] = local_max;
    __syncthreads();

    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_data[tid + stride] > s_data[tid]) {
                s_data[tid] = s_data[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        row_max_out[row] = s_data[0];
    }
}

/**
 * @brief Kernel 2: Computes the sum of shifted exponentials for each row.
 * 
 * Grid:   gridDim.x = rows
 * Block:  blockDim.x = threads_per_block
 * Shared: threads_per_block * sizeof(float)
 */
__global__ void kernel_exp_sum(const float* __restrict__ input, const float* __restrict__ row_max_in, float* __restrict__ row_sum_out, int rows, int cols) {
    extern __shared__ float s_data[];
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    float r_max = row_max_in[row];

    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += bdim) {
        float val = input[row * cols + c];
        local_sum += __expf(val - r_max);
    }
    s_data[tid] = local_sum;
    __syncthreads();

    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        row_sum_out[row] = s_data[0];
    }
}

/**
 * @brief Kernel 3: Normalizes the row elements by dividing shifted exponentials by the row sum.
 * 
 * Grid:   gridDim.x = rows
 * Block:  blockDim.x = threads_per_block
 */
__global__ void kernel_normalize(const float* __restrict__ input, const float* __restrict__ row_max_in, const float* __restrict__ row_sum_in, float* __restrict__ output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    float r_max = row_max_in[row];
    float r_sum = row_sum_in[row];
    float inv_sum = 1.0f / r_sum;

    for (int c = tid; c < cols; c += bdim) {
        int idx = row * cols + c;
        float val = input[idx];
        output[idx] = __expf(val - r_max) * inv_sum;
    }
}

struct SeparateTimingDetails {
    float max_kernel_ms;
    float sum_kernel_ms;
    float norm_kernel_ms;
    float total_pipeline_ms;
};

/**
 * @brief Dedicated pipeline execution timing without intermediate event records.
 * Uses exact same timing boundary as the fused kernel (start event -> complete sequence -> stop event).
 */
inline float launch_separate_pipeline_clean(
    const float* d_input,
    float* d_output,
    float* d_row_max,
    float* d_row_sum,
    int rows,
    int cols,
    int block_size,
    cudaStream_t stream = 0)
{
    dim3 grid(rows);
    dim3 block(block_size);
    size_t shared_bytes = block_size * sizeof(float);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    kernel_row_max<<<grid, block, shared_bytes, stream>>>(d_input, d_row_max, rows, cols);
    kernel_exp_sum<<<grid, block, shared_bytes, stream>>>(d_input, d_row_max, d_row_sum, rows, cols);
    kernel_normalize<<<grid, block, 0, stream>>>(d_input, d_row_max, d_row_sum, d_output, rows, cols);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));

    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float pipeline_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&pipeline_ms, ev_start, ev_stop));

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return pipeline_ms;
}

/**
 * @brief Measures individual kernel execution times separately.
 */
inline SeparateTimingDetails measure_separate_individual_kernels(
    const float* d_input,
    float* d_output,
    float* d_row_max,
    float* d_row_sum,
    int rows,
    int cols,
    int block_size,
    cudaStream_t stream = 0)
{
    SeparateTimingDetails timings;
    dim3 grid(rows);
    dim3 block(block_size);
    size_t shared_bytes = block_size * sizeof(float);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Measure K1
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    kernel_row_max<<<grid, block, shared_bytes, stream>>>(d_input, d_row_max, rows, cols);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&timings.max_kernel_ms, ev_start, ev_stop));

    // Measure K2
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    kernel_exp_sum<<<grid, block, shared_bytes, stream>>>(d_input, d_row_max, d_row_sum, rows, cols);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&timings.sum_kernel_ms, ev_start, ev_stop));

    // Measure K3
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    kernel_normalize<<<grid, block, 0, stream>>>(d_input, d_row_max, d_row_sum, d_output, rows, cols);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&timings.norm_kernel_ms, ev_start, ev_stop));

    // Pure pipeline time
    timings.total_pipeline_ms = launch_separate_pipeline_clean(
        d_input, d_output, d_row_max, d_row_sum, rows, cols, block_size, stream);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return timings;
}

} // namespace separate_kernels

#endif // SEPARATE_KERNELS_CUH
