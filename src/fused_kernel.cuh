#ifndef FUSED_KERNEL_CUH
#define FUSED_KERNEL_CUH

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <string>
#include "cuda_utils.h"

namespace fused_kernel {

enum FusedImplementationType {
    SHARED_MEM_CACHED,   // Entire row cached in shared memory (1 input read, 1 output write = 8RC bytes)
    STREAMING_FALLBACK   // Multi-pass streaming fallback when row exceeds shared memory capacity
};

/**
 * @brief Fused Softmax with Full Shared Memory Row Caching (Preferred Implementation).
 * 
 * Cooperatively loads the input row into dynamic shared memory, computes row maximum
 * and sum of exponentials in-place with parallel tree reductions, and writes the
 * normalized output directly to global memory.
 * 
 * Matrix-level logical memory traffic:
 *   - Read X:  R * C * 4 bytes (cooperatively loaded into shared memory ONCE)
 *   - Write Y: R * C * 4 bytes (written directly to global memory ONCE)
 *   - Global intermediate scalars: 0 bytes (row max and sum remain in shared memory)
 *   - Total Logical Traffic = 8 * R * C bytes (8 bytes/element in FP32)
 * 
 * Grid:   gridDim.x = rows (one block per row)
 * Block:  blockDim.x = block_size
 * Shared Memory Layout:
 *   - s_row:    cols * sizeof(float)
 *   - s_reduce: block_size * sizeof(float)
 *   Total Dynamic Shared Memory = (cols + block_size) * sizeof(float)
 */
__global__ void kernel_fused_softmax_cached(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
    extern __shared__ float s_dynamic[];
    float* s_row = s_dynamic;               // Cached row elements (size = cols)
    float* s_reduce = s_dynamic + cols;    // Reduction workspace (size = blockDim.x)

    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // -----------------------------------------------------------------
    // STEP 1: Cooperative Loading of Input Row into Shared Memory
    // -----------------------------------------------------------------
    // Each thread loads elements with block-stride. Read from global memory ONCE.
    for (int c = tid; c < cols; c += bdim) {
        s_row[c] = row_in[c];
    }
    __syncthreads(); // Barrier: Entire row is now resident in fast on-chip shared memory

    // -----------------------------------------------------------------
    // STEP 2: Row Maximum Reduction from Cached Shared Memory
    // -----------------------------------------------------------------
    float thread_max = -FLT_MAX;
    for (int c = tid; c < cols; c += bdim) {
        float val = s_row[c];
        if (val > thread_max) {
            thread_max = val;
        }
    }
    s_reduce[tid] = thread_max;
    __syncthreads();

    // Parallel tree reduction in shared memory workspace
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_reduce[tid + stride] > s_reduce[tid]) {
                s_reduce[tid] = s_reduce[tid + stride];
            }
        }
        __syncthreads();
    }
    float row_max = s_reduce[0];
    __syncthreads(); // Ensure broadcast is visible before s_reduce reuse

    // -----------------------------------------------------------------
    // STEP 3: Shifted Exponentials in-place & Sum Reduction
    // -----------------------------------------------------------------
    float thread_sum = 0.0f;
    for (int c = tid; c < cols; c += bdim) {
        float exp_val = __expf(s_row[c] - row_max);
        s_row[c] = exp_val; // Cache exponential back in shared memory!
        thread_sum += exp_val;
    }
    s_reduce[tid] = thread_sum;
    __syncthreads();

    // Parallel tree reduction for sum
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = s_reduce[0];
    float inv_sum = 1.0f / row_sum;
    __syncthreads(); // Ensure broadcast is visible

    // -----------------------------------------------------------------
    // STEP 4: Normalization and Output Write to Global Memory
    // -----------------------------------------------------------------
    // Reads cached exponentials from shared memory, writes output ONCE.
    for (int c = tid; c < cols; c += bdim) {
        row_out[c] = s_row[c] * inv_sum;
    }
}

/**
 * @brief Fallback Streaming Fused Softmax Kernel.
 * 
 * Used when row length exceeds the device's shared memory capacity.
 * Shared memory is sized only to blockDim.x for reduction workspace.
 * Global memory traffic:
 *   - Read X: Pass 1 (max) + Pass 2 (sum) + Pass 3 (normalize) = 3 * R * C * 4 bytes
 *   - Write Y: 1 * R * C * 4 bytes
 *   - Total Logical Traffic = 16 * R * C bytes
 */
__global__ void kernel_fused_softmax_streaming(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
    extern __shared__ float s_reduce[];
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // Phase 1: Max
    float thread_max = -FLT_MAX;
    for (int c = tid; c < cols; c += bdim) {
        float val = row_in[c];
        if (val > thread_max) thread_max = val;
    }
    s_reduce[tid] = thread_max;
    __syncthreads();

    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_reduce[tid + stride] > s_reduce[tid]) {
                s_reduce[tid] = s_reduce[tid + stride];
            }
        }
        __syncthreads();
    }
    float row_max = s_reduce[0];
    __syncthreads();

    // Phase 2: Exp Sum
    float thread_sum = 0.0f;
    for (int c = tid; c < cols; c += bdim) {
        float val = row_in[c];
        thread_sum += __expf(val - row_max);
    }
    s_reduce[tid] = thread_sum;
    __syncthreads();

    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = s_reduce[0];
    float inv_sum = 1.0f / row_sum;
    __syncthreads();

    // Phase 3: Normalize & Write
    for (int c = tid; c < cols; c += bdim) {
        float val = row_in[c];
        row_out[c] = __expf(val - row_max) * inv_sum;
    }
}

/**
 * @brief Dispatcher for Fused Softmax with Shared Memory Capacity Check.
 */
inline FusedImplementationType select_fused_implementation(int cols, int block_size, size_t max_shared_bytes) {
    size_t needed_bytes = static_cast<size_t>(cols + block_size) * sizeof(float);
    if (needed_bytes <= max_shared_bytes) {
        return SHARED_MEM_CACHED;
    }
    return STREAMING_FALLBACK;
}

/**
 * @brief Executes Fused Kernel with pure start/stop CUDA Event timing.
 */
inline float launch_fused_softmax(
    const float* d_input,
    float* d_output,
    int rows,
    int cols,
    int block_size,
    FusedImplementationType* impl_used = nullptr,
    cudaStream_t stream = 0)
{
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    FusedImplementationType impl = select_fused_implementation(cols, block_size, prop.sharedMemPerBlock);
    if (impl_used) *impl_used = impl;

    dim3 grid(rows);
    dim3 block(block_size);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start, stream));

    if (impl == SHARED_MEM_CACHED) {
        size_t dynamic_bytes = static_cast<size_t>(cols + block_size) * sizeof(float);
        kernel_fused_softmax_cached<<<grid, block, dynamic_bytes, stream>>>(d_input, d_output, rows, cols);
    } else {
        size_t dynamic_bytes = static_cast<size_t>(block_size) * sizeof(float);
        kernel_fused_softmax_streaming<<<grid, block, dynamic_bytes, stream>>>(d_input, d_output, rows, cols);
    }

    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return elapsed_ms;
}

} // namespace fused_kernel

#endif // FUSED_KERNEL_CUH
