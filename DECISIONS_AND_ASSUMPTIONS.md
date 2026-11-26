# Decisions and Assumptions: Softmax and Kernel Fusion Using CUDA

**Academic Laboratory Evaluation — PGC LAB (Theme 10)**  
**Target Platform:** NVIDIA GeForce RTX 3050 Laptop GPU / CUDA 13.4  
**Date:** November 2026  

---

## 1. Architectural Decisions

### Decision 1: Thread Block Mapping
- **Decision:** In the Fused Kernel, map **one CUDA block to one matrix row** (`gridDim.x = rows`, `blockDim.x = block_size`).
- **Rationale:** 
  1. Softmax is strictly an independent row-wise operation. Elements in row $r$ have zero mathematical dependence on elements in row $r' \neq r$.
  2. One block per row allows using ultra-fast **on-chip shared memory** (`__shared__`) to perform parallel tree reductions across the row without requiring inter-block synchronization across global memory.
  3. Inter-block synchronization in CUDA either requires kernel boundaries (the separate kernel approach) or cooperative groups (which introduces architectural constraints and latency). Assigning 1 block per row is the gold standard design for high-performance Softmax (used in OpenAI Triton, FlashAttention, and TensorRT).

### Decision 2: Handling Arbitrary Matrix Columns (Strided Loop)
- **Decision:** Support matrix column counts $C$ that are less than, equal to, or greater than the block size $B$ (e.g. $B=256$, $C=4096$) using a block-stride loop:
  ```cpp
  for (int c = threadIdx.x; c < cols; c += blockDim.x) { ... }
  ```
- **Rationale:** 
  1. Real-world machine learning models feature varying sequence lengths and vocabulary sizes ($C = 1000, 2048, 4096, 32000$).
  2. If $C > B$, threads iterate across the row with strided memory access, preserving **coalesced memory reads** across adjacent threads in the warp ($c, c+1, c+2, \dots$).
  3. For non-power-of-two dimensions (e.g. $C=127, 1000$), strict boundary checks (`if (c < cols)`) prevent illegal memory access.

### Decision 3: Shared Memory vs Register Caching
- **Decision:** Use **dynamic shared memory** (`kernel_fused_softmax_cached`) to cooperatively load the entire row into `s_row[C]` and maintain a separate workspace `s_reduce[blockDim.x]` for parallel tree reductions whenever `(C + blockDim.x) * sizeof(float) <= sharedMemPerBlock` (48 KB on RTX 3050). For rows up to 4096 columns ($C=4096$ requires $(4096 + 256) \times 4 = 17.4$ KB, well within 48 KB), the input row is loaded from DRAM exactly ONCE, transformed into shifted exponentials in-place, and written out to global memory once ($8RC$ logical bytes, a 50.0% traffic reduction). For extreme rows that exceed shared memory limits, an automatic fallback streaming kernel (`kernel_fused_softmax_streaming`) is dispatched and clearly documented.
- **Rationale:** An RTX 3050 SM has 48 KB of configurable shared memory per block. Cooperatively caching the row on-chip completely eliminates redundant DRAM reads and eliminates intermediate row maximum and sum memory round-trips!

### Decision 4: Baseline Separate Kernels Design
- **Decision:** Implement exactly 3 distinct computational kernels:
  1. `kernel_row_max`: calculates and outputs `d_row_max[r]` in global memory.
  2. `kernel_exp_sum`: reads `d_row_max[r]`, calculates $\exp(X[r, c] - \text{row\_max})$, performs reduction, outputs `d_row_sum[r]` in global memory.
  3. `kernel_normalize`: reads `d_row_max[r]`, `d_row_sum[r]`, recomputes $\exp(X[r, c] - \text{row\_max}) / \text{row\_sum}$, and writes to $Y[r, c]$.
- **Rationale:** Recomputing the exponential in Kernel 3 is the standard baseline textbook approach because storing an intermediate exponential matrix $R \times C \times 4$ bytes would consume massive extra global memory bandwidth and VRAM allocation ($4096 \times 1024 \times 4 \approx 16.8$ MB). Clearly documenting this design accurately reflects both compute and memory trade-offs.

---

## 2. Mathematical Stability Decisions

### Decision 5: Max-Subtraction for Numerical Stability
- **Decision:** Every implementation (CPU, Separate CUDA, Fused CUDA) strictly enforces $x'_i = x_i - \max_k(x_k)$ before exponentiation.
- **Rationale:** 
  In standard FP32, $e^x$ overflows to `+inf` when $x > 88.72$. Deep neural network logits frequently reach $100$ or higher during early training or adversarial inputs. By subtracting the row maximum, the largest exponent is $e^0 = 1.0$, completely precluding overflow. Furthermore, the sum is guaranteed to be at least $1.0$, precluding division by zero.

### Decision 6: Verification Tolerances
- **Decision:** 
  - Absolute error threshold: $\epsilon_{\text{abs}} = 1.0 \times 10^{-5}$.
  - Relative error threshold: $\epsilon_{\text{rel}} = 1.0 \times 10^{-4}$ for floating-point accumulation variances.
  - Row-sum criterion: $|\sum Y[r, :] - 1.0| < 1.0 \times 10^{-5}$.
- **Rationale:** 
  Floating-point addition is non-associative: $(a + b) + c \neq a + (b + c)$. The sequential CPU code accumulates linearly, whereas the CUDA kernels accumulate via tree reductions in parallel. Tiny differences at the 6th or 7th decimal place are expected and mathematically natural. An absolute tolerance of $1.0 \times 10^{-5}$ proves numerical correctness.

---

## 3. Benchmarking & Measurement Methodology

### Decision 7: Timing Mechanism
- **Decision:** Use high-resolution CUDA Events (`cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`) for all GPU kernel timing. Use `std::chrono::high_resolution_clock` for CPU timing.
- **Rationale:** Host-side CPU clocks cannot accurately measure asynchronous GPU kernel execution without heavy host-device synchronization overhead. CUDA Events provide sub-microsecond timestamping on the GPU timeline.

### Decision 8: Warmup & Iteration Counts
- **Decision:** 10 warmup iterations followed by 100 timed iterations. Report the arithmetic mean and standard deviation.
- **Rationale:** 
  1. The initial kernel launch incurs driver setup, page table warming, and GPU clock frequency ramping (P-state transitions).
  2. 100 iterations provide statistically robust averages and filter out OS scheduling jitter.

### Decision 9: Data Integrity Guarantee
- **Decision:** All numbers reported in CSV, JSON, the visualization dashboard, the PowerPoint presentation, and the Word documents are derived strictly from genuine executions on the host hardware. Zero numbers or screenshots will be fabricated.
