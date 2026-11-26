# Project Context: Softmax and Kernel Fusion Using CUDA

**Academic Laboratory Evaluation — PGC LAB (Theme 10)**  
**Document Version:** 1.0.0  
**Date:** November 2026  
**Status:** In Active Execution  

---

## 1. Project Identification
- **Project Title:** Softmax and Kernel Fusion Using CUDA
- **Evaluation Theme:** Theme 10 — Softmax and Kernel Fusion
- **Evaluation Source:** *CUDA Lab Evaluation Tasks* (Experiment 5 & Experiment 6 Alignment)
- **Course / Lab:** PGC Lab — GPU Computing and High-Performance Parallel Programming
- **Target Audience:** Academic Evaluators, External Examiners, Graduate/Undergraduate CUDA Lab Viva

---

## 2. Academic Problem Statement & Objectives
The Softmax function is an essential non-linear activation and normalization layer widely employed across machine learning architectures, specifically:
1. Final classification layers in convolutional neural networks (CNNs), multi-layer perceptrons (MLPs), and language models.
2. Scaled Dot-Product Attention in Transformer architectures:
   $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

In deep neural networks, Softmax is an inherently **memory-bound operation** with low arithmetic intensity. When computed naively as a chain of distinct operations (find row max, compute exponentials and sum, normalize), traditional GPU execution launches multiple separate kernels. Each kernel reads from and writes to off-chip global memory (High Bandwidth Memory / DRAM), incurring massive memory bus traffic, cache invalidation, and kernel launch overhead.

### Primary Objectives:
1. **Mathematical Rigor & Numerical Stability:** Implement numerically stable row-wise Softmax preventing arithmetic overflow in IEEE 754 32-bit floating-point (FP32) arithmetic.
2. **CPU Reference Implementation:** Develop a clean, sequential C++ reference implementation with double-precision accumulation to serve as the ground truth.
3. **Three Separate CUDA Kernels (Baseline):**
   - Kernel 1: Parallel reduction to find the maximum value of each row.
   - Kernel 2: Parallel reduction to compute the sum of exponentials $\sum \exp(x_i - \max(x))$.
   - Kernel 3: Element-wise normalization dividing adjusted exponentials by the sum.
4. **Fused CUDA Kernel (Optimized):**
   - Execute the entire Softmax computation in a single CUDA kernel.
   - Assign one thread block per matrix row.
   - Retain intermediate row maximums and sums within high-speed on-chip **shared memory** and registers.
   - Perform tree-based parallel reductions with proper `__syncthreads()` barrier synchronizations.
   - Support arbitrary matrix dimensions including non-power-of-two widths and rows exceeding block size via grid/block-strided loops.
5. **Memory Traffic & Performance Benchmarking:**
   - Quantify logical global memory traffic reduction from Kernel Fusion.
   - Measure accurate GPU execution times using `cudaEvent_t` over 10 warmup and 100 measured iterations.
   - Verify speedup across standard benchmark configurations ($1024 \times 256$, $2048 \times 512$, $4096 \times 1024$) and edge cases.
6. **Academic Deliverables:**
   - Interactive local engineering visualization dashboard.
   - Professional PowerPoint presentation (`deliverables/Softmax_and_Kernel_Fusion.pptx`) with exactly 5 checkpoint slides + 1 title slide and full speaker notes.
   - Comprehensive technical explanation Word document (`deliverables/01_Complete_Project_Explanation.docx`) featuring 40+ viva preparation questions.
   - Operation manual (`deliverables/02_How_to_Run_Project.docx`).
   - Rapid morning revision guide (`deliverables/03_Quick_Morning_Revision.docx`).

---

## 3. Mathematical Formulation & Numerical Stability

### 3.1 Standard Mathematical Softmax
Given an input vector $\mathbf{x} = [x_1, x_2, \dots, x_N] \in \mathbb{R}^N$, the standard Softmax function is:
$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}}$$

### 3.2 Numerical Instability in FP32
In standard IEEE 754 single-precision floating-point arithmetic:
- The maximum representable finite number is $\approx 3.4028235 \times 10^{38}$.
- The natural exponential function overflows when $x > 88.7228$:
  $$\exp(88.8) \to +\infty$$
When an overflow occurs, $\sum e^{x_j} \to +\infty$, leading to catastrophic $\frac{\infty}{\infty} \to \text{NaN}$ (Not-a-Number) errors. Conversely, for large negative values ($x < -103.97$), underflow occurs ($\exp(x) \to 0$), and if all elements underflow, the denominator becomes 0, causing division by zero ($0/0 \to \text{NaN}$).

### 3.3 The Safe Softmax Identity
To ensure absolute numerical stability, we subtract the maximum element of the row, $m = \max_{k} x_k$, from each input element:
$$\frac{e^{x_i - m}}{\sum_{j=1}^N e^{x_j - m}} = \frac{e^{x_i} \cdot e^{-m}}{\sum_{j=1}^N (e^{x_j} \cdot e^{-m})} = \frac{e^{-m} \cdot e^{x_i}}{e^{-m} \cdot \sum_{j=1}^N e^{x_j}} = \frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}}$$

Because $x_i - m \le 0$ for all $i \in \{1, \dots, N\}$:
1. The maximum exponent evaluated is $e^0 = 1.0$.
2. Overflow is mathematically impossible.
3. At least one term in the denominator is $e^{m - m} = e^0 = 1.0$, guaranteeing the sum is $\ge 1.0$, completely eliminating division by zero!

---

## 4. Hardware & Software Environment
- **Operating System:** Microsoft Windows 11 Home / Pro (x64)
- **Host CPU:** Intel / AMD multi-core processor
- **GPU Accelerator:** NVIDIA GeForce RTX 3050 Laptop GPU
  - Compute Capability: 8.6 (Ampere Architecture: `sm_86`)
  - Streaming Multiprocessors (SMs): Dedicated Ampere SMs
  - Device Memory: 6144 MB (6 GB) GDDR6
  - Memory Bus Width: 96-bit
  - Theoretical Bandwidth: ~168 - 192 GB/s
- **CUDA Toolkit:** NVIDIA CUDA Toolkit 13.4 (`nvcc` release 13.4, V13.4.92)
- **Host Compiler:** Microsoft Visual C++ (MSVC) 2022 Build Tools (`cl.exe` 14.44.35207)
- **Python Runtime:** Python 3.13.2 with `python-docx`, `python-pptx`, `matplotlib`, `pandas`, `numpy`, `pillow`
