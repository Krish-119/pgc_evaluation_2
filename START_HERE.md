# START HERE — Student Quick Start & Examination Guide

**Project:** Softmax and Kernel Fusion Using CUDA  
**Evaluation Theme:** Theme 10 (CUDA Lab Evaluation Tasks — PGC Lab)  
**Hardware Tested:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere `sm_86`)  
**Overall Readiness:** **100% AUDITED, TESTED, BENCHMARKED & READY FOR EVALUATION**  

---

## 1. Project Readiness Summary
- **Is the project ready?** **YES, ABSOLUTELY.** Every single requirement specified in the evaluation rubric and task statement has been independently built, compiled with native NVCC and MSVC, verified under **NVIDIA Compute Sanitizer** (0 errors across `memcheck`, `racecheck`, `synccheck`, and `initcheck`), benchmarked across 7 configurations, and packaged into professional deliverables.
- **Testing Status:** **12 / 12 Automated Tests PASSED.** Maximum absolute error against double-precision CPU reference is $< 1.49 \times 10^{-8}$ (strictly within documented single-precision floating-point tolerance of $1.0 \times 10^{-5}$). Zero NaN, zero overflow, zero out-of-bounds errors.
- **Genuine GPU Performance:** On the mandatory $4096 \times 1024$ evaluation matrix:
  - Separate Kernels: **0.4448 ms**
  - Fused Kernel: **0.2261 ms** (Dynamic Shared-Memory Cached)
  - Speedup: **1.97x Faster** (49.2% runtime reduction)
  - Estimated Logical Memory Traffic Saved: **33.64 Megabytes** (50.1% reduction: $8RC$ vs $16RC + 20R$ bytes)
  - Effective Bandwidth: **148.39 GB/s** (derived from logical bytes / execution time)

---

## 2. What Has Been Implemented & Audited
1. **CPU Reference Softmax (`src/cpu_softmax.h`):** Numerically stable sequential C++ Softmax with double-precision accumulator for ground truth.
2. **Three Separate CUDA Kernels (`src/separate_kernels.cuh`):**
   - Kernel 1: Row Maximum parallel tree reduction.
   - Kernel 2: Sum of Exponentials parallel tree reduction.
   - Kernel 3: Element-wise Normalization.
   - Individual timing with CUDA Events for each kernel.
3. **Fused CUDA Kernel (`src/fused_kernel.cuh`):**
   - 1 block per row mapping.
   - In-block parallel reductions in fast on-chip shared memory.
   - Synchronized via `__syncthreads()`.
   - Strided processing supporting arbitrary column counts.
   - Zero intermediate global memory arrays.
4. **NVIDIA Compute Sanitizer Verification:**
   - `memcheck`: 0 memory errors, zero out-of-bounds accesses.
   - `racecheck`: 0 hazards, zero race conditions in shared/global memory.
   - `synccheck`: 0 synchronization errors.
   - `initcheck`: 0 uninitialized memory reads.
5. **Automated Testing Suite (`tests/correctness_tests.cu`):**
   - 12 comprehensive edge-case tests including $+1000$ overflow test, $-1000$ underflow test, non-power-of-two ($128 \times 127$), and long rows up to 4096 columns.
6. **Benchmarking Harness (`src/benchmark.h`):**
   - 10 warmup + 100 timed runs using CUDA Events.
   - Exported to `results/benchmark_results.csv`, `results/benchmark_results.json`, and `results/memory_traffic_analysis.csv`.
7. **Block Size Tuning (`results/block_size_results.csv`):**
   - Tested 64, 128, 256, 512, 1024 threads. Peak efficiency identified at **128-256 threads/block** (146.2 GB/s effective bandwidth).
8. **Interactive Local Engineering Dashboard (`dashboard/index.html`):**
   - Responsive web dashboard with educational step-by-step calculator, architecture diagrams, performance graphs, and CSV export.
9. **Academic Presentation & Technical Documentation:**
   - PowerPoint: `deliverables/Softmax_and_Kernel_Fusion.pptx` (6 slides, full speaker notes).
   - Complete Technical Explanation: `deliverables/01_Complete_Project_Explanation.docx` (with 42 Viva Q&As).
   - Operational Manual: `deliverables/02_How_to_Run_Project.docx`.
   - Quick Morning Revision: `deliverables/03_Quick_Morning_Revision.docx`.

---

## 3. Which Document Should You Read First?

Follow this 30-minute reading order:
1. **First (10 mins):** Read `deliverables/03_Quick_Morning_Revision.docx`. It has the 1-minute pitch, the cake baking analogy, the key numbers, and the 10 most critical viva questions.
2. **Second (10 mins):** Open `deliverables/Softmax_and_Kernel_Fusion.pptx` in PowerPoint and read through the 6 slides and their **Speaker Notes** (displayed at the bottom of each slide).
3. **Third (10 mins):** Browse `deliverables/01_Complete_Project_Explanation.docx` Section 18 for the full 42 Viva Voce questions and answers.

---

## 4. How to Launch and Demonstrate the Project

You do not need to remember any complex compiler commands. Everything is automated:

### Method A: Desktop One-Click Launcher
Double-click **`Launch_Softmax_CUDA_Project.bat`** or **`CUDA Softmax Lab.lnk`** directly on your **Windows Desktop**!
Or double-click **`START_PROJECT.bat`** inside this project folder.

An interactive menu will appear:
- Press **`1`** to run the **Mandatory Demonstration** ($2 \times 4$ matrix).
- Press **`3`** to run the **Complete Benchmark Suite** ($1024 \times 256$ to $4096 \times 1024$).
- Press **`4`** to run the **Correctness Test Suite** (12 tests).
- Press **`5`** to run **Block Size Analysis** (64 to 1024 threads).
- Press **`6`** to open the **Visualization Dashboard** in your browser.

### Method B: Direct Command Line Execution
Open PowerShell or Command Prompt in this folder:
```powershell
# 1. Run demonstration mode (shows small matrix & kernel timings)
.\build\softmax_fusion.exe --demo

# 2. Run complete benchmark suite
.\build\softmax_fusion.exe --benchmark

# 3. Run correctness test suite
.\build\correctness_tests.exe

# 4. Start local visualization dashboard
.\scripts\start_dashboard.bat
```

---

## 5. Exact 3-Minute Live Examiner Demonstration Script

When the external examiner asks you to demonstrate your project, follow this exact sequence:

1. **Step 1 (Start Demo):**
   Open terminal and run:
   ```cmd
   build\softmax_fusion.exe --demo
   ```
   *Say to examiner:*
   > "Here is our demonstration mode running on our NVIDIA GeForce RTX 3050 Laptop GPU. For the mandatory 2x4 matrix, Row 0 outputs [0.032059, 0.087144, 0.236883, 0.643914], matching the CPU double-precision reference within floating-point tolerance (< 5.96e-8). You can see our baseline separate pipeline takes 0.051 ms across three kernel launches, while our dynamic shared-memory cached fused kernel executes in just 0.013 ms, achieving a 3.84x speedup."

2. **Step 2 (Show Benchmark & Memory Savings):**
   Run:
   ```cmd
   build\softmax_fusion.exe --benchmark
   ```
   *Say to examiner:*
   > "Here are our measured results over 10 warmup and 100 timed iterations using CUDA Events. On the 4096 by 1024 evaluation matrix, the separate pipeline takes 0.4448 ms, while the fused kernel takes 0.2261 ms—achieving a 1.97x speedup. Furthermore, our algorithmic memory traffic analysis shows that by caching the row in dynamic shared memory and eliminating intermediate global memory round-trips for row maximum and exponential sum, we save 33.64 Megabytes of logical data movement, a 50.1% reduction (from 16RC+20R down to 8RC bytes)."

3. **Step 3 (Show Interactive Dashboard):**
   Double-click `scripts\start_dashboard.bat`.
   *Say to examiner:*
   > "We have also built an interactive engineering laboratory dashboard that visualizes the step-by-step numerical stability transformations, the architectural dataflow difference, and the block size tuning curve where 128-256 threads achieved peak effective bandwidth of 146.2 GB/s."

---

## 6. Where All Files Are Located
- **PowerPoint Presentation:** `deliverables/Softmax_and_Kernel_Fusion.pptx`
- **Complete Explanation Document:** `deliverables/01_Complete_Project_Explanation.docx`
- **How-to-Run Operational Manual:** `deliverables/02_How_to_Run_Project.docx`
- **Morning Quick Revision Guide:** `deliverables/03_Quick_Morning_Revision.docx`
- **High-Resolution Diagrams:** `diagrams/` (7 PNG charts)
- **Mandatory Real Screenshots:** `screenshots/` (12 PNG captures)
- **Benchmark Data Files:** `results/benchmark_results.csv` and `.json`
- **System Hardware Report:** `results/system_information.txt`
- **Master Entry Point:** `START_PROJECT.bat`

Good luck with your evaluation! Everything is thoroughly tested, sanitized, and verified.
