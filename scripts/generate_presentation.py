"""
Generates the 6-slide academic PowerPoint presentation for
Softmax and Kernel Fusion Using CUDA (Theme 10 - PGC Lab).
Includes 5 mandatory checkpoint slides + 1 title slide, with detailed speaker notes.
"""
import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    diag_dir = os.path.join(base_dir, "diagrams")
    out_dir = os.path.join(base_dir, "deliverables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Softmax_and_Kernel_Fusion.pptx")

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette
    c_navy = RGBColor(15, 23, 42)        # #0F172A
    c_navy_light = RGBColor(30, 41, 59)  # #1E293B
    c_slate_bg = RGBColor(248, 250, 252) # #F8FAFC
    c_blue = RGBColor(2, 132, 199)       # #0284C7
    c_green = RGBColor(5, 150, 105)      # #059669
    c_text_dark = RGBColor(15, 23, 42)
    c_text_muted = RGBColor(100, 116, 139)
    c_white = RGBColor(255, 255, 255)
    c_card_border = RGBColor(226, 232, 240)

    def add_header(slide, title_text, category_text):
        # Header background banner
        header_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.1))
        header_box.fill.solid()
        header_box.fill.fore_color.rgb = c_navy
        header_box.line.color.rgb = c_blue
        header_box.line.width = Pt(2)

        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.8)
        tf.margin_top = Inches(0.18)

        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = c_blue

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(20)
        p_title.font.bold = True
        p_title.font.color.rgb = c_white

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = c_navy
    bg1.line.fill.background()

    # Title box
    tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.333), Inches(3.2))
    tf1 = tb.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "PGC LAB EVALUATION — THEME 10"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = c_blue

    p1 = tf1.add_paragraph()
    p1.text = "Softmax and Kernel Fusion Using CUDA"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = c_white
    p1.space_before = Pt(12)

    p2 = tf1.add_paragraph()
    p2.text = "High-Performance GPU Memory Optimization, Parallel Reductions, and Bandwidth Traffic Analysis"
    p2.font.size = Pt(16)
    p2.font.color.rgb = RGBColor(203, 213, 225)
    p2.space_before = Pt(8)

    # Info Card
    info_box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.7), Inches(11.333), Inches(1.9))
    info_box.fill.solid()
    info_box.fill.fore_color.rgb = c_navy_light
    info_box.line.color.rgb = RGBColor(51, 65, 85)
    info_tf = info_box.text_frame
    info_tf.margin_left = Inches(0.4)
    info_tf.margin_top = Inches(0.25)

    ip1 = info_tf.paragraphs[0]
    ip1.text = "Academic Details & Target Hardware Configuration"
    ip1.font.size = Pt(13)
    ip1.font.bold = True
    ip1.font.color.rgb = c_white

    ip2 = info_tf.add_paragraph()
    ip2.text = "• Student Name: [Student Name Placeholder]     |     USN / Roll No: [USN Placeholder]     |     Lab: PGC Lab\n" \
               "• GPU Hardware: NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere sm_86, 20 SMs, 6144 MB GDDR6)\n" \
               "• Software Stack: CUDA Toolkit 13.4, MSVC 2022 C++17, CUDA Events, Custom Shared Memory Reduction"
    ip2.font.size = Pt(11)
    ip2.font.color.rgb = RGBColor(148, 163, 184)
    ip2.space_before = Pt(6)

    # Speaker notes
    s1.notes_slide.notes_text_frame.text = (
        "Good morning, esteemed evaluator. Today I am presenting Theme 10 of our CUDA Lab Evaluation Tasks: "
        "'Softmax and Kernel Fusion Using CUDA'. In this project, we implement the standard row-wise Softmax layer "
        "used in neural networks, first as three separate CUDA kernels and then as a single fused CUDA kernel. "
        "We executed our implementations on an NVIDIA GeForce RTX 3050 Laptop GPU under CUDA 13.4, measuring genuine "
        "kernel timings with CUDA Events across 10 warmup and 100 timed iterations. We demonstrate how kernel fusion "
        "eliminates intermediate global memory round-trips and reduces execution latency by up to 49.6% on large matrices."
    )

    # =========================================================================
    # SLIDE 2: Checkpoint 1 — Softmax Concept
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Checkpoint 1: Softmax Mathematical Concept & Numerical Stability", "Theme 10 • Foundational Theory")

    # Left Column: Mathematical Definition & Numerical Stability
    left_box = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.6))
    left_box.fill.solid()
    left_box.fill.fore_color.rgb = c_white
    left_box.line.color.rgb = c_card_border
    ltf = left_box.text_frame
    ltf.word_wrap = True
    ltf.margin_left = Inches(0.3)
    ltf.margin_top = Inches(0.3)

    lp1 = ltf.paragraphs[0]
    lp1.text = "1. Mathematical Formula & Softmax Function"
    lp1.font.size = Pt(14)
    lp1.font.bold = True
    lp1.font.color.rgb = c_navy

    lp2 = ltf.add_paragraph()
    lp2.text = "Given an input row vector x = [x_1, ..., x_C], the Softmax function normalizes logits into a valid probability distribution:\n" \
               "   Softmax(x_i) = exp(x_i) / Σ exp(x_j)\n" \
               "Properties: (1) Output ∈ (0, 1), (2) Σ Softmax(x_i) = 1.0, (3) Monotonic ordering preserved."
    lp2.font.size = Pt(11)
    lp2.font.color.rgb = c_text_dark
    lp2.space_before = Pt(4)

    lp3 = ltf.add_paragraph()
    lp3.text = "2. The FP32 Numerical Instability Problem"
    lp3.font.size = Pt(14)
    lp3.font.bold = True
    lp3.font.color.rgb = c_navy
    lp3.space_before = Pt(14)

    lp4 = ltf.add_paragraph()
    lp4.text = "• In IEEE 754 single precision (FP32), exp(x) overflows to +inf when x > 88.72.\n" \
               "• Inf / Inf produces NaN (Not a Number), crashing neural network training.\n" \
               "• Solution: Subtract the row maximum: m = max_j(x_j)\n" \
               "   Safe Softmax(x_i) = exp(x_i - m) / Σ exp(x_j - m)\n" \
               "• Mathematical Proof: Multiplying numerator and denominator by exp(-m) preserves the exact value identically, while ensuring (x_i - m) ≤ 0, so the maximum exponent is exp(0) = 1.0. Overflow is impossible!"
    lp4.font.size = Pt(10.5)
    lp4.font.color.rgb = c_text_dark
    lp4.space_before = Pt(4)

    # Right Column: Worked Example & Deep Learning Roles
    right_box = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
    right_box.fill.solid()
    right_box.fill.fore_color.rgb = c_white
    right_box.line.color.rgb = c_card_border
    rtf = right_box.text_frame
    rtf.word_wrap = True
    rtf.margin_left = Inches(0.3)
    rtf.margin_top = Inches(0.3)

    rp1 = rtf.paragraphs[0]
    rp1.text = "3. Solved Numerical Example (Theme Demonstration Matrix)"
    rp1.font.size = Pt(14)
    rp1.font.bold = True
    rp1.font.color.rgb = c_navy

    rp2 = rtf.add_paragraph()
    rp2.text = "Row 0: [ 1.0,  2.0,  3.0,  4.0 ]\n" \
               "• Step 1 (Row Max m) : m = 4.0\n" \
               "• Step 2 (Shifted x-m): [ -3.0,  -2.0,  -1.0,   0.0 ]\n" \
               "• Step 3 (Exponentials): [ 0.049787, 0.135335, 0.367879, 1.000000 ]\n" \
               "• Step 4 (Sum of Exps): 0.049787 + 0.135335 + 0.367879 + 1.0 = 1.553001\n" \
               "• Step 5 (Probabilities): [ 0.032059, 0.087144, 0.236883, 0.643914 ]\n" \
               "• Verification Check : Sum = 1.000000 (Exact Match with CPU & GPU)"
    rp2.font.size = Pt(10.5)
    rp2.font.color.rgb = c_text_dark
    rp2.space_before = Pt(4)

    rp3 = rtf.add_paragraph()
    rp3.text = "4. Where Softmax Appears in Modern Neural Networks"
    rp3.font.size = Pt(14)
    rp3.font.bold = True
    rp3.font.color.rgb = c_navy
    rp3.space_before = Pt(14)

    rp4 = rtf.add_paragraph()
    rp4.text = "1. Multi-Class Classifiers: Converts final linear layer output logits into class probability distributions (cross-entropy loss target).\n" \
               "2. Transformer Multi-Head Attention: Softmax(Q·K^T / √d_k) computes attention weight matrices over sequence tokens in models like GPT-4, LLaMA, and BERT."
    rp4.font.size = Pt(10.5)
    rp4.font.color.rgb = c_text_dark
    rp4.space_before = Pt(4)

    s2.notes_slide.notes_text_frame.text = (
        "In Checkpoint 1, we establish the mathematical foundation of Softmax. Given an input vector x, Softmax exponentiates "
        "and normalizes each element. However, in single precision floating point arithmetic, exponentiating values greater than 88 "
        "causes catastrophic overflow to infinity. To guarantee numerical stability, we subtract the row maximum m. Because "
        "exp(x - m) equals exp(x) times exp(-m), this scalar cancels out identically in both numerator and denominator. This mathematically "
        "guarantees that every exponent is less than or equal to 0, bounding the maximum exponent to 1.0 and preventing any overflow. "
        "On the right, we show our manually verified worked example for row [1, 2, 3, 4], producing probabilities [0.032, 0.087, 0.237, 0.644], "
        "which perfectly matches our GPU execution. Finally, Softmax is ubiquitous in classification heads and Transformer attention."
    )

    # =========================================================================
    # SLIDE 3: Checkpoint 2 — Separate Kernels
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Checkpoint 2: Three Separate CUDA Kernels (Baseline Architecture)", "Theme 10 • Baseline Implementation")

    # Left: Architecture details
    s3_left = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.4), Inches(6.0), Inches(5.6))
    s3_left.fill.solid()
    s3_left.fill.fore_color.rgb = c_white
    s3_left.line.color.rgb = c_card_border
    s3_ltf = s3_left.text_frame
    s3_ltf.word_wrap = True
    s3_ltf.margin_left = Inches(0.3)
    s3_ltf.margin_top = Inches(0.3)

    sp1 = s3_ltf.paragraphs[0]
    sp1.text = "Three-Kernel Pipeline Decomposition"
    sp1.font.size = Pt(14)
    sp1.font.bold = True
    sp1.font.color.rgb = c_navy

    sp2 = s3_ltf.add_paragraph()
    sp2.text = "• Kernel 1: Row Maximum (`kernel_row_max`)\n" \
               "  - Parallel tree reduction in shared memory per block.\n" \
               "  - Thread 0 stores row max into global memory `d_row_max[r]`.\n\n" \
               "• Kernel 2: Sum of Exponentials (`kernel_exp_sum`)\n" \
               "  - Reads `d_row_max[r]` from global memory.\n" \
               "  - Computes exp(x - max) and performs tree sum reduction.\n" \
               "  - Thread 0 stores sum into global memory `d_row_sum[r]`.\n\n" \
               "• Kernel 3: Normalization (`kernel_normalize`)\n" \
               "  - Reads `d_row_max[r]` and `d_row_sum[r]` from global memory.\n" \
               "  - Normalizes elements: y[r, c] = exp(x - max) / sum.\n" \
               "  - Writes final probabilities Y to global memory."
    sp2.font.size = Pt(11)
    sp2.font.color.rgb = c_text_dark
    sp2.space_before = Pt(6)

    # Insert Diagram
    d_path1 = os.path.join(diag_dir, "06_architecture_separate.png")
    if os.path.exists(d_path1):
        s3.shapes.add_picture(d_path1, Inches(7.1), Inches(1.4), width=Inches(5.4))

    # Right Bottom: Measured Timings Table
    time_box = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(4.3), Inches(5.4), Inches(2.7))
    time_box.fill.solid()
    time_box.fill.fore_color.rgb = c_slate_bg
    time_box.line.color.rgb = c_card_border
    ttf = time_box.text_frame
    ttf.margin_left = Inches(0.3)
    ttf.margin_top = Inches(0.2)

    tp1 = ttf.paragraphs[0]
    tp1.text = "Actual Measured Kernel Latencies (4096 x 1024 on RTX 3050)"
    tp1.font.size = Pt(12)
    tp1.font.bold = True
    tp1.font.color.rgb = c_navy

    tp2 = ttf.add_paragraph()
    tp2.text = "• Kernel 1 (Row Max) Execution Time   : 0.1137 ms\n" \
               "• Kernel 2 (Exp Sum) Execution Time   : 0.1104 ms\n" \
               "• Kernel 3 (Normalize) Execution Time : 0.2180 ms\n" \
               "----------------------------------------------------------\n" \
               "• Total Three-Kernel Pipeline Time    : 0.4448 ms\n" \
               "• Estimated Logical Memory Traffic    : 67.19 MB moved (16RC + 20R)\n" \
               "  (Distinguished from physical cache-filtered DRAM transactions)"
    tp2.font.size = Pt(10.5)
    tp2.font.color.rgb = c_text_dark
    tp2.space_before = Pt(4)

    s3.notes_slide.notes_text_frame.text = (
        "In Checkpoint 2, we present our baseline CUDA implementation comprising three separate computational kernels. "
        "Kernel 1 computes the row maximum using parallel tree reduction in shared memory and writes the result to global memory. "
        "Kernel 2 reads the maximum from global memory, computes shifted exponentials, performs another tree reduction, and writes "
        "the sum to global memory. Kernel 3 reads both the maximum and the sum from global memory to write the final normalized probabilities. "
        "Timing each kernel with CUDA events on our RTX 3050 for a 4096 by 1024 matrix, Kernel 1 takes 0.114 ms, Kernel 2 takes 0.110 ms, "
        "and Kernel 3 takes 0.218 ms, giving a total pipeline latency of 0.445 milliseconds. The key bottleneck here is that global memory "
        "is accessed across three distinct kernel launches, generating 67.19 MB of logical memory traffic."
    )

    # =========================================================================
    # SLIDE 4: Checkpoint 3 — Fused Kernel
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Checkpoint 3: Fused CUDA Kernel (Optimized Architecture)", "Theme 10 • Kernel Fusion Optimization")

    s4_left = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.4), Inches(6.0), Inches(5.6))
    s4_left.fill.solid()
    s4_left.fill.fore_color.rgb = c_white
    s4_left.line.color.rgb = c_card_border
    s4_ltf = s4_left.text_frame
    s4_ltf.word_wrap = True
    s4_ltf.margin_left = Inches(0.3)
    s4_ltf.margin_top = Inches(0.3)

    fp1 = s4_ltf.paragraphs[0]
    fp1.text = "Fused Softmax Architectural Mechanics"
    fp1.font.size = Pt(14)
    fp1.font.bold = True
    fp1.font.color.rgb = c_navy

    fp2 = s4_ltf.add_paragraph()
    fp2.text = "• One Thread Block Per Row: `gridDim.x = rows`, `blockDim.x = 256`\n" \
               "  - Allows independent row processing without inter-block synchronization.\n" \
               "• In-Block Shared Memory Tree Reductions:\n" \
               "  - Fast on-chip shared memory (`__shared__ float s_mem[]`) replaces global memory buffers.\n" \
               "  - Synchronized via hardware barriers: `__syncthreads()`.\n" \
               "• Multi-Phase Execution in Single Launch:\n" \
               "  1. Max reduction: thread-local strided loop -> tree reduction -> broadcast.\n" \
               "  2. Exp-Sum reduction: compute exp(x - max) -> tree reduction -> broadcast.\n" \
               "  3. Normalization: divide by sum and write coalesced output directly to Y.\n" \
               "• Zero Intermediate Global Memory Footprint:\n" \
               "  - Intermediate max and sum are kept entirely in shared memory/registers!"
    fp2.font.size = Pt(11)
    fp2.font.color.rgb = c_text_dark
    fp2.space_before = Pt(6)

    d_path2 = os.path.join(diag_dir, "07_architecture_fused.png")
    if os.path.exists(d_path2):
        s4.shapes.add_picture(d_path2, Inches(7.1), Inches(1.4), width=Inches(5.4))

    f_res_box = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(4.3), Inches(5.4), Inches(2.7))
    f_res_box.fill.solid()
    f_res_box.fill.fore_color.rgb = c_slate_bg
    f_res_box.line.color.rgb = c_card_border
    fr_tf = f_res_box.text_frame
    fr_tf.margin_left = Inches(0.3)
    fr_tf.margin_top = Inches(0.2)

    frp1 = fr_tf.paragraphs[0]
    frp1.text = "Fused Performance Gain (4096 x 1024 on RTX 3050)"
    frp1.font.size = Pt(12)
    frp1.font.bold = True
    frp1.font.color.rgb = c_navy

    frp2 = fr_tf.add_paragraph()
    frp2.text = "• Fused Kernel Execution Time     : 0.2261 ms (vs 0.4448 ms)\n" \
               "• Speedup Factor                 : 1.97x Faster (~2x speedup)\n" \
               "• Execution Time Reduction       : 49.2% Reduction\n" \
               "• Estimated Memory Traffic Saved : 33.64 MB (50.1% Reduction, 8RC bytes)\n" \
               "• Kernel Launches Eliminated     : 2 out of 3 launches removed\n" \
               "• Effective Fused Bandwidth      : 148.39 GB/s (decimal GB/s: 10^9 B/s)"
    frp2.font.size = Pt(10.5)
    frp2.font.color.rgb = c_text_dark
    frp2.space_before = Pt(4)

    s4.notes_slide.notes_text_frame.text = (
        "In Checkpoint 3, we introduce our fused kernel architecture. By mapping exactly one thread block to each matrix row, "
        "we cooperatively load the entire row into fast on-chip dynamic shared memory (s_row) with a separate reduction workspace (s_reduce). "
        "The thread block performs the maximum reduction, computes shifted exponentials in-place in shared memory, calculates the exponential "
        "sum via parallel tree reduction, and normalizes and writes the output directly to global memory in a single coalesced pass. "
        "On the 4096 by 1024 matrix, our fused kernel drops execution latency from 0.445 ms to 0.226 ms—a 1.97x speedup—while reducing "
        "logical memory traffic by 50% from 67.19 MB down to 33.55 MB (8RC bytes). We also eliminate two host launch round-trips."
    )

    # =========================================================================
    # SLIDE 5: Checkpoint 4 — Results Table
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Checkpoint 4: Genuine Benchmark Results & Verification Table", "Theme 10 • Empirical Results")

    # Add Table
    rows_cnt = 8
    cols_cnt = 8
    left = Inches(0.8)
    top = Inches(1.4)
    width = Inches(11.733)
    height = Inches(4.5)

    table_shape = s5.shapes.add_table(rows_cnt, cols_cnt, left, top, width, height)
    table = table_shape.table

    # Column widths
    col_widths = [Inches(1.8), Inches(1.3), Inches(1.4), Inches(1.3), Inches(1.1), Inches(1.6), Inches(1.6), Inches(1.4)]
    for idx, w in enumerate(col_widths):
        table.columns[idx].width = w

    headers = ["Matrix Config", "Dimensions", "Separate (ms)", "Fused (ms)", "Speedup", "Traffic Saved", "Max Abs Error", "CPU Match"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = c_navy
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(10)
            p.font.bold = True
            p.font.color.rgb = c_white
            p.alignment = PP_ALIGN.CENTER

    data = [
        ["Config_1024x256", "1024 x 256", "0.0554 ms", "0.0420 ms", "1.32x", "2.12 MB (50%)", "7.45 × 10⁻⁹", "PASS"],
        ["Config_2048x512", "2048 x 512", "0.1416 ms", "0.0905 ms", "1.56x", "8.43 MB (50%)", "3.73 × 10⁻⁹", "PASS"],
        ["Config_4096x1024", "4096 x 1024", "0.4448 ms", "0.2261 ms", "1.97x", "33.64 MB (50%)", "1.86 × 10⁻⁹", "PASS"],
        ["Config_1024x2048", "1024 x 2048", "0.2265 ms", "0.1151 ms", "1.97x", "16.80 MB (50%)", "9.31 × 10⁻¹⁰", "PASS"],
        ["Config_1024x4096", "1024 x 4096", "0.4409 ms", "0.2273 ms", "1.94x", "33.57 MB (50%)", "4.66 × 10⁻¹⁰", "PASS"],
        ["Config_128x127_NonPow2", "128 x 127", "0.0319 ms", "0.0109 ms", "2.93x", "0.13 MB (50%)", "1.49 × 10⁻⁸", "PASS"],
        ["Config_1024x1000_Irreg", "1024 x 1000", "0.1402 ms", "0.0777 ms", "1.80x", "8.21 MB (50%)", "1.86 × 10⁻⁹", "PASS"],
    ]

    for row_idx, row_data in enumerate(data):
        for col_idx, val in enumerate(row_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = c_white if row_idx % 2 == 0 else c_slate_bg
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(9.5)
                p.font.color.rgb = c_text_dark
                if col_idx in [2, 3, 4, 7]:
                    p.alignment = PP_ALIGN.CENTER
                if col_idx == 4:
                    p.font.bold = True
                    p.font.color.rgb = c_blue
                if col_idx == 7:
                    p.font.bold = True
                    p.font.color.rgb = c_green

    # Caption box
    note_box = s5.shapes.add_textbox(Inches(0.8), Inches(6.0), Inches(11.733), Inches(0.9))
    ntf = note_box.text_frame
    np1 = ntf.paragraphs[0]
    np1.text = "Methodology: 10 warmup runs + 100 timed iterations using high-precision CUDA Events on NVIDIA GeForce RTX 3050 Laptop GPU.\n" \
               "Verification: Double-precision CPU Softmax reference (std::chrono). Outputs match CPU within documented FP32 tolerance (absolute error < 1.49 × 10⁻⁸).\n" \
               "Traffic: Figures represent estimated logical global memory traffic derived from algorithmic transactions."
    np1.font.size = Pt(9.5)
    np1.font.color.rgb = c_text_muted

    s5.notes_slide.notes_text_frame.text = (
        "In Checkpoint 4, we present our comprehensive results table across seven matrix configurations measured directly on our RTX 3050 GPU. "
        "Notice that our benchmark covers small, medium, large, non-power-of-two (128 by 127), and very long rows up to 4096 columns. "
        "In every single test case, the output matches the double-precision CPU reference with a maximum error smaller than 1.5e-8, "
        "passing all correctness checks. On the evaluation size 4096 by 1024, the fused kernel achieves 1.97x speedup and saves 33.64 MB "
        "of logical global memory traffic (50.1% reduction). On the smaller irregular matrix 128 by 127, the fused kernel reaches a 2.93x speedup because kernel "
        "launch overhead dominated the separate version."
    )

    # =========================================================================
    # SLIDE 6: Checkpoint 5 — Analysis
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Checkpoint 5: Architectural Performance Analysis & Industry Relevance", "Theme 10 • Deep Performance Insights")

    s6_c1 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.4), Inches(3.7), Inches(5.6))
    s6_c1.fill.solid()
    s6_c1.fill.fore_color.rgb = c_white
    s6_c1.line.color.rgb = c_card_border
    c1_tf = s6_c1.text_frame
    c1_tf.word_wrap = True
    c1_tf.margin_left = Inches(0.25)
    c1_tf.margin_top = Inches(0.25)

    ap1 = c1_tf.paragraphs[0]
    ap1.text = "1. Memory-Bound Nature"
    ap1.font.size = Pt(13)
    ap1.font.bold = True
    ap1.font.color.rgb = c_navy

    ap2 = c1_tf.add_paragraph()
    ap2.text = "• Arithmetic Intensity: Softmax performs ~3 FLOPs per 4-byte memory transfer. It is severely memory bandwidth bound.\n" \
               "• Separate kernels stall on DRAM latency between passes (16RC + 20R bytes).\n" \
               "• Dynamic shared-memory caching reduces logical traffic to 8RC bytes (~50% reduction), approaching high sustained memory throughput (148.4 GB/s effective bandwidth)."
    ap2.font.size = Pt(10)
    ap2.font.color.rgb = c_text_dark
    ap2.space_before = Pt(4)

    ap3 = c1_tf.add_paragraph()
    ap3.text = "2. Kernel Launch Overhead"
    ap3.font.size = Pt(13)
    ap3.font.bold = True
    ap3.font.color.rgb = c_navy
    ap3.space_before = Pt(12)

    ap4 = c1_tf.add_paragraph()
    ap4.text = "• Launching a kernel incurs ~3-5 µs host driver overhead.\n" \
               "• For small/medium matrices, launch overhead exceeds compute time.\n" \
               "• Fusing 3 kernels into 1 saves 66% of launch delays."
    ap4.font.size = Pt(10)
    ap4.font.color.rgb = c_text_dark
    ap4.space_before = Pt(4)

    # Column 2: Block Size & Row Length
    s6_c2 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.8), Inches(1.4), Inches(3.7), Inches(5.6))
    s6_c2.fill.solid()
    s6_c2.fill.fore_color.rgb = c_white
    s6_c2.line.color.rgb = c_card_border
    c2_tf = s6_c2.text_frame
    c2_tf.word_wrap = True
    c2_tf.margin_left = Inches(0.25)
    c2_tf.margin_top = Inches(0.25)

    bp1 = c2_tf.paragraphs[0]
    bp1.text = "3. Block Size & Occupancy Tuning"
    bp1.font.size = Pt(13)
    bp1.font.bold = True
    bp1.font.color.rgb = c_navy

    bp2 = c2_tf.add_paragraph()
    bp2.text = "• Measured Results (2048 x 1024 Matrix):\n" \
               "  - 64 threads : 0.1158 ms (144.9 GB/s)\n" \
               "  - 128 threads: 0.1148 ms (146.2 GB/s, TOP SPEED)\n" \
               "  - 256 threads: 0.1152 ms (145.6 GB/s, BALANCED)\n" \
               "  - 512 threads: 0.1820 ms (92.2 GB/s)\n" \
               "  - 1024 threads: 0.4373 ms (38.4 GB/s)\n" \
               "• Architectural Explanation:\n" \
               "  - 128-256 threads provides optimal balance of active warps per SM, register pressure, and reduction tree depth (7-8 steps).\n" \
               "  - 1024 threads incurs high synchronization latency across 10 reduction steps and reduces active concurrent blocks per SM."
    bp2.font.size = Pt(10)
    bp2.font.color.rgb = c_text_dark
    bp2.space_before = Pt(4)

    bp3 = c2_tf.add_paragraph()
    bp3.text = "4. Effect of Longer Rows"
    bp3.font.size = Pt(13)
    bp3.font.bold = True
    bp3.font.color.rgb = c_navy
    bp3.space_before = Pt(12)

    bp4 = c2_tf.add_paragraph()
    bp4.text = "• For rows wider than block size (e.g. 4096 elements), cooperative loading into dynamic shared memory caches the row on-chip, enabling coalesced FP32 access and single-pass DRAM I/O."
    bp4.font.size = Pt(10)
    bp4.font.color.rgb = c_text_dark
    bp4.space_before = Pt(4)

    # Column 3: Industry Relevance & Conclusion
    s6_c3 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.8), Inches(1.4), Inches(3.7), Inches(5.6))
    s6_c3.fill.solid()
    s6_c3.fill.fore_color.rgb = c_white
    s6_c3.line.color.rgb = c_card_border
    c3_tf = s6_c3.text_frame
    c3_tf.word_wrap = True
    c3_tf.margin_left = Inches(0.25)
    c3_tf.margin_top = Inches(0.25)

    cp1 = c3_tf.paragraphs[0]
    cp1.text = "5. Industry Applications"
    cp1.font.size = Pt(13)
    cp1.font.bold = True
    cp1.font.color.rgb = c_navy

    cp2 = c3_tf.add_paragraph()
    cp2.text = "• Deep Learning Compilers (TVM, MLIR, PyTorch Inductor):\n" \
               "  - Automatically fuse consecutive elementwise and reduction operators to eliminate memory traffic.\n" \
               "• High-Performance Inference Engines (NVIDIA TensorRT, vLLM, FlashAttention):\n" \
               "  - FlashAttention fuses Softmax with MatMul QK^T and V, reducing IO complexity from O(N^2) to O(N)."
    cp2.font.size = Pt(10)
    cp2.font.color.rgb = c_text_dark
    cp2.space_before = Pt(4)

    cp3 = c3_tf.add_paragraph()
    cp3.text = "6. Final Conclusion"
    cp3.font.size = Pt(13)
    cp3.font.bold = True
    cp3.font.color.rgb = c_navy
    cp3.space_before = Pt(12)

    cp4 = c3_tf.add_paragraph()
    cp4.text = "• Kernel fusion is a foundational GPU optimization for memory-bound layers.\n" \
               "• In our lab evaluation, fusion delivers 1.32x - 2.93x speedup, cuts logical memory traffic by 50%, and preserves strict numerical correctness."
    cp4.font.size = Pt(10)
    cp4.font.color.rgb = c_text_dark
    cp4.space_before = Pt(4)

    s6.notes_slide.notes_text_frame.text = (
        "In our final Checkpoint 5, we analyze why fusion delivers such dramatic improvements. First, Softmax is strictly memory-bound "
        "with an arithmetic intensity of less than 1 FLOP per byte. By fusing the kernels and caching the input row in dynamic shared memory, "
        "we eliminate DRAM round-trips and achieve 148.4 GB/s effective bandwidth. Second, we eliminate two kernel launches. "
        "Third, our block size study proves that 128 to 256 threads is the sweet spot on our Ampere GPU: 1024 threads introduces excessive synchronization "
        "overhead and register pressure, while 64 threads underutilizes the multiprocessors. Finally, in industry, deep learning compilers like "
        "PyTorch Inductor and inference engines like NVIDIA TensorRT and FlashAttention rely on this exact pattern of kernel fusion to accelerate "
        "Transformer inference worldwide. This concludes my presentation, and I am ready for any viva questions."
    )

    prs.save(out_path)
    print(f"[OK] Presentation successfully generated at: {out_path}")

if __name__ == "__main__":
    create_presentation()
