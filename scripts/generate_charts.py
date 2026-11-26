"""
Generates high-resolution academic charts and technical diagrams for
Softmax and Kernel Fusion Using CUDA.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def setup_style():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'
    plt.rcParams['axes.edgecolor'] = '#CBD5E1'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#E2E8F0'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(base_dir, "results")
    diag_dir = os.path.join(base_dir, "diagrams")
    os.makedirs(diag_dir, exist_ok=True)

    bench_csv = os.path.join(results_dir, "benchmark_results.csv")
    block_csv = os.path.join(results_dir, "block_size_results.csv")
    mem_csv = os.path.join(results_dir, "memory_traffic_analysis.csv")

    setup_style()
    df_bench = pd.read_csv(bench_csv)
    df_block = pd.read_csv(block_csv)
    df_mem = pd.read_csv(mem_csv)

    labels = [f"{r['Rows']}x{r['Cols']}" for _, r in df_bench.iterrows()]
    x = np.arange(len(labels))
    width = 0.35

    # -------------------------------------------------------------
    # 1. Execution Time Comparison (Separate vs Fused)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    rects1 = ax.bar(x - width/2, df_bench['SeparateTotalMs'], width, label='Separate Kernels (3 Passes)', color='#1E3A8A', edgecolor='#0F172A')
    rects2 = ax.bar(x + width/2, df_bench['FusedTimeMs'], width, label='Fused Kernel (1 Block/Row)', color='#0284C7', edgecolor='#0369A1')

    ax.set_ylabel('Execution Time (milliseconds)', fontsize=11, fontweight='bold', color='#1E293B')
    ax.set_title('Kernel Execution Time: Separate vs Fused Softmax (NVIDIA RTX 3050 Laptop GPU)', fontsize=13, fontweight='bold', pad=15, color='#0F172A')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9.5, fontweight='bold', color='#334155')
    ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=10)

    for i in range(len(x)):
        sp = df_bench.loc[i, 'Speedup']
        ax.annotate(f"{sp:.2f}x\nFaster", 
                    xy=(x[i] + width/2, df_bench.loc[i, 'FusedTimeMs']),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#0369A1')

    fig.tight_layout()
    p1 = os.path.join(diag_dir, "01_execution_time_comparison.png")
    fig.savefig(p1, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated {p1}")

    # -------------------------------------------------------------
    # 2. Kernel Breakdown (Stacked Bar for Separate vs Fused)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    rects_max = ax.bar(x - width/2, df_bench['SeparateMaxMs'], width, label='K1: Row Maximum', color='#3B82F6')
    rects_sum = ax.bar(x - width/2, df_bench['SeparateSumMs'], width, bottom=df_bench['SeparateMaxMs'], label='K2: Exp Sum Reduction', color='#60A5FA')
    rects_norm = ax.bar(x - width/2, df_bench['SeparateNormMs'], width, bottom=df_bench['SeparateMaxMs'] + df_bench['SeparateSumMs'], label='K3: Normalization', color='#93C5FD')
    rects_fused = ax.bar(x + width/2, df_bench['FusedTimeMs'], width, label='Fused Kernel Total', color='#0D9488', edgecolor='#115E59')

    ax.set_ylabel('Time Breakdown (milliseconds)', fontsize=11, fontweight='bold', color='#1E293B')
    ax.set_title('Kernel Execution Breakdown & Pipeline Latency (RTX 3050)', fontsize=13, fontweight='bold', pad=15, color='#0F172A')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9.5, fontweight='bold', color='#334155')
    ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5)

    fig.tight_layout()
    p2 = os.path.join(diag_dir, "02_kernel_breakdown.png")
    fig.savefig(p2, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated {p2}")

    # -------------------------------------------------------------
    # 3. Speedup Scaling Curve
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.plot(x, df_bench['Speedup'], marker='o', color='#0284C7', linewidth=2.5, markersize=8, label='Measured Speedup')
    ax.axhline(1.0, color='#DC2626', linestyle='--', linewidth=1.2, label='Baseline (1.0x)')
    ax.set_ylabel('Speedup Factor (x)', fontsize=11, fontweight='bold', color='#1E293B')
    ax.set_title('Kernel Fusion Speedup across Matrix Dimensions', fontsize=13, fontweight='bold', pad=15, color='#0F172A')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9.5, fontweight='bold', color='#334155')
    ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=10)

    for i, txt in enumerate(df_bench['Speedup']):
        ax.annotate(f"{txt:.2f}x", (x[i], txt), textcoords="offset points", xytext=(0, 8), ha='center', fontweight='bold', color='#0369A1')

    fig.tight_layout()
    p3 = os.path.join(diag_dir, "03_speedup_scaling.png")
    fig.savefig(p3, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated {p3}")

    # -------------------------------------------------------------
    # 4. Block Size Analysis (RTX 3050)
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    b_sizes = [str(val) for val in df_block['BlockSize']]
    xb = np.arange(len(b_sizes))
    color1 = '#0284C7'
    color2 = '#D97706'

    ax1.set_xlabel('Thread Block Size (Threads per Block)', fontsize=11, fontweight='bold', color='#1E293B')
    ax1.set_ylabel('Fused Execution Latency (ms)', color=color1, fontsize=11, fontweight='bold')
    line1 = ax1.plot(xb, df_block['FusedTimeMs'], marker='s', color=color1, linewidth=2.5, markersize=8, label='Fused Latency (ms)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(xb)
    ax1.set_xticklabels(b_sizes, fontweight='bold', color='#334155')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Effective Memory Bandwidth (GB/s)', color=color2, fontsize=11, fontweight='bold')
    line2 = ax2.plot(xb, df_block['FusedBandwidthGBs'], marker='^', color=color2, linewidth=2.5, markersize=8, linestyle='--', label='Bandwidth (GB/s)')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(False)

    plt.title('Block Size Tuning & Occupancy Impact (Matrix 2048 x 1024)', fontsize=13, fontweight='bold', pad=15, color='#0F172A')
    lines = line1 + line2
    labels_b = [l.get_label() for l in lines]
    ax1.legend(lines, labels_b, loc='center right', frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1')

    fig.tight_layout()
    p4 = os.path.join(diag_dir, "04_block_size_analysis.png")
    fig.savefig(p4, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated {p4}")

    # -------------------------------------------------------------
    # 5. Global Memory Traffic Comparison (MB)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    rects_m1 = ax.bar(x - width/2, df_mem['Separate_Total_MB'], width, label='Separate Kernels (16RC+20R Bytes)', color='#475569')
    rects_m2 = ax.bar(x + width/2, df_mem['Fused_Total_MB'], width, label='Fused Kernel (8RC Bytes Shared-Cached)', color='#059669')

    ax.set_ylabel('Total Global Memory Traffic (Megabytes)', fontsize=11, fontweight='bold', color='#1E293B')
    ax.set_title('Logical Global Memory Traffic Reduction (~50% Reduction via Shared-Memory Caching)', fontsize=13, fontweight='bold', pad=15, color='#0F172A')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9.5, fontweight='bold', color='#334155')
    ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=10)

    for i in range(len(x)):
        saved = df_mem.loc[i, 'MB_Saved']
        pct = df_mem.loc[i, 'Traffic_Reduction_Pct']
        ax.annotate(f"-{saved:.1f} MB\n(-{pct:.1f}%)", 
                    xy=(x[i] + width/2, df_mem.loc[i, 'Fused_Total_MB']),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#047857')

    fig.tight_layout()
    p5 = os.path.join(diag_dir, "05_memory_traffic_comparison.png")
    fig.savefig(p5, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated {p5}")

    # -------------------------------------------------------------
    # 6. Separate Architecture Schematic
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    boxes = [
        ("Input Matrix X\n(Global Memory)", 0.8, 2.0, "#E2E8F0", "#334155"),
        ("Kernel 1\nRow Maximum", 2.8, 2.0, "#DBEAFE", "#1E40AF"),
        ("Global Memory\nd_row_max", 4.8, 2.0, "#FEF3C7", "#92400E"),
        ("Kernel 2\nExp Sum Reduction", 6.8, 2.0, "#DBEAFE", "#1E40AF"),
        ("Global Memory\nd_row_sum", 8.8, 2.0, "#FEF3C7", "#92400E"),
    ]
    k3_box = ("Kernel 3\nNormalize & Output", 5.8, 0.5, "#DCFCE7", "#166534")

    for text, bx, by, bg, border in boxes:
        ax.text(bx, by, text, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.6', facecolor=bg, edgecolor=border, linewidth=1.5), fontsize=8.5, fontweight='bold', color='#0F172A')

    ax.text(k3_box[1], k3_box[2], k3_box[0], ha='center', va='center', bbox=dict(boxstyle='round,pad=0.6', facecolor=k3_box[3], edgecolor=k3_box[4], linewidth=1.5), fontsize=9, fontweight='bold', color='#0F172A')

    arrowprops = dict(arrowstyle='->', lw=1.8, color='#475569')
    ax.annotate('', xy=(2.0, 2.0), xytext=(1.6, 2.0), arrowprops=arrowprops)
    ax.annotate('', xy=(4.0, 2.0), xytext=(3.6, 2.0), arrowprops=arrowprops)
    ax.annotate('', xy=(6.0, 2.0), xytext=(5.6, 2.0), arrowprops=arrowprops)
    ax.annotate('', xy=(8.0, 2.0), xytext=(7.6, 2.0), arrowprops=arrowprops)

    ax.annotate('', xy=(5.8, 1.0), xytext=(4.8, 1.5), arrowprops=arrowprops)
    ax.annotate('', xy=(6.2, 1.0), xytext=(8.8, 1.5), arrowprops=arrowprops)

    ax.set_title("Separate Kernels Architecture: 3 Kernel Launches + Global Memory Intermediates (16RC+20R Bytes)", fontsize=12, fontweight='bold', color='#0F172A', pad=10)
    fig.tight_layout()
    p6 = os.path.join(diag_dir, "06_architecture_separate.png")
    fig.savefig(p6, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated {p6}")

    # -------------------------------------------------------------
    # 7. Fused Architecture Schematic
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    ax.text(1.0, 2.0, "Input Matrix X\n(Global Memory)\n[Read 4RC]", ha='center', va='center', bbox=dict(boxstyle='round,pad=0.6', facecolor='#E2E8F0', edgecolor='#334155', linewidth=1.5), fontsize=8.5, fontweight='bold', color='#0F172A')
    
    # Large Fused Block
    fused_box = patches.FancyBboxPatch((2.4, 0.3), 5.2, 3.4, boxstyle="round,pad=0.2", facecolor='#EFF6FF', edgecolor='#2563EB', linewidth=2.0, linestyle='--')
    ax.add_patch(fused_box)
    ax.text(5.0, 3.35, "FUSED CUDA KERNEL (1 Block per Row, Dynamic Shared Memory)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1E40AF')

    ax.text(3.6, 2.2, "1. Cooperative Load\n(__shared__ s_row[C])\n__syncthreads()", ha='center', va='center', bbox=dict(boxstyle='round,pad=0.4', facecolor='#DBEAFE', edgecolor='#3B82F6', linewidth=1.2), fontsize=8, fontweight='bold')
    ax.text(6.4, 2.2, "2. On-Chip Reductions\nMax & Exp-Sum (s_reduce)\n__syncthreads()", ha='center', va='center', bbox=dict(boxstyle='round,pad=0.4', facecolor='#DBEAFE', edgecolor='#3B82F6', linewidth=1.2), fontsize=8, fontweight='bold')
    ax.text(5.0, 0.95, "3. In-Place Normalize & Coalesced Write\n(s_row[c] = exp(...) / sum -> Write Y)\n[Total: 8RC Logical Bytes]", ha='center', va='center', bbox=dict(boxstyle='round,pad=0.4', facecolor='#DCFCE7', edgecolor='#16A34A', linewidth=1.2), fontsize=8, fontweight='bold')

    ax.text(9.0, 2.0, "Output Matrix Y\n(Global Memory)\n[Write 4RC]", ha='center', va='center', bbox=dict(boxstyle='round,pad=0.6', facecolor='#DCFCE7', edgecolor='#166534', linewidth=1.5), fontsize=8.5, fontweight='bold', color='#0F172A')

    ax.annotate('', xy=(2.4, 2.0), xytext=(1.8, 2.0), arrowprops=arrowprops)
    ax.annotate('', xy=(8.2, 2.0), xytext=(7.6, 2.0), arrowprops=arrowprops)
    ax.annotate('', xy=(4.9, 1.4), xytext=(3.9, 1.7), arrowprops=arrowprops)
    ax.annotate('', xy=(5.1, 1.4), xytext=(6.1, 1.7), arrowprops=arrowprops)

    ax.set_title("Fused Kernel Architecture: 1 Launch, Row Cached On-Chip, 50% Logical Traffic Reduction (8RC Bytes)", fontsize=12, fontweight='bold', color='#0F172A', pad=10)
    fig.tight_layout()
    p7 = os.path.join(diag_dir, "07_architecture_fused.png")
    fig.savefig(p7, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated {p7}")

if __name__ == "__main__":
    main()
