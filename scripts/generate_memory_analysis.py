"""
Generates memory_traffic_analysis.csv detailing logical global memory traffic
for Separate Kernels vs Fused Kernel across all evaluated benchmark configurations.
"""
import os
import pandas as pd

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(base_dir, "results")
    bench_csv = os.path.join(results_dir, "benchmark_results.csv")
    out_csv = os.path.join(results_dir, "memory_traffic_analysis.csv")

    if not os.path.exists(bench_csv):
        print(f"Error: {bench_csv} not found.")
        return

    df = pd.read_csv(bench_csv)
    
    rows_data = []
    for _, r in df.iterrows():
        R = int(r["Rows"])
        C = int(r["Cols"])
        cfg = r["Config"]
        
        # Kernel 1 (Row Max): Read X (R*C*4), Write d_row_max (R*4)
        k1_read_x = R * C * 4
        k1_write_max = R * 4
        k1_total = k1_read_x + k1_write_max
        
        # Kernel 2 (Exp Sum): Read X (R*C*4), Read d_row_max (R*4), Write d_row_sum (R*4)
        k2_read_x = R * C * 4
        k2_read_max = R * 4
        k2_write_sum = R * 4
        k2_total = k2_read_x + k2_read_max + k2_write_sum
        
        # Kernel 3 (Normalize): Read X (R*C*4), Read d_row_max (R*4), Read d_row_sum (R*4), Write Y (R*C*4)
        k3_read_x = R * C * 4
        k3_read_max = R * 4
        k3_read_sum = R * 4
        k3_write_y = R * C * 4
        k3_total = k3_read_x + k3_read_max + k3_read_sum + k3_write_y
        
        sep_total = k1_total + k2_total + k3_total
        
        # Check if row fits in shared memory (RTX 3050 has 48KB = 49152 bytes)
        # Required smem: (C + 256) * 4 bytes
        fits_in_smem = ((C + 256) * 4) <= 49152

        if fits_in_smem:
            fused_impl = "SharedMemCached"
            # 1 cooperative input read into shared memory (R*C*4)
            # Row maximum & exponential sum computed in shared memory (0 DRAM)
            # In-place normalization in shared memory
            # 1 output write from shared memory to global memory (R*C*4)
            fused_read_x = R * C * 4
            fused_write_y = R * C * 4
            fused_total = fused_read_x + fused_write_y  # 8RC bytes
        else:
            fused_impl = "StreamingFallback"
            # Fallback streaming: 3 passes over input X (12RC) + 1 write Y (4RC) = 16RC
            fused_read_x = 3 * R * C * 4
            fused_write_y = R * C * 4
            fused_total = fused_read_x + fused_write_y

        bytes_saved = sep_total - fused_total
        pct_saved = (bytes_saved / sep_total) * 100.0

        rows_data.append({
            "Config": cfg,
            "Rows": R,
            "Cols": C,
            "Total_Elements": R * C,
            "Implementation": fused_impl,
            "K1_Max_Bytes": k1_total,
            "K2_Sum_Bytes": k2_total,
            "K3_Norm_Bytes": k3_total,
            "Separate_Total_Logical_Bytes": sep_total,
            "Separate_Total_MB": round(sep_total / (1024 * 1024), 3),
            "Fused_Read_Bytes": fused_read_x,
            "Fused_Write_Bytes": fused_write_y,
            "Fused_Total_Logical_Bytes": fused_total,
            "Fused_Total_MB": round(fused_total / (1024 * 1024), 3),
            "Bytes_Saved": bytes_saved,
            "MB_Saved": round(bytes_saved / (1024 * 1024), 3),
            "Traffic_Reduction_Pct": round(pct_saved, 2),
            "Speedup_Observed": r["Speedup"]
        })

    out_df = pd.DataFrame(rows_data)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved memory traffic analysis to {out_csv}")
    print(out_df[["Config", "Dimensions" if "Dimensions" in out_df else "Cols", "Separate_Total_MB", "Fused_Total_MB", "MB_Saved", "Traffic_Reduction_Pct", "Speedup_Observed"]])

if __name__ == "__main__":
    main()
