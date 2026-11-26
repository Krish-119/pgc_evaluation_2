"""
Inspects system hardware, GPU, CUDA, MSVC, and Python configuration.
Writes comprehensive diagnostic information to results/system_information.txt.
"""
import os
import sys
import platform
import subprocess
import psutil
from datetime import datetime

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        return res.stdout.strip()
    except Exception as e:
        return f"Error executing '{cmd}': {str(e)}"

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "system_information.txt")

    lines = []
    lines.append("========================================================================")
    lines.append("        CUDA SOFTMAX & KERNEL FUSION - SYSTEM HARDWARE & SOFTWARE REPORT")
    lines.append("========================================================================")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("--- [1] HOST OPERATING SYSTEM & ARCHITECTURE ---")
    lines.append(f"OS: {platform.system()} {platform.release()} (Version: {platform.version()})")
    lines.append(f"Architecture: {platform.machine()}")
    lines.append(f"Platform Detail: {platform.platform()}")
    lines.append("")

    lines.append("--- [2] CPU DETAILS & SYSTEM RAM ---")
    lines.append(f"Processor: {platform.processor()}")
    lines.append(f"Physical CPU Cores: {psutil.cpu_count(logical=False)}")
    lines.append(f"Logical CPU Processors (Threads): {psutil.cpu_count(logical=True)}")
    vm = psutil.virtual_memory()
    lines.append(f"Total Physical RAM: {vm.total / (1024**3):.2f} GB")
    lines.append(f"Available RAM: {vm.available / (1024**3):.2f} GB ({vm.percent}% used)")
    lines.append("")

    lines.append("--- [3] NVIDIA GPU STATUS (nvidia-smi) ---")
    smi_query = run_cmd("nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv")
    lines.append(smi_query)
    lines.append("")
    smi_full = run_cmd("nvidia-smi")
    lines.append(smi_full)
    lines.append("")

    lines.append("--- [4] NVIDIA CUDA COMPILER (nvcc) ---")
    nvcc_info = run_cmd("nvcc --version")
    lines.append(nvcc_info)
    lines.append("")

    lines.append("--- [5] MICROSOFT VISUAL C++ COMPILER (MSVC) ---")
    vcvars_path = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if os.path.exists(vcvars_path):
        lines.append(f"MSVC 2022 Build Tools vcvars64.bat FOUND at:\n  {vcvars_path}")
    else:
        lines.append(f"vcvars64.bat NOT found at {vcvars_path}")
    cl_check = run_cmd(f'cmd.exe /c "call \"{vcvars_path}\" && where cl"')
    lines.append("MSVC cl location:")
    lines.append(cl_check)
    lines.append("")

    lines.append("--- [6] PYTHON ENVIRONMENT & LIBRARIES ---")
    lines.append(f"Python Executable: {sys.executable}")
    lines.append(f"Python Version: {platform.python_version()}")
    for pkg in ["numpy", "pandas", "matplotlib", "docx", "pptx", "PIL", "psutil"]:
        try:
            m = __import__(pkg)
            ver = getattr(m, "__version__", "installed")
            lines.append(f"  - {pkg}: {ver}")
        except ImportError:
            lines.append(f"  - {pkg}: Not installed")
    lines.append("")
    lines.append("========================================================================")
    lines.append("END OF SYSTEM INFORMATION REPORT")
    lines.append("========================================================================")

    content = "\n".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"System information saved to {out_file}")

if __name__ == "__main__":
    main()
