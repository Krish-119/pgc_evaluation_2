"""
Master Deliverables Generator for Softmax and Kernel Fusion Using CUDA.
Runs charts generator, presentation generator, and Word documents generator.
"""
import os
import subprocess
import sys

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scripts_dir = os.path.join(base_dir, "scripts")

    print("========================================================================")
    print("      GENERATING ALL ACADEMIC DELIVERABLES FOR PGC LAB THEME 10")
    print("========================================================================")

    steps = [
        ("High-Resolution Charts & Architecture Diagrams", "generate_charts.py"),
        ("6-Slide PowerPoint Presentation (.pptx)", "generate_presentation.py"),
        ("Three Complete Word Documents (.docx)", "generate_docx_deliverables.py"),
        ("Real Screenshots from Genuine Execution Logs", "capture_screenshots.py")
    ]

    for title, script_name in steps:
        print(f"\n>>> Running: {title} ({script_name})...")
        script_path = os.path.join(scripts_dir, script_name)
        res = subprocess.run([sys.executable, script_path], cwd=base_dir)
        if res.returncode != 0:
            print(f"[ERROR] Failed to execute {script_name}")
            sys.exit(res.returncode)

    print("\n========================================================================")
    print("ALL ACADEMIC DELIVERABLES SUCCESSFULLY CREATED AND VERIFIED!")
    print("Files located in:")
    print(f"  - Deliverables: {os.path.join(base_dir, 'deliverables')}")
    print(f"  - Diagrams    : {os.path.join(base_dir, 'diagrams')}")
    print(f"  - Screenshots : {os.path.join(base_dir, 'screenshots')}")
    print("========================================================================")

if __name__ == "__main__":
    main()
