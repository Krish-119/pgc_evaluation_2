"""
Creates desktop launcher shortcuts for the Softmax and Kernel Fusion project.
"""
import os
import subprocess

def main():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    proj_dir = r"C:\Users\ariha\Desktop\PGC LAB\Softmax_Kernel_Fusion"
    start_bat = os.path.join(proj_dir, "START_PROJECT.bat")

    # 1. Create Launch_Softmax_CUDA_Project.bat on Desktop
    desktop_bat = os.path.join(desktop, "Launch_Softmax_CUDA_Project.bat")
    with open(desktop_bat, "w", encoding="utf-8") as f:
        f.write("@echo off\n")
        f.write(f'cd /d "{proj_dir}"\n')
        f.write('call "START_PROJECT.bat"\n')
    print(f"[OK] Created Desktop Batch Launcher: {desktop_bat}")

    # 2. Also create inside project directory as LAUNCH_PROJECT.bat
    proj_launcher = os.path.join(proj_dir, "LAUNCH_PROJECT.bat")
    with open(proj_launcher, "w", encoding="utf-8") as f:
        f.write("@echo off\n")
        f.write(f'cd /d "%~dp0"\n')
        f.write('call "START_PROJECT.bat"\n')
    print(f"[OK] Created Project Launcher: {proj_launcher}")

    # 3. Create Windows .lnk shortcut on Desktop via PowerShell script
    ps_cmd = f"""
    $WshShell = New-Object -comObject WScript.Shell
    $shortcut = $WshShell.CreateShortcut('{os.path.join(desktop, "CUDA Softmax Lab.lnk")}')
    $shortcut.TargetPath = '{start_bat}'
    $shortcut.WorkingDirectory = '{proj_dir}'
    $shortcut.Description = 'Theme 10 - Softmax and Kernel Fusion Using CUDA'
    $shortcut.Save()
    """
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=True)
        print(f"[OK] Created Desktop .lnk Shortcut: {os.path.join(desktop, 'CUDA Softmax Lab.lnk')}")
    except Exception as e:
        print(f"[NOTE] PowerShell shortcut creation: {e}")

if __name__ == "__main__":
    main()
