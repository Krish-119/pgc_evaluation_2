@echo off
setlocal enabledelayedexpansion

echo ========================================================================
echo        CUDA SOFTMAX AND KERNEL FUSION - ENVIRONMENT VERIFICATION
echo ========================================================================

echo.
echo [1/4] Checking NVIDIA Driver and GPU...
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
if %errorlevel% neq 0 (
    echo [ERROR] nvidia-smi failed! NVIDIA driver or GPU not detected.
    exit /b 1
)

echo.
echo [2/4] Checking NVIDIA CUDA Compiler (nvcc)...
nvcc --version
if %errorlevel% neq 0 (
    echo [ERROR] nvcc is not found in PATH!
    exit /b 1
)

echo.
echo [3/4] Checking Microsoft Visual C++ Compiler (MSVC)...
set "VCVARS_BAT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if exist "%VCVARS_BAT%" (
    echo [OK] Found MSVC vcvars64.bat: "%VCVARS_BAT%"
) else (
    set "VCVARS_BAT=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
    if exist "%VCVARS_BAT%" (
        echo [OK] Found MSVC vcvars64.bat: "%VCVARS_BAT%"
    ) else (
        echo [WARNING] vcvars64.bat not found at default paths. Make sure MSVC is installed.
    )
)

echo.
echo [4/4] Checking Python Environment...
python --version
if %errorlevel% neq 0 (
    echo [WARNING] Python not found in PATH.
)

echo.
echo Running comprehensive Python environment inspector...
python "%~dp0inspect_environment.py"

echo.
echo ========================================================================
echo Environment verification completed successfully.
echo Report saved to results/system_information.txt
echo ========================================================================
