@echo off
setlocal
title CUDA Softmax and Kernel Fusion - PGC Lab (Theme 10)

cd /d "%~dp0"
set "PROJ_DIR=%cd%"

REM Check if binaries exist, compile if missing
if not exist "%PROJ_DIR%\build\softmax_fusion.exe" (
    echo [INFO] Binaries not found. Triggering automated build...
    call "%PROJ_DIR%\scripts\build.bat"
    if errorlevel 1 (
        echo [ERROR] Build failed. Please inspect errors above.
        pause
        exit /b 1
    )
)

:MENU
cls
echo ========================================================================
echo          CUDA SOFTMAX AND KERNEL FUSION - PGC LAB (THEME 10)
echo ========================================================================
echo  Active Hardware Target : NVIDIA GeForce RTX 3050 6GB Laptop GPU
echo  CUDA Compiler / Toolkit: nvcc release 13.4 (Compute sm_86)
echo  Host Platform          : Windows 11 64-bit / MSVC 2022 C++17
echo ========================================================================
echo.
echo  1. Run Softmax Demonstration (2x4 Mandatory Theme Matrix)
echo  2. Enter Custom Matrix (Interactive User Input)
echo  3. Run Complete Benchmark (7 Configurations, 10 Warmup + 100 Runs)
echo  4. Run Correctness and Stability Tests (12 Automated Test Cases)
echo  5. Run Block Size Analysis (64 to 1024 Threads Tuning)
echo  6. Open Visualization Dashboard (Browser Laboratory Interface)
echo  7. Generate Academic Deliverables (PPTX, DOCX and Charts)
echo  8. Exit
echo.
set "CHOICE="
set /p "CHOICE=Enter selection [1-8]: "

if "%CHOICE%"=="1" (
    echo.
    "%PROJ_DIR%\build\softmax_fusion.exe" --demo
    echo.
    pause
    goto MENU
)
if "%CHOICE%"=="2" (
    echo.
    "%PROJ_DIR%\build\softmax_fusion.exe"
    goto MENU
)
if "%CHOICE%"=="3" (
    echo.
    "%PROJ_DIR%\build\softmax_fusion.exe" --benchmark
    echo.
    pause
    goto MENU
)
if "%CHOICE%"=="4" (
    echo.
    "%PROJ_DIR%\build\correctness_tests.exe"
    echo.
    pause
    goto MENU
)
if "%CHOICE%"=="5" (
    echo.
    "%PROJ_DIR%\build\softmax_fusion.exe" --block-analysis
    echo.
    pause
    goto MENU
)
if "%CHOICE%"=="6" (
    echo.
    start "" "%PROJ_DIR%\scripts\start_dashboard.bat"
    goto MENU
)
if "%CHOICE%"=="7" (
    echo.
    echo Generating charts and academic presentations...
    python "%PROJ_DIR%\scripts\generate_charts.py"
    python "%PROJ_DIR%\scripts\generate_presentation.py"
    python "%PROJ_DIR%\scripts\generate_docx_deliverables.py"
    echo [OK] All academic deliverables regenerated successfully!
    echo.
    pause
    goto MENU
)
if "%CHOICE%"=="8" (
    echo Exiting. Goodbye!
    exit /b 0
)

if not defined CHOICE (
    echo [INFO] Exiting interactive menu.
    exit /b 0
)

echo [WARNING] Invalid selection. Please choose 1-8.
ping 127.0.0.1 -n 2 >nul
goto MENU
