@echo off
setlocal enabledelayedexpansion

echo ========================================================================
echo        CUDA SOFTMAX AND KERNEL FUSION - PROJECT COMPILATION
echo ========================================================================

set "PROJ_DIR=%~dp0.."
set "BUILD_DIR=%PROJ_DIR%\build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo.
echo [1/3] Locating Microsoft Visual Studio MSVC environment...
set "VCVARS_BAT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS_BAT%" (
    set "VCVARS_BAT=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
)

if not exist "%VCVARS_BAT%" (
    echo [ERROR] Could not find vcvars64.bat at known locations.
    echo Please ensure Visual Studio or Build Tools is installed.
    exit /b 1
)

echo Found MSVC environment: "%VCVARS_BAT%"
call "%VCVARS_BAT%" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to initialize MSVC x64 build environment!
    exit /b 1
)

echo.
echo [2/3] Compiling main application (src/main.cu -^> build/softmax_fusion.exe)...
nvcc -O3 -arch=sm_86 -std=c++17 ^
     -I"%PROJ_DIR%\src" ^
     -o "%BUILD_DIR%\softmax_fusion.exe" ^
     "%PROJ_DIR%\src\main.cu"

if %errorlevel% neq 0 (
    echo [ERROR] Compilation of main application failed!
    exit /b 1
)
echo [OK] main application compiled successfully: "%BUILD_DIR%\softmax_fusion.exe"

echo.
echo [3/3] Compiling test suite (tests/correctness_tests.cu -^> build/correctness_tests.exe)...
nvcc -O3 -arch=sm_86 -std=c++17 ^
     -I"%PROJ_DIR%\src" ^
     -o "%BUILD_DIR%\correctness_tests.exe" ^
     "%PROJ_DIR%\tests\correctness_tests.cu"

if %errorlevel% neq 0 (
    echo [ERROR] Compilation of test suite failed!
    exit /b 1
)
echo [OK] test suite compiled successfully: "%BUILD_DIR%\correctness_tests.exe"

echo.
echo ========================================================================
echo ALL TARGETS COMPILED SUCCESSFULLY!
echo Binaries located in: %BUILD_DIR%
echo ========================================================================
