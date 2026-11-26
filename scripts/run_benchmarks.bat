@echo off
setlocal
cd /d "%~dp0.."
if not exist "build\softmax_fusion.exe" call "scripts\build.bat"
build\softmax_fusion.exe --benchmark
echo.
build\softmax_fusion.exe --block-analysis
pause
