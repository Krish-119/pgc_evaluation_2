@echo off
setlocal
cd /d "%~dp0.."
if not exist "build\correctness_tests.exe" call "scripts\build.bat"
build\correctness_tests.exe
pause
