@echo off
setlocal

echo ========================================================================
echo        STARTING CUDA SOFTMAX VISUALIZATION DASHBOARD
echo ========================================================================

python "%~dp0..\dashboard\server.py"
pause
