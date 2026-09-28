@echo off
title Furniture Recommendation AI

cd /d "%~dp0"

echo ==========================================
echo   Furniture Recommendation AI
echo ==========================================
echo.
echo Starting Streamlit application...
echo.

if not exist ".venv-1\Scripts\python.exe" (
    echo ERROR: .venv-1 Python environment was not found.
    echo.
    pause
    exit /b 1
)

".venv-1\Scripts\python.exe" -m streamlit run app.py

pause
