@echo off
REM CreditIQ launcher for Windows. Double-click or run from a terminal in this folder.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Could not create a virtual environment. Install Python 3.11 or 3.12 from python.org
        echo and tick "Add python.exe to PATH" / the "py launcher" option, then run this file again.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

if not exist ".venv\.creditiq_installed" (
    echo Installing core requirements - first run only, this takes a few minutes ...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Installing requirements failed. See docs\TROUBLESHOOTING.md
        pause
        exit /b 1
    )
    echo Installing optional extras: LightGBM and SHAP. The app still works if this step fails ...
    python -m pip install -r requirements-optional.txt
    echo installed> ".venv\.creditiq_installed"
)

echo Starting CreditIQ at http://localhost:8501 - press Ctrl+C in this window to stop
python -m streamlit run app.py
pause
