@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PYTHON=python
set PYTHONPATH=%~dp0
cd /d "%~dp0"

echo.
echo ============================================================
echo    TECNOMATE  --  Federated Learning Edge System
echo ============================================================
echo.

:: ── 1. FL Server ─────────────────────────────────────────────
echo [1/5] Starting FL Server (always-on, min 2 clients)...
start "Tecnomate - FL Server" cmd /k "set PYTHONIOENCODING=utf-8 && set PYTHONPATH=%~dp0 && %PYTHON% -u backend/app.py --mode server"
timeout /t 6 /nobreak >nul

:: ── 2. Local FL Clients ───────────────────────────────────────
echo [2/5] Starting FL Client 1...
start "Tecnomate - Client 1" cmd /k "set PYTHONIOENCODING=utf-8 && set PYTHONPATH=%~dp0 && %PYTHON% -u backend/client.py --client_id 1"
timeout /t 2 /nobreak >nul

echo [3/5] Starting FL Client 2...
start "Tecnomate - Client 2" cmd /k "set PYTHONIOENCODING=utf-8 && set PYTHONPATH=%~dp0 && %PYTHON% -u backend/client.py --client_id 2"
timeout /t 2 /nobreak >nul

echo [4/5] Starting FL Client 3...
start "Tecnomate - Client 3" cmd /k "set PYTHONIOENCODING=utf-8 && set PYTHONPATH=%~dp0 && %PYTHON% -u backend/client.py --client_id 3"
timeout /t 3 /nobreak >nul

:: ── 3. Streamlit Clinical UI ───────────────────────────────────────
echo [5/5] Starting Tecnomate Clinical UI (Streamlit)...
start "Tecnomate - Clinical UI" cmd /k "set PYTHONIOENCODING=utf-8 && set PYTHONPATH=%~dp0 && %PYTHON% -u -m streamlit run backend/app.py --server.port 8501"

echo.
echo ============================================================
echo  All Tecnomate edge systems are running!
echo  Clinical UI:  http://localhost:8501
echo  FL Server:    localhost:8080
echo  Close each window individually to stop a component.
echo ============================================================
echo.
pause
