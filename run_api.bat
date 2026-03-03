@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  Tecnomate Clinical AI — FastAPI Launcher  (Windows)
REM  Double-click to start the backend + open the frontend.
REM ============================================================

cd /d "%~dp0"

set PROJECT_ROOT=%~dp0
set PYTHON=python
set HOST=127.0.0.1
set PORT=8000
set PYTHONPATH=%~dp0
set PYTHONIOENCODING=utf-8

echo.
echo  ============================================================
echo   Tecnomate Clinical AI ^| FastAPI Backend Launcher
echo  ============================================================
echo.

REM ── Check Python exists ──────────────────────────────────────
if not exist "%PYTHON%" (
    echo  [ERROR] Python not found at:
    echo          %PYTHON%
    echo.
    echo  Edit the PYTHON variable in this file to point to
    echo  your Python 3.14 installation.
    pause
    exit /b 1
)

echo  [1/4] Python found: %PYTHON%

REM ── Install / verify FastAPI + uvicorn ───────────────────────
echo  [2/4] Checking FastAPI + uvicorn installation...
%PYTHON% -m pip install --quiet --upgrade fastapi uvicorn[standard] python-multipart 2>nul
if errorlevel 1 (
    echo  [WARN] pip install returned a non-zero exit code.
    echo         If the server fails to start, run manually:
    echo         pip install fastapi uvicorn[standard] python-multipart
)
echo         Done.

REM ── Check the frontend folder exists ─────────────────────────
if not exist "%PROJECT_ROOT%frontend\index.html" (
    echo.
    echo  [WARN] frontend\index.html not found.
    echo         The API will still start but the UI will not be served.
    echo.
)

REM ── Launch the server ─────────────────────────────────────────
echo  [3/4] Starting API server on http://%HOST%:%PORT% ...
echo.
echo  Press Ctrl+C in this window to stop the server.
echo  ============================================================
echo.

REM ── Open browser after a short delay (background job) ────────
echo  [4/4] Opening browser in 3 seconds...
start "" /B cmd /c "timeout /t 3 /nobreak >nul && start http://%HOST%:%PORT%"

REM ── Run uvicorn (blocking — stays in foreground) ─────────────
%PYTHON% -m uvicorn backend.api:app --host %HOST% --port %PORT% --reload

echo.
echo  Server stopped.
pause
endlocal
