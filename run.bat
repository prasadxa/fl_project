@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  Federated Learning - Local Orchestrator  (Windows)
REM  Double-click or run from anywhere - it auto-locates itself.
REM ============================================================

REM Always cd to the directory containing this batch file
cd /d "%~dp0"

set PROJECT_ROOT=%~dp0
set PYTHON=python
set LOGS=%PROJECT_ROOT%logs
set PYTHONIOENCODING=utf-8

if not exist "%LOGS%" mkdir "%LOGS%"

echo ============================================================
echo  Starting Federated Learning (5 rounds, 3 clients)
echo ============================================================

REM ── 1. Start server ──────────────────────────────────────────
echo [1/5] Starting Flower server...
start "FL-Server" /B %PYTHON% "%PROJECT_ROOT%backend\server.py" > "%LOGS%\server.log" 2>&1

REM ── 2. Wait for server to bind port 8080 ─────────────────────
echo [2/5] Waiting 6 s for server to initialise...
timeout /t 6 /nobreak > nul

REM ── 3. Start clients ─────────────────────────────────────────
echo [3/5] Starting Client 1...
start "FL-Client-1" /B %PYTHON% "%PROJECT_ROOT%backend\client.py" --client_id 1 > "%LOGS%\client1.log" 2>&1

echo [4/5] Starting Client 2...
start "FL-Client-2" /B %PYTHON% "%PROJECT_ROOT%backend\client.py" --client_id 2 > "%LOGS%\client2.log" 2>&1

echo [5/5] Starting Client 3...
start "FL-Client-3" /B %PYTHON% "%PROJECT_ROOT%backend\client.py" --client_id 3 > "%LOGS%\client3.log" 2>&1

echo.
echo  All processes launched.
echo  Logs are written to: %LOGS%\
echo  (server.log  client1.log  client2.log  client3.log)
echo.
echo  Press any key to STOP all FL processes and exit...
pause > nul

REM ── Cleanup: kill by window title ────────────────────────────
echo Terminating FL processes...
taskkill /F /FI "WindowTitle eq FL-Server"   /T > nul 2>&1
taskkill /F /FI "WindowTitle eq FL-Client-1" /T > nul 2>&1
taskkill /F /FI "WindowTitle eq FL-Client-2" /T > nul 2>&1
taskkill /F /FI "WindowTitle eq FL-Client-3" /T > nul 2>&1
echo Done.

endlocal
