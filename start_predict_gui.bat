@echo off
rem ============================================================
rem  CS2-AWPredict web service launcher (canonical, repo-rooted)
rem  Service : http://127.0.0.1:8765
rem  Repo    : E:\world\CS2-AWPredict
rem  Log     : gui.out / gui.err (this folder, overwritten per run)
rem  Usage   : scheduled task at logon, or double-click.
rem  This file MUST live in the repo root; it always cd's here.
rem ============================================================
setlocal
cd /d "%~dp0"
set "PORT=8765"
set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=python"

rem 1) already listening? -> assume the service is already up, do nothing
set "USED=0"
for /f "usebackq delims=" %%c in (`powershell -NoProfile -Command "(Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | Measure-Object).Count"`) do set "USED=%%c"
if not "%USED%"=="0" (
  echo [%date% %time%] port %PORT% already in use; launcher exits.>>"%~dp0gui.launcher.log"
  goto :eof
)

rem 2) start hidden, detached, with stdout/stderr captured
echo [%date% %time%] starting %PY% in %CD%>>"%~dp0gui.launcher.log"
powershell -NoProfile -Command "Start-Process -FilePath '%PY%' -ArgumentList '-X','utf8','predict_gui.py','--port','%PORT%' -WorkingDirectory '%~dp0.' -RedirectStandardOutput '%~dp0gui.out' -RedirectStandardError '%~dp0gui.err' -WindowStyle Hidden"
endlocal
