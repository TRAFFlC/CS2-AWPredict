@echo off
rem ============================================================
rem  Daily VRS fetcher (canonical, repo-rooted)
rem  Fetch : https://www.hltv.org/valve-ranking/teams/<today>
rem  Output: data\vrs_rank_<YYYY-MM-DD>.json
rem  Skip  : same-day snapshot already exists -> no network
rem  Log   : data\cache\update_vrs.log
rem  Usage : scheduled task, or double-click.
rem ============================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=python"
set "LOG=%~dp0data\cache\update_vrs.log"
if not exist "%~dp0data\cache" mkdir "%~dp0data\cache"

echo. 1>>"%LOG%"
echo [%date% %time%] update_vrs start 1>>"%LOG%"
"%PY%" -X utf8 "%~dp0fetch_vrs.py" %* 1>>"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo [%date% %time%] update_vrs exit=%RC% 1>>"%LOG%"
endlocal & exit /b %RC%
