@echo off
cd /d "%~dp0"
set "PY="
where py >nul 2>nul && set "PY=py -3.11"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python 3.11 is needed to build this app.
  goto fail
)
if not exist ".buildenv\Scripts\python.exe" (
  %PY% -m venv .buildenv
  if errorlevel 1 goto fail
)
".buildenv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
".buildenv\Scripts\python.exe" build.py
if errorlevel 1 goto fail
echo Finished. The app is No-Nonsense Photo Sorter.exe in the Photo Sorter folder.
pause
exit /b 0
:fail
echo Build failed. Please review the output above.
pause
exit /b 1
