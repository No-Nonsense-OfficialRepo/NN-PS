@echo off
cd /d "%~dp0"
".testenv\Scripts\python.exe" bump_version.py %*
pause
