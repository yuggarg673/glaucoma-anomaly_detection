@echo off
rem Starts the website. Double-click this file.
rem The first run creates a virtual environment (.venv) and installs the requirements. That happens once and takes a
rem few minutes because PyTorch is a large download. After that it starts in a few seconds.
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (set PYCMD=py -3) else (set PYCMD=python)

%PYCMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 goto nopython

if not exist ".venv\Scripts\python.exe" (
  echo Creating a virtual environment in .venv ...
  %PYCMD% -m venv .venv
  if errorlevel 1 goto failed
)

if not exist ".venv\.requirements-installed" (
  echo Installing the requirements. First run only, this takes a few minutes ...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto failed
  echo done> ".venv\.requirements-installed"
)

echo Starting the website. Close this window to stop it.
".venv\Scripts\python.exe" -m backend.app --open %*
goto end

:nopython
echo Python 3.10 or newer was not found.
echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH" in the installer. Then double-click this file again.
goto end

:failed
echo Something went wrong. Read the messages above. See docs\TROUBLESHOOTING.md.

:end
pause
