@echo off
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer from python.org and tick Add python.exe to PATH.
  pause
  exit /b 1
)

if not exist venv\Scripts\activate.bat (
  echo Creating virtual environment...
  python -m venv venv
)
call venv\Scripts\activate.bat

if not exist venv\.packages_installed (
  echo Installing packages. The first time takes several minutes...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Package install failed. Check your internet connection and run this file again.
    pause
    exit /b 1
  )
  echo ok> venv\.packages_installed
)

if not exist .env copy .env.example .env >nul

dir /b data\*.pdf >nul 2>&1
if errorlevel 1 (
  echo No PDF found. Copy a text-based PDF into the data folder, then run run.bat again.
  start "" explorer data
  pause
  exit /b 1
)

if not exist chroma_db (
  echo Building the search index. The first time can take a few minutes...
  python -m src.ingest
  if errorlevel 1 (
    echo Index build failed. See the message above.
    pause
    exit /b 1
  )
)

if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  echo [general]> "%USERPROFILE%\.streamlit\credentials.toml"
  echo email = "">> "%USERPROFILE%\.streamlit\credentials.toml"
)

echo.
echo Starting the app. If the browser does not open, go to http://localhost:8501
echo Press Ctrl+C in this window to stop.
python -m streamlit run app.py
pause
