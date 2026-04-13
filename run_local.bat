@echo off
REM Double-click this file to start the Streamlit dashboard on localhost.
REM Edit the two model paths below if your folders are different.

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\streamlit.exe" (
    echo ERROR: Virtual environment not found.
    echo Create it first: python -m venv .venv
    echo Then run: .\.venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Offline Hugging Face models (must exist on disk before indexing/query).
set "OFFLINE_ONLY=true"
set "LOCAL_LLM_ONLY=true"
set "EMBEDDING_MODEL_LOCAL_PATH=C:\models\all-MiniLM-L6-v2"
set "HF_CHAT_MODEL_LOCAL_PATH=C:\models\flan-t5-base"

echo Starting dashboard at http://127.0.0.1:8501
echo Close this window to stop the server.
echo.

".venv\Scripts\streamlit.exe" run "App\dashboard.py" --server.address 127.0.0.1 --server.port 8501

pause
