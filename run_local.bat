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

REM Offline model paths -- all three must exist on disk before indexing or chat.
REM Run scripts\download_models.py once on a new machine to populate these.
REM The chat model path points at the folder containing the .gguf file; the
REM loader picks the first .gguf it finds inside.
set "OFFLINE_ONLY=true"
set "LOCAL_LLM_ONLY=true"
set "EMBEDDING_MODEL_LOCAL_PATH=C:\models\all-MiniLM-L6-v2"
REM Qwen 1.5B at Q4_K_M is the quality/speed sweet spot for this laptop.
REM The 3B model has better answers but takes 5-6 min per response on a
REM 2-core CPU -- swap the line below if you move to faster hardware.
set "HF_CHAT_MODEL_LOCAL_PATH=C:\models\qwen2.5-1.5b-instruct"
set "RERANKER_MODEL_LOCAL_PATH=C:\models\ms-marco-MiniLM-L6-v2"

echo Starting dashboard at http://127.0.0.1:8501
echo Network binding, telemetry, and error display are locked down in
echo .streamlit\config.toml so these flags don't need to be repeated here.
echo Close this window to stop the server.
echo.

".venv\Scripts\streamlit.exe" run "App\dashboard.py"

pause
