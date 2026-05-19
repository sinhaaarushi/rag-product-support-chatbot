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
REM Must match scripts\download_models.py (Qwen2.5-3B-Instruct-Q4_K_M.gguf).
REM If you use a different .gguf folder, change this path to that folder.
set "HF_CHAT_MODEL_LOCAL_PATH=C:\models\qwen2.5-3b-instruct"
set "RERANKER_MODEL_LOCAL_PATH=C:\models\ms-marco-MiniLM-L6-v2"

REM Pre-launch: materialise App\static\pdfs\ so Streamlit's static file
REM handler can serve the PDFs that the source chips link to. Streamlit
REM checks for this folder at *server start*, before the app script runs
REM -- if we only created it from inside dashboard.py, the first few
REM clicks would 404 until a browser session had connected.
".venv\Scripts\python.exe" -m App.pdf_mirror
if errorlevel 1 (
    echo WARNING: PDF mirror step failed. Source-chip links will 404.
)

echo Starting dashboard -- URL: http://127.0.0.1:8501
echo.
echo IMPORTANT: Browser does NOT open automatically (server.headless is true in
echo .streamlit\config.toml so a random browser profile isn't launched). Opening
echo the link for you after a short delay so the server is ready first.
echo You can bookmark that URL later. Close this window to stop the server.
echo.

REM Streamlit listens on localhost only; spawn the default browser when the socket is up.
start "" cmd /c "timeout /t 6 /nobreak >nul && start http://127.0.0.1:8501/"

".venv\Scripts\streamlit.exe" run "App\dashboard.py"
if errorlevel 1 (
    echo.
    echo If you saw ^"Port 8501 is not available^", another process is using it.
    echo Close the other terminal or browser tab serving Streamlit, or change
    echo server.port in .streamlit\config.toml , then try again.
)

pause
