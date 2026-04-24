# Local smoke-test: run one end-to-end query against the indexed corpus.
#
# Usage (from repo root):
#     .\scripts\run_sample_query.ps1
#     .\scripts\run_sample_query.ps1 "What cloud services are offered?"
#     .\scripts\run_sample_query.ps1 "backup policy" sales 3
#
# Args:
#   1. Query text   (default: "What cloud services are offered?")
#   2. Role         (default: "customer"; valid: customer | sales)
#   3. Top-N source rows to print (default: 5)
param(
    [string]$Query = "What cloud services are offered?",
    [string]$Role = "customer",
    [int]$Top = 5
)

# Anchor to the repo root no matter where the user runs from. This keeps
# relative paths (.venv, scripts, Data) valid even when invoked via a
# double-click or a different cwd.
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$env:EMBEDDING_MODEL_LOCAL_PATH = "C:\models\all-MiniLM-L6-v2"
$env:HF_CHAT_MODEL_LOCAL_PATH   = "C:\models\qwen2.5-3b-instruct"
$env:RERANKER_MODEL_LOCAL_PATH  = "C:\models\ms-marco-MiniLM-L6-v2"
$env:OFFLINE_ONLY               = "true"

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "ERROR: venv python not found at $python" -ForegroundColor Red
    Write-Host "Create the venv first, then install requirements.txt."   -ForegroundColor Yellow
    exit 1
}

# -u forces unbuffered stdout on top of the script's own flush=True so you
# see progress lines live instead of one silent wall of nothing for minutes.
& $python -u scripts\sample_query.py $Query --role $Role --top $Top
