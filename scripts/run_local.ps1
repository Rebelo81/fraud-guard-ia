# Script para iniciar o Backend (FastAPI) e o Frontend (Streamlit) localmente
$ErrorActionPreference = "Stop"

Write-Host "1. Garantindo que o PostgreSQL (Docker) esta rodando..." -ForegroundColor Cyan
docker compose up -d db

Write-Host "2. Aplicando migracoes Alembic..." -ForegroundColor Cyan
.\.venv\Scripts\alembic.exe -c backend/alembic.ini upgrade head

Write-Host "3. Iniciando Backend FastAPI (http://localhost:8000) em nova janela..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\.venv\Scripts\uvicorn.exe backend.app.main:app --reload --host 0.0.0.0 --port 8000"

Write-Host "4. Iniciando Frontend Streamlit (http://localhost:8501)..." -ForegroundColor Green
.\.venv\Scripts\streamlit.exe run frontend/app.py --server.port 8501

