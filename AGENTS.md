# AGENTS.md — Diretrizes para Agentes (Antigravity & Codex)

## 1. Fonte de Verdade
- `docs/PRD.md` (e `docs/PRD_FraudGuard_AI.md`) é a **fonte de verdade** dos requisitos, contratos REST, modelo de dados e regras de ML.
- Qualquer alteração de contrato de API ou schema de banco exige atualização documental simultânea.

## 2. Comandos de Setup, Execução e Testes

### Ambiente Local (Python)
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### Banco de Dados e Migrações (Alembic)
```bash
# Subir PostgreSQL local via Docker
docker compose up -d db

# Aplicar migrações
alembic -c backend/alembic.ini upgrade head

# Reverter última migração (teste de rollback)
alembic -c backend/alembic.ini downgrade -1
```

### Pipeline Offline de Machine Learning
```bash
# Treinar e avaliar modelos (Regressão Logística baseline + Random Forest)
python -m ml.src.training --csv-path creditcard_2023.csv --output-dir ml/artifacts

# Importar dataset e registrar modelos no banco
python scripts/seed_and_import.py --csv-path creditcard_2023.csv
```

### Executar Backend (FastAPI) e Frontend (Streamlit)
```bash
# Backend
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
streamlit run frontend/app.py --server.port 8501
```

### Testes e Qualidade (Pytest & Ruff)
```bash
ruff check .
pytest -v
```

## 3. Padrões de Arquitetura e Regras Inegociáveis
1. **Separação de camadas:**
   - `frontend/` (Streamlit) **nunca** acessa o banco de dados diretamente; comunica-se exclusivamente com `backend/` via HTTP REST (`/api/v1`).
   - `backend/app/api/v1/` lida apenas com HTTP, validação Pydantic e autenticação/autorização.
   - `backend/app/services/` concentra regras de negócio, atomicidade transacional, política de limiares e alertas.
   - `backend/app/repositories/` encapsula consultas SQLAlchemy 2.0.
   - `ml/src/` executa ingestão, pré-processamento, treinamento, avaliação e carregamento seguro de artefatos.
2. **Prevenção de Data Leakage:**
   - `id` é identificador e **jamais** entra como feature (`X`).
   - `Class` é o alvo (`y`) e **jamais** entra no payload de inferência.
   - Ordem canônica das 29 features (`creditcard-2023-v1`): `["V1", "V2", ..., "V28", "Amount"]`.
   - `StandardScaler` e qualquer transformador são ajustados (`fit`) **somente** no conjunto de treino (70%), dentro de um `Pipeline` do Scikit-learn.
3. **Segurança e Governança:**
   - Nunca commitar `.env`, credenciais reais do Supabase, `creditcard_2023.csv`, arquivos `.parquet` ou binários `.joblib`.
   - Sempre validar o checksum SHA-256 do artefato `.joblib` antes de carregá-lo em memória.
   - Operações administrativas (`POST /imports`, `POST /models/{id}/activate`) exigem autenticação de perfil Administrador.
   - Manter invariante de **apenas um modelo ativo por vez** no banco de dados (`uq_one_active_model`).

