# 🛡️ FraudGuard AI — Sistema Inteligente de Detecção de Fraudes

**Mini Capstone — Pós-Graduação em Engenharia de Inteligência Artificial / Fundamentos de IA**

O **FraudGuard AI** é uma aplicação demonstrativa e acadêmica de ponta a ponta para detecção de fraudes em transações financeiras utilizando aprendizado supervisionado clássico (**Scikit-learn**), API RESTful (**FastAPI**), persistência relacional auditável (**PostgreSQL / Supabase + SQLAlchemy 2 + Alembic**) e dashboard analítico (**Streamlit + Plotly**).

> **Aviso Importante (PRD §1 e §2.2):** Este sistema é estritamente acadêmico e demonstrativo. Trabalha com o dataset público anonimizado `creditcard_2023.csv` (taxa balanceada artificialmente em 50% por classe, sem marcação temporal real ou moeda confirmada). Não deve ser utilizado como sistema bancário certificado nem para autorizar, bloquear ou liquidar pagamentos reais.

---

## 1. Estrutura do Monorepositório

```text
fraudguard-ai/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # Rotas FastAPI REST (/api/v1)
│   │   ├── core/            # Config, segurança RBAC, exceções e logs JSON
│   │   ├── db/              # Engine SQLAlchemy 2.0 e sessão
│   │   ├── models/          # Entidades ORM e invariantes SQL
│   │   ├── schemas/         # Contratos Pydantic v2 (validação estrita)
│   │   ├── repositories/    # Consultas e agregações SQLAlchemy
│   │   ├── services/        # Regras de negócio, atomicidade e registry ML
│   │   └── main.py          # Entrypoint FastAPI + Swagger
│   ├── alembic/             # Migrações versionadas de schema
│   ├── tests/               # Testes unitários, ML e integração (Pytest)
│   └── Dockerfile
├── frontend/
│   ├── app.py               # Página principal: Visão Geral
│   ├── pages/               # Transações, Simulador, Alertas, Modelos, Importações
│   ├── components/          # Cliente HTTPX isolado (sem acesso direto ao banco)
│   └── Dockerfile
├── ml/
│   ├── data/                # Parquet derivado para treino (ignorado no Git)
│   ├── notebooks/           # 01_eda_and_model_comparison.ipynb
│   ├── src/
│   │   ├── ingestion.py     # Validação CSV em chunks, SHA-256 e carga em lotes
│   │   ├── preprocessing.py # Vetor canônico 29 features e split 70/15/15
│   │   ├── training.py      # Pipelines LogisticRegression e RandomForest
│   │   ├── evaluation.py    # Métricas, seleção de limiar e latência p50/p95
│   │   └── inference.py     # Motor de inferência com verificação SHA-256
│   └── artifacts/           # Binários .joblib versionados (ignorados no Git)
├── scripts/
│   └── seed_and_import.py   # Treino, registro de modelos no banco e carga em lotes
├── docs/
│   ├── PRD.md               # Especificação oficial do produto
│   ├── PRD_FraudGuard_AI.md # Documento original recebido
│   ├── ARCHITECTURE.md      # Diagrama e responsabilidades de camadas
│   ├── DATABASE.md          # ERD, índices e invariantes PostgreSQL
│   └── ML_PIPELINE.md       # Metodologia experimental e prevenção de vazamento
├── .github/workflows/ci.yml # Pipeline GitHub Actions (Ruff + Pytest)
├── AGENTS.md                # Guia operacional para Antigravity e Codex
├── .env.example             # Template seguro de variáveis de ambiente
├── docker-compose.yml       # Orquestração PostgreSQL + Backend + Frontend
├── pyproject.toml           # Dependências e configuração de ferramentas
└── README.md
```

---

## 2. Dataset e Termos de Uso

- **Arquivo:** `creditcard_2023.csv` (**568.630 linhas**, **31 colunas**: `id`, `V1`…`V28`, `Amount`, `Class`).
- **Origem:** Conjunto público *Credit Card Fraud Detection Dataset 2023* (atributos transformados via PCA e anonimizados). Por boas práticas de versionamento e governança (PRD §2.2), **o arquivo CSV, os arquivos `.parquet` e os artefatos `.joblib` são ignorados no Git (`.gitignore`)**.

---

## 3. Como Executar Localmente

### Opção A — Execução Rápida com Docker Compose (Recomendada)

1. Copie `.env.example` para `.env`:
   ```bash
   cp .env.example .env
   ```
2. Treine os modelos e gere os artefatos locais (ou execute via script de inicialização):
   ```bash
   docker compose up -d --build
   ```
3. Acesse:
   - **Swagger / OpenAPI:** http://localhost:8000/docs
   - **Healthcheck API:** http://localhost:8000/api/v1/health
   - **Dashboard Streamlit:** http://localhost:8501

### Opção B — Ambiente Virtual Python Local (Desenvolvimento / Supabase)

1. Crie o ambiente virtual e instale as dependências:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -e ".[dev]"
   ```
2. Configure `.env` apontando `DATABASE_URL` para seu banco PostgreSQL local ou projeto **Supabase**.
3. Aplique as migrações Alembic:
   ```powershell
   alembic -c backend/alembic.ini upgrade head
   ```
4. Execute o pipeline de treinamento ML, registre os modelos e importe o CSV em lotes:
   ```powershell
   python scripts/seed_and_import.py --csv-path creditcard_2023.csv
   ```
5. Inicie a API FastAPI e o Dashboard Streamlit:
   ```powershell
   uvicorn backend.app.main:app --reload --port 8000
   streamlit run frontend/app.py --server.port 8501
   ```

---

## 4. Roteiro de Demonstração Ponta a Ponta (Sprint 4)

1. **Importar e Reconciliar (`Importações`):** Com o perfil **Administrador** selecionado na barra lateral do Streamlit, acione `POST /api/v1/imports` ou consulte a reconciliação das 568.630 linhas (`0: 284.315`, `1: 284.315`).
2. **Comparar Modelos (`Modelos`):** Compare as métricas de teste intocado entre `0.9.0-baseline` (`LogisticRegression`) e `1.0.0` (`RandomForestClassifier`), inspecionando Matriz de Confusão, Curva Precision-Recall e latência p50/p95.
3. **Classificar Transação (`Transações` ou `Simulador`):** Selecione uma transação importada ou use os perfis pré-configurados no Simulador para enviar `Amount` + `V1..V28` a `POST /api/v1/predictions`.
4. **Revisar Alerta (`Alertas`):** Quando `fraud_score >= threshold`, um alerta com status `open` é persistido na mesma transação SQL. Acesse a aba **Alertas** e atualize o parecer humano para `confirmed` ou `dismissed`.

---

## 5. Qualidade e Testes Automatizados

```powershell
ruff check .
pytest -v
```

