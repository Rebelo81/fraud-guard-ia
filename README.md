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

## 2. Análise Comparativa de Alternativas: Regras vs. Modelos de ML vs. LLMs

Durante a análise do problema e desenho de software do **FraudGuard AI**, quatro abordagens arquiteturais foram avaliadas explicitamente sob três critérios de engenharia: **Custo (FinOps / TCO)**, **Eficácia (Capacidade Preditiva, Latência e Determinismo)** e **Facilidade de Implementação, Manutenção e Governança**.

### 2.1 Comparação Quantitativa e Arquitetural (Medida no Conjunto de Teste Intocado — 85.295 Transações)

| Dimensão / Critério | Alternativa A: Motor de Regras Determinísticas (`if/else` Heurístico) | Alternativa B: ML Supervisionado Linear (`LogisticRegression` — Baseline) | Alternativa C: ML Supervisionado Ensemble (`RandomForestClassifier` — Escolhido) | Alternativa D: Modelos de Linguagem de Grande Porte (LLMs / IA Generativa) |
|---|---|---|---|---|
| **Mecanismo de Decisão** | Limiares estáticos sobre variáveis isoladas (`V14 < -2.5 OR (V4 > 2.0 AND V12 < -2.0) OR Amount > 15000`). | Combinação linear ponderada das 29 features padronizadas (`StandardScaler` + função sigmoide). | Ensemble de árvores de decisão capturando interações não-lineares multivariadas entre `V1..V28` e `Amount`. | Serialização textual das 29 features em prompt para inferência *zero-shot*/*few-shot* via API ou modelo generativo. |
| **Eficácia Preditiva (Teste = 85.295 linhas)** | **Baixa/Moderada:**<br>• `Precision`: `0,9781`<br>• `Recall`: `0,7642` (**perde 10.056 fraudes**)<br>• `F1-Score`: `0,8580`<br>• `PR-AUC`: `0,8391` | **Alta (Baseline Linear):**<br>• `Precision`: `0,9731`<br>• `Recall`: `0,9559` (1.882 FN)<br>• `F1-Score`: `0,9644`<br>• `PR-AUC`: `0,9945` | **Muito Alta (Modelo Ativo):**<br>• `Precision`: **`0,9956`** (apenas 189 FP)<br>• `Recall`: **`0,9930`** (apenas 297 FN)<br>• `F1-Score`: **`0,9943`**<br>• `PR-AUC`: **`0,9998`** | **Inadequada para vetores PCA anonimizados:**<br>Baixa discriminação em 28 floats contínuos sem semântica textual e ausência de probabilidades calibradas (`predict_proba`). |
| **Latência e Throughput (SLA)** | **Ultra-rápida:** `< 0,05 ms` por transação. | **Muito rápida:** `p50 = 0,65 ms`, `p95 = 1,02 ms` (`0,94 ms` para lote de 1.000). | **Rápida e compatível com tempo real:** `p50 = 16,99 ms`, `p95 = 21,79 ms` (`19,96 ms` para lote de 1.000). | **Proibitiva no caminho crítico:** `500 ms a 3.500 ms` por chamada, sujeita a *timeouts* e *rate limits* (HTTP 429). |
| **Custo Operacional e Infraestrutura (FinOps)** | **Infra:** `~$0,00` (CPU local).<br>**Manutenção (OpEx):** **Alto custo humano** para revisar conflitos de regras manualmente. | **Infra:** `~$0,00` marginal por predição; roda em container CPU leve (`< 256 MB` RAM).<br>**Treino:** `~2 segundos`. | **Infra:** `~$0,00` marginal por predição; roda em CPU multi-core padrão sem necessidade de GPU.<br>**Treino:** `~10 segundos`. | **Muito Alto:** US$ 1.137 a US$ 5.686 por lote de 568.630 transações em APIs comerciais, ou alto custo de GPUs dedicadas. |
| **Facilidade de Implementação e Manutenção** | Fácil na primeira versão, porém degrada rapidamente em código frágil ("espaguete de regras") sem capacidade de aprendizado automático. | Muito simples (`Pipeline` Scikit-learn), determinística, excelente como linha de base comparativa. | **Equilíbrio ideal:** mesma facilidade de implementação e encapsulamento REST do Scikit-learn, reduzindo **83,3% dos Falsos Positivos** e **84,2% dos Falsos Negativos** frente ao modelo linear. | **Baixa e frágil:** exige parsing defensivo de JSON, engenharia de prompt, proteção contra *prompt injection* e não garante reprodutibilidade estrita entre versões do provedor. |

### 2.2 Justificativa da Arquitetura Híbrida Adotada (Regras de Contrato + ML Supervisionado + Revisão Humana)
1. **Por que não usar apenas Regras Fixas?** Regras estáticas univariadas deixaram escapar **10.056 transações fraudulentas** no conjunto de teste (`Recall = 76,42%`), pois fraudes reais combinam perturbações simultâneas sutis em múltiplos componentes `V1..V28` que nenhuma regra manual `if/else` consegue mapear sem sobreajuste.
2. **Por que escolher `RandomForestClassifier` sobre `LogisticRegression`?** Embora a Regressão Logística seja extremamente rápida (`0,65 ms`) e atinja `F1 = 0,9644`, sua fronteira de decisão linear gerou **1.129 falsos alarmes (FP)** e **1.882 fraudes não detectadas (FN)** no teste. O `RandomForestClassifier` (`v1.0.0`) captura não-linearidades e interações entre atributos com o mesmo custo de infraestrutura local, reduzindo os falsos positivos para **189 (-83,3%)** e os falsos negativos para **297 (-84,2%)**, alcançando `F1 = 0,9943` e `PR-AUC = 0,9998`.
3. **Onde Regras Determinísticas permanecem essenciais na arquitetura?** As regras determinísticas foram posicionadas onde são insubstituíveis:
   - **Camada de Entrada e Qualidade (`Pydantic v2` + `Ingestion`):** validação estrita das 28 variáveis `V1..V28` + `Amount >= 0`, bloqueio de `id` e `Class`, rejeição de `NaN`/`Inf` e deduplicação por `request_id` e `sha256`.
   - **Camada de Governança e Decisão (`Domain Services`):** aplicação determinística do limiar (`predicted_class = 1 if fraud_score >= threshold else 0`), abertura atômica de alerta e máquina de estados (`ALLOWED_ALERT_TRANSITIONS`) para revisão humana.

---

## 3. Mapa Abrangente de Riscos de Implementação (Software, Dados, ML e Uso de LLMs)

A tabela abaixo consolida o **Mapa de Riscos** do projeto, detalhando falhas potenciais de implementação — com destaque especial para os **riscos associados ao uso de LLMs** (tanto na aplicação quanto no desenvolvimento assistido por IA) — e as mitigações técnicas implementadas no código:

| ID | Categoria | Risco e Falha Potencial na Implementação | Prob. | Impacto | Severidade | Mitigação Técnica e Controle Implementado no Repositório |
|---|---|---|---|---|---|---|
| **R-LLM-01** | **Uso de LLMs na Aplicação** | **Alucinação Explicativa sobre Variáveis Anonimizadas (`V1..V28`):** Usar um LLM para gerar justificativas de fraude faz o modelo inventar semânticas falsas para componentes PCA anonimizados (ex.: dizer que `V14` representa "compra internacional"), enganando o analista humano. | Alta | Crítico | **Crítica** | **ADR-004:** LLMs não são utilizados para inventar explicações sobre `V1..V28`. O dashboard e a API expõem apenas evidências empíricas reais (`fraud_score`, `threshold`, versão do modelo e métricas do teste). |
| **R-LLM-02** | **Uso de LLMs na Aplicação** | **Não-Determinismo e Violação de Reprodutibilidade (RNF-01):** LLMs podem retornar escores ou classificações diferentes para o mesmo payload devido a amostragem estocástica ou atualizações silenciosas do modelo no provedor. | Alta | Alto | **Alta** | Inferência 100% determinística via Scikit-learn (`random_state=42`) com artefato `.joblib` imutável validado por checksum **SHA-256** em `LoadedInferenceEngine` (`ml/src/inference.py`). |
| **R-LLM-03** | **Uso de LLMs (SLA / Resiliência)** | **Latência Elevada, Timeouts e Rate Limits (HTTP 429):** Acoplar `POST /api/v1/predictions` a uma API externa de LLM elevaria a latência para `>1.500 ms` e causaria falhas em cascata sob carga. | Média | Alto | **Alta** | Execução em memória local via CPU (`p50 = 0,65 ms` no baseline e `16,99 ms` no Random Forest), sem chamadas de rede externas durante a inferência. |
| **R-LLM-04** | **Uso de LLMs (Segurança / FinOps)** | **Prompt Injection Indireto, Vazamento de Dados e Explosão de Custos (*Denial of Wallet*):** Textos livres em `notes`/`request_id` manipulando instruções de um agente LLM ou envio de 568.630 transações gerando custos de milhares de dólares em tokens. | Média | Alto | **Alta** | Máquina de estados determinística (`ALLOWED_ALERT_TRANSITIONS`), validação Pydantic estrita (`extra="forbid"`) e processamento 100% local sem custo por token. |
| **R-LLM-05** | **Uso de LLMs no Desenvolvimento (Eng. Software)** | **Regressões Silenciosas, Alucinação de APIs ou Data Leakage Gerado por IA (*Vibe Coding*):** Assistentes de código baseados em LLM podem ajustar `StandardScaler` fora do treino, esquecer rollback SQL ou divergir do contrato REST. | Média | Alto | **Alta** | Diretrizes mandatórias em `AGENTS.md`, contrato único em `docs/PRD.md`, linter `ruff` e suíte `pytest` (`test_unit.py`, `test_ml.py`, `test_integration_api.py`) executada no GitHub Actions CI. |
| **R-SWE-01** | **Engenharia de Software (Banco)** | **Persistência Parcial em Falha de Inferência:** Gravar a transação no banco, mas falhar antes de gravar a predição ou o alerta, gerando estado inconsistente. | Média | Crítico | **Crítica** | Transação SQL atômica em `PredictionService.create_prediction` (`db.commit()` único para `Transaction` + `Prediction` + `FraudAlert` e `db.rollback()` em caso de erro, testado em `test_model_failure_rolls_back_and_leaves_no_partial_data`). |
| **R-SWE-02** | **Engenharia de Software (Segurança)** | **Carregamento de Artefato `.joblib` Adulterado ou Incompatível:** Desserialização de binário corrompido ou com ordem de features divergente. | Baixa | Crítico | **Alta** | Verificação obrigatória do hash **SHA-256** e de `feature_schema_version = "creditcard-2023-v1"` antes do `joblib.load`; ausência de endpoint de upload arbitrário de binários. |
| **R-SWE-03** | **Engenharia de Software (Concorrência)** | **Múltiplos Modelos Ativos Simultaneamente:** Condição de corrida ao ativar versões concorrentes em `ml_models`. | Baixa | Alto | **Média** | Índice parcial único no banco (`CREATE UNIQUE INDEX uq_one_active_model ON ml_models (is_active) WHERE is_active = TRUE`) e ativação transacional em `ModelService.activate_model`. |
| **R-SWE-04** | **Engenharia de Software (Memória/IO)** | **Exaustão de Memória (OOM) ao Importar 568.630 Linhas (~325 MB):** Leitura monolítica do CSV derrubando o processo da API. | Alta | Alto | **Alta** | Leitura em chunks (`chunksize=10_000`), processamento em background (`202 Accepted`) e inserção idempotente em lotes (`ON CONFLICT DO NOTHING` + `sha256` único). |
| **R-ML-01** | **Ciência de Dados / ML** | **Vazamento de Dados (*Data Leakage*) via `id` ou `Class`:** O atributo `id` codifica a ordenação do dataset e infla artificialmente as métricas se usado no treino. | Média | Crítico | **Crítica** | Exclusão estrita de `id` e `Class` em `extract_X_y` (`ml/src/preprocessing.py`), vetor fixo de 29 features `FEATURES` independente da ordem JSON e testes automatizados dedicados. |
| **R-ML-02** | **Ciência de Dados / ML** | **Extrapolação Indevida da Base Balanceada (50%/50%) e Ausência de Timestamp:** Supor que a precisão medida no CSV se repete na prevalência bancária real (`< 0,2%`) ou tratar `created_at` como data da compra. | Alta | Alto | **Alta** | Documentação transparente das limitações metodológicas, separação semântica clara entre data de ingestão (`created_at`), rótulo histórico (`actual_class`), previsão (`predicted_class`) e decisão humana (`status`). |

---

## 4. Dataset e Termos de Uso

- **Arquivo:** `creditcard_2023.csv` (**568.630 linhas**, **31 colunas**: `id`, `V1`…`V28`, `Amount`, `Class`).
- **Origem:** Conjunto público *Credit Card Fraud Detection Dataset 2023* (atributos transformados via PCA e anonimizados). Por boas práticas de versionamento e governança (PRD §2.2), **o arquivo CSV, os arquivos `.parquet` e os artefatos `.joblib` são ignorados no Git (`.gitignore`)**.

---

## 5. Como Executar Localmente

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

## 6. Roteiro de Demonstração Ponta a Ponta (Sprint 4)

1. **Importar e Reconciliar (`Importações`):** Com o perfil **Administrador** selecionado na barra lateral do Streamlit, acione `POST /api/v1/imports` ou consulte a reconciliação das 568.630 linhas (`0: 284.315`, `1: 284.315`).
2. **Comparar Modelos (`Modelos`):** Compare as métricas de teste intocado entre `0.9.0-baseline` (`LogisticRegression`) e `1.0.0` (`RandomForestClassifier`), inspecionando Matriz de Confusão, Curva Precision-Recall e latência p50/p95.
3. **Classificar Transação (`Transações` ou `Simulador`):** Selecione uma transação importada ou use os perfis pré-configurados no Simulador para enviar `Amount` + `V1..V28` a `POST /api/v1/predictions`.
4. **Revisar Alerta (`Alertas`):** Quando `fraud_score >= threshold`, um alerta com status `open` é persistido na mesma transação SQL. Acesse a aba **Alertas** e atualize o parecer humano para `confirmed` ou `dismissed`.

---

## 7. Qualidade e Testes Automatizados

```powershell
ruff check .
pytest -v
```

