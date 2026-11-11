# Modelo de Dados e Persistência — FraudGuard AI

## 1. Visão Geral

O **FraudGuard AI** utiliza **PostgreSQL** (gerenciado pelo **Supabase** em nuvem ou container `postgres:16-alpine` via Docker Compose em desenvolvimento local), acessado via **SQLAlchemy 2.0** (`psycopg 3`) e versionado por migrações **Alembic** (`backend/alembic/versions/0001_initial_schema.py`).

---

## 2. Diagrama Entidade-Relacionamento (ERD)

```mermaid
erDiagram
    DATASET_IMPORTS ||--o{ TRANSACTIONS : "origina (em lotes)"
    TRANSACTIONS ||--o{ PREDICTIONS : "possui"
    ML_MODELS ||--o{ PREDICTIONS : "classifica"
    ML_MODELS ||--o{ TRAINING_RUNS : "gerado por"
    PREDICTIONS ||--o| FRAUD_ALERTS : "dispara (se score >= threshold)"

    DATASET_IMPORTS {
        uuid id PK
        text filename
        char_64 sha256 UK
        text status
        integer row_count
        integer inserted_rows
        text error_message
        timestamptz created_at
        timestamptz completed_at
    }

    TRANSACTIONS {
        uuid id PK
        uuid dataset_import_id FK
        bigint source_record_id
        double amount
        jsonb features
        smallint actual_class
        text source
        timestamptz created_at
    }

    ML_MODELS {
        uuid id PK
        text version UK
        text algorithm
        text feature_schema_version
        text artifact_uri
        char_64 artifact_sha256
        jsonb metrics
        double threshold
        boolean is_active
        timestamptz created_at
        timestamptz activated_at
    }

    TRAINING_RUNS {
        uuid id PK
        uuid model_id FK
        char_64 dataset_sha256
        integer split_seed
        jsonb parameters
        jsonb metrics
        timestamptz started_at
        timestamptz finished_at
        text status
    }

    PREDICTIONS {
        uuid id PK
        uuid transaction_id FK
        uuid model_id FK
        double fraud_score
        double threshold
        smallint predicted_class
        timestamptz created_at
        text request_id UK
    }

    FRAUD_ALERTS {
        uuid id PK
        uuid prediction_id FK, UK
        text status
        text notes
        timestamptz reviewed_at
        timestamptz created_at
    }
```

---

## 3. Invariantes, Índices e Regras de Integridade

1. **Modelo Único Ativo (`uq_one_active_model`):**
   - Índice parcial único:
     ```sql
     CREATE UNIQUE INDEX uq_one_active_model ON ml_models (is_active) WHERE is_active = TRUE;
     ```
   - Garante no nível do SGBD que jamais existam dois modelos simultaneamente ativos. A troca de modelo ativo ocorre de forma transacional em `ModelService.activate_model`.
2. **Idempotência de Importação (`dataset_imports` e `transactions`):**
   - `dataset_imports.sha256` é `UNIQUE`, impedindo reimportação acidental do mesmo arquivo CSV (`409 Conflict`).
   - `UNIQUE (dataset_import_id, source_record_id)` em `transactions` garante que a importação em lotes (`ON CONFLICT DO NOTHING`) possa ser retomada sem duplicar linhas.
3. **Atomicidade na Inferência (`POST /api/v1/predictions`):**
   - A inserção de `Transaction`, `Prediction` e `FraudAlert` (quando `predicted_class = 1`) ocorre dentro de uma única transação SQL. Qualquer erro faz `rollback` completo.
4. **Semântica Temporal (`created_at`):**
   - O dataset `creditcard_2023.csv` não possui timestamp real da compra. Todos os campos `created_at` representam estritamente o **momento de ingestão/registro** no banco.
5. **Conexão com Supabase:**
   - Aplicação utiliza `DATABASE_URL` (suportando Connection Pooler com `?sslmode=require`).
   - Migrações Alembic suportam `DIRECT_DATABASE_URL` para conexão direta na porta `5432`.

