# Pipeline de Machine Learning e Relatório Experimental — FraudGuard AI

## 1. Perfil do Dataset e Restrições Metodológicas

| Propriedade | Valor Verificado |
|---|---|
| Arquivo | `creditcard_2023.csv` |
| Total de Registros | **568.630** |
| Total de Colunas | **31** (`id`, `V1`…`V28`, `Amount`, `Class`) |
| Valores Ausentes (NaN) | **0** |
| Distribuição de Classes (`Class`) | **`0` (Legítima): 284.315 (50,0%)**; **`1` (Fraude): 284.315 (50,0%)** |
| Identificadores (`id`) Duplicados | **0** |

### 1.1 Cuidados Inegociáveis Contra Vazamento de Dados (*Data Leakage*)
1. **Exclusão de `id` e `Class`:**
   - `id` é um identificador sequencial do dataset que apresenta forte correlação artificial com a ordenação das classes no arquivo bruto. **Jamais é incluído em `X`**.
   - `Class` é a variável-alvo (`y`) e **nunca entra no payload de inferência**.
   - A lista canônica versionada (`feature_schema_version = "creditcard-2023-v1"`) contém exatamente **29 atributos** em ordem fixa:
     ```python
     FEATURES = ["V1", "V2", ..., "V28", "Amount"]
     ```
2. **Prevalência Artificial (50% / 50%):**
   - Ao contrário de cenários bancários reais (onde a prevalência de fraude costuma ser < 0,2%), o dataset `creditcard_2023.csv` é balanceado artificialmente em 50%/50%.
   - Portanto, Precision, Taxa de Falsos Positivos e calibração probabilística **não devem ser extrapoladas diretamente para produção bancária real**.
3. **Ausência de Marcação Temporal:**
   - Não há coluna de data/hora real da transação. Utiliza-se divisão estratificada por classe (`70%` treino, `15%` validação, `15%` teste com `random_state=42`).
4. **Ajuste de Transformadores Restrito ao Treino:**
   - O `StandardScaler` compõe o `sklearn.pipeline.Pipeline` de ambos os algoritmos e é ajustado (`fit`) **exclusivamente nos 70% de treino** (`398.041` linhas).

---

## 2. Metodologia Experimental

1. **Divisão dos Dados (`stratified_train_val_test_split`):**
   - **Treino (70%):** `398.041` registros (`199.020` classe 0 / `199.021` classe 1)
   - **Validação (15%):** `85.294` registros (`42.647` classe 0 / `42.647` classe 1)
   - **Teste Intocado (15%):** `85.295` registros (`42.648` classe 0 / `42.647` classe 1)
2. **Modelos Comparados:**
   - **Baseline (`0.9.0-baseline`):** `Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42))])`
   - **Candidato (`1.0.0`):** `Pipeline([("scaler", StandardScaler()), ("classifier", RandomForestClassifier(n_estimators=80, max_depth=16, min_samples_leaf=4, class_weight="balanced_subsample", random_state=42))])`
3. **Seleção do Limiar Operacional (`select_optimal_threshold`):**
   - O limiar de decisão (`threshold`) é selecionado **estritamente na partição de validação (15%)**, maximizando o F1-score em uma grade de candidatos `[0.05, 0.95]`.
   - Após congelado, o limiar é aplicado **uma única vez** na partição de teste intocada (15%).

---

## 3. Resultados Reais Medidos (`creditcard_2023.csv` — Partição de Teste Intocada 15% = 85.295 linhas)

| Métrica / Propriedade | Baseline `0.9.0-baseline` (`LogisticRegression`) | Candidato Ativo `1.0.0` (`RandomForestClassifier`) |
|---|---|---|
| **Limiar Selecionado (Validação)** | `0.45` | `0.37` |
| **Precision (Classe 1)** | `0.973051` | `0.995557` |
| **Recall (Classe 1)** | `0.955870` | `0.993036` |
| **F1-Score (Classe 1)** | `0.964384` | `0.994295` |
| **ROC-AUC** | `0.993445` | `0.999826` |
| **Average Precision (PR-AUC)** | `0.994513` | `0.999811` |
| **Matriz de Confusão (TN / FP / FN / TP)** | `41.519 / 1.129 / 1.882 / 40.765` | `42.459 / 189 / 297 / 42.350` |
| **Latência Predição Unitária (p50 / p95)** | `0.65 ms / 1.02 ms` | `16.99 ms / 21.79 ms` |
| **SHA-256 do Artefato (`.joblib`)** | `26611061b028350b...` | `fc117d35092b43fb...` |

---

## 4. Registry Simples e Verificação de Artefatos

Cada execução gera:
- `ml/artifacts/0.9.0-baseline/pipeline.joblib` e `metadata.json`
- `ml/artifacts/1.0.0/pipeline.joblib` e `metadata.json`
- `ml/artifacts/experiment_comparison.json`

Antes de servir inferências em `LoadedInferenceEngine`, o backend verifica:
1. Existência física do arquivo `.joblib` em diretório controlado.
2. Igualdade exata do checksum **SHA-256** calculado em tempo de carregamento contra o `artifact_sha256` registrado.
3. Compatibilidade de `feature_schema_version` (`creditcard-2023-v1`) e das 29 colunas `FEATURES`.

