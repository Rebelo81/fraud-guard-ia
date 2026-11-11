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

## 3. Comparação Explícita de Abordagens e Resultados Reais Medidos (`creditcard_2023.csv` — Teste Intocado 15% = 85.295 linhas)

Para fundamentar a escolha arquitetural com rigor de Engenharia de Software e Ciência de Dados, comparamos quatro alternativas no mesmo problema:
1. **Motor de Regras Heurísticas Determinísticas** (`evaluate_rule_based_heuristic` em `ml/src/evaluation.py`): regra estática baseada nos maiores desvios univariados (`V14 < -2.5 OR (V4 > 2.0 AND V12 < -2.0) OR Amount > 15000`).
2. **Baseline Supervisionado Linear (`0.9.0-baseline` — `LogisticRegression`)**: `Pipeline(StandardScaler, LogisticRegression)`.
3. **Modelo Candidato Ensemble (`1.0.0` — `RandomForestClassifier` — Escolhido)**: `Pipeline(StandardScaler, RandomForestClassifier)`.
4. **Abordagem via LLMs (IA Generativa)**: analisada arquiteturalmente e descartada para o caminho crítico de inferência numérica tabular (ADR-004).

### 3.1 Tabela Comparativa Quantitativa e de Engenharia (Custo, Eficácia e Facilidade de Implementação)

| Métrica / Critério de Engenharia | 1. Regras Heurísticas Estáticas | 2. Baseline `0.9.0-baseline` (`LogisticRegression`) | 3. Candidato Ativo `1.0.0` (`RandomForestClassifier`) | 4. LLM / IA Generativa (Referência Arquitetural) |
|---|---|---|---|---|
| **Limiar Operacional** | `0.35` (fixo por regra) | `0.45` (otimizado na validação) | `0.37` (otimizado na validação) | N/A (prompt textual sem calibração nativa) |
| **Precision (Classe 1 — Teste)** | `0.9781` | `0.973051` | **`0.995557`** | Baixa/Instável em 28 floats PCA anonimizados |
| **Recall (Classe 1 — Teste)** | `0.7642` (perde 23,6% das fraudes) | `0.955870` | **`0.993036`** | Baixa reprodutibilidade |
| **F1-Score (Classe 1 — Teste)** | `0.8580` | `0.964384` | **`0.994295`** | Inferior a modelos tabulares supervisionados |
| **ROC-AUC (Teste)** | `0.8735` | `0.993445` | **`0.999826`** | Não produz curva contínua confiável |
| **Average Precision / PR-AUC** | `0.8391` | `0.994513` | **`0.999811`** | Não aplicável |
| **Matriz de Confusão (TN / FP / FN / TP)** | `41.918 / 730 / 10.056 / 32.591` | `41.519 / 1.129 / 1.882 / 40.765` | **`42.459 / 189 / 297 / 42.350`** | Alta variância entre execuções |
| **Latência Unitária (p50 / p95)** | `< 0.05 ms` | `0.65 ms / 1.02 ms` | `16.99 ms / 21.79 ms` | `500 ms a 3.500 ms` (100x+ mais lento) |
| **Custo por 568.630 Inferências (FinOps)** | `~$0.00` (CPU local), alto OpEx de manutenção manual | `~$0.00` (CPU local, `< 1s` em batch) | `~$0.00` (CPU local, `~11s` para 568k linhas) | **US$ 1.137 a US$ 5.686** em tokens de API ou GPUs dedicadas |
| **Facilidade de Implementação e Governança** | Fácil no início, mas degrada em dezenas de `if/else` difíceis de manter e sem aprendizado automático. | Muito alta; excelente linha de base rápida e interpretável. | **Equilíbrio ótimo:** mesma simplicidade de deploy do Scikit-learn com redução de **83,3% nos Falsos Positivos** e **84,2% nos Falsos Negativos** em relação à Regressão Logística. | Baixa para este domínio: exige parsing defensivo, proteção contra alucinação/prompt injection e gestão de indisponibilidade externa. |

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

