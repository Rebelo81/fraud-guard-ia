import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.components.api_client import FraudGuardApiClient, render_sidebar_controls

st.set_page_config(page_title="Modelos de ML — FraudGuard AI", page_icon="📊", layout="wide")
headers = render_sidebar_controls()
api = FraudGuardApiClient(headers=headers)

st.title("📊 Comparação de Modelos e Métricas Reais de Experimentos")
st.markdown(
    "Comparação entre **Regressão Logística (Baseline)** e **Random Forest (Candidato)** "
    "avaliados no conjunto de teste intocado (15%), com limiar selecionado na validação (15%)."
)

try:
    models = api.list_models()
except Exception as exc:
    st.error(f"Erro ao carregar modelos: {exc}")
    st.stop()

if not models:
    st.warning("Nenhum modelo registrado no banco de dados.")
    st.stop()

comp_rows = []
for m in models:
    metrics = m.get("metrics", {})
    cm = metrics.get("confusion_matrix", {})
    lat = metrics.get("latency", {})
    comp_rows.append(
        {
            "Versão": m["version"],
            "Algoritmo": m["algorithm"],
            "Ativo": "✅ Sim" if m["is_active"] else "Não",
            "Limiar": m["threshold"],
            "Precision": metrics.get("precision"),
            "Recall": metrics.get("recall"),
            "F1-Score": metrics.get("f1"),
            "ROC-AUC": metrics.get("roc_auc"),
            "PR-AUC (Avg Precision)": metrics.get("average_precision"),
            "TP": cm.get("tp"),
            "FP": cm.get("fp"),
            "TN": cm.get("tn"),
            "FN": cm.get("fn"),
            "Latência p50 (ms)": lat.get("single_prediction_p50_ms"),
            "Latência p95 (ms)": lat.get("single_prediction_p95_ms"),
            "SHA-256": m["artifact_sha256"][:16] + "...",
        }
    )

st.subheader("Tabela Comparativa (Conjunto de Teste Intocado — 15%)")
st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

st.divider()
selected_version = st.selectbox(
    "Inspecionar Detalhes da Versão",
    options=[m["version"] for m in models],
)
selected_model = next(m for m in models if m["version"] == selected_version)
m_metrics = selected_model.get("metrics", {})
cm = m_metrics.get("confusion_matrix", {"tn": 0, "fp": 0, "fn": 0, "tp": 0})

col_cm, col_pr = st.columns(2)

with col_cm:
    st.subheader(f"Matriz de Confusão — `{selected_version}`")
    z = [[cm.get("tn", 0), cm.get("fp", 0)], [cm.get("fn", 0), cm.get("tp", 0)]]
    fig_cm = px.imshow(
        z,
        x=["Previsto 0 (Legítima)", "Previsto 1 (Fraude)"],
        y=["Real 0 (Legítima)", "Real 1 (Fraude)"],
        text_auto=True,
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_pr:
    st.subheader(f"Curva Precision-Recall (Validação) — `{selected_version}`")
    thr_info = m_metrics.get("threshold_selection", {})
    pr_sample = thr_info.get("pr_curve_sample", [])
    if pr_sample:
        pr_df = pd.DataFrame(pr_sample)
        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(
                x=pr_df["recall"],
                y=pr_df["precision"],
                mode="lines+markers",
                name="Precision-Recall",
            )
        )
        fig_pr.update_layout(
            xaxis_title="Recall (Sensibilidade)",
            yaxis_title="Precision (Precisão)",
        )
        st.plotly_chart(fig_pr, use_container_width=True)
    if thr_info.get("justification"):
        st.caption(f"**Justificativa do Limiar:** {thr_info['justification']}")

st.divider()
st.subheader("⚖️ Comparação Arquitetural Explícita: Regras vs. ML Supervisionado vs. LLMs")
st.markdown(
    "Justificativa de Engenharia de Software e IA abrangendo **Custo (FinOps)**, **Eficácia (Métricas no Teste Intocado de 85.295 linhas e Latência)** e **Facilidade de Implementação/Manutenção**:"
)
arch_comparison_df = pd.DataFrame(
    [
        {
            "Abordagem": "1. Motor de Regras Heurísticas (if/else)",
            "Precision (Teste)": "0.9781",
            "Recall (Teste)": "0.7642 (10.056 FN)",
            "F1-Score (Teste)": "0.8580",
            "PR-AUC": "0.8391",
            "Latência p50": "< 0.05 ms",
            "Custo (FinOps / 568k tx)": "~$0 compute | Alto OpEx humano",
            "Facilidade e Manutenção": "Fácil no dia 1; degrada em regras conflitantes sem aprendizado multivariado.",
        },
        {
            "Abordagem": "2. Regressão Logística (0.9.0-baseline)",
            "Precision (Teste)": "0.9731",
            "Recall (Teste)": "0.9559 (1.882 FN)",
            "F1-Score (Teste)": "0.9644",
            "PR-AUC": "0.9945",
            "Latência p50": "0.65 ms",
            "Custo (FinOps / 568k tx)": "~$0 marginal (CPU leve)",
            "Facilidade e Manutenção": "Muito alta; excelente baseline linear interpretável, mas limitada em fronteiras não-lineares.",
        },
        {
            "Abordagem": "3. Random Forest (1.0.0 — Escolhido)",
            "Precision (Teste)": "0.9956 (189 FP)",
            "Recall (Teste)": "0.9930 (297 FN)",
            "F1-Score (Teste)": "0.9943",
            "PR-AUC": "0.9998",
            "Latência p50": "16.99 ms",
            "Custo (FinOps / 568k tx)": "~$0 marginal (CPU local)",
            "Facilidade e Manutenção": "Ótimo trade-off: mesma API Pipeline Scikit-learn, reduzindo >83% dos erros (FP e FN).",
        },
        {
            "Abordagem": "4. LLMs / IA Generativa (Descartado p/ Score)",
            "Precision (Teste)": "Instável em PCA numérico",
            "Recall (Teste)": "Não calibrado",
            "F1-Score (Teste)": "Inferior em dados tabulares",
            "PR-AUC": "N/A",
            "Latência p50": "500 a 3.500 ms",
            "Custo (FinOps / 568k tx)": "US$ 1.137 a US$ 5.686 / lote",
            "Facilidade e Manutenção": "Baixa/Frágil: exige parsing JSON defensivo, proteção contra alucinação e rate-limits.",
        },
    ]
)
st.dataframe(arch_comparison_df, use_container_width=True)

with st.expander("🛡️ Mapa Abrangente de Riscos de Implementação (Incluindo Riscos de LLMs, Software e Dados)"):
    risk_df = pd.DataFrame(
        [
            {
                "ID": "R-LLM-01",
                "Categoria": "Uso de LLMs (Aplicação)",
                "Risco / Falha na Implementação": "Alucinação semântica inventando significado para features PCA anonimizadas (V1..V28).",
                "Severidade": "Crítica",
                "Mitigação Implementada": "ADR-004: Proibição de LLMs para interpretar V1..V28; exibição restrita a escores e métricas empíricas.",
            },
            {
                "ID": "R-LLM-02",
                "Categoria": "Uso de LLMs (Auditoria)",
                "Risco / Falha na Implementação": "Não-determinismo entre chamadas ou atualizações silenciosas do modelo no provedor externo.",
                "Severidade": "Alta",
                "Mitigação Implementada": "Pipeline Scikit-learn determinístico (random_state=42) + verificação obrigatória de checksum SHA-256.",
            },
            {
                "ID": "R-LLM-03",
                "Categoria": "Uso de LLMs (SLA / FinOps)",
                "Risco / Falha na Implementação": "Latência >1.5s, timeouts HTTP 429 e explosão de custos por token em lotes de 568.630 transações.",
                "Severidade": "Alta",
                "Mitigação Implementada": "Inferência 100% local em CPU (0.65 ms a 16.99 ms) com custo marginal zero.",
            },
            {
                "ID": "R-LLM-04",
                "Categoria": "Uso de LLMs (Segurança)",
                "Risco / Falha na Implementação": "Prompt Injection indireto via campos livres (notes, request_id) manipulando triagem.",
                "Severidade": "Alta",
                "Mitigação Implementada": "Máquina de estados estática ALLOWED_ALERT_TRANSITIONS e validação Pydantic (extra='forbid').",
            },
            {
                "ID": "R-LLM-05",
                "Categoria": "Uso de LLMs (Eng. Software)",
                "Risco / Falha na Implementação": "Regressões ou data leakage introduzidos por geração assistida de código (Vibe Coding).",
                "Severidade": "Alta",
                "Mitigação Implementada": "Contratos travados no PRD, regras em AGENTS.md, linter Ruff e suíte Pytest em CI.",
            },
            {
                "ID": "R-SWE-01",
                "Categoria": "Banco de Dados / API",
                "Risco / Falha na Implementação": "Falha parcial gravando transação sem predição/alerta ou ativando dois modelos simultâneos.",
                "Severidade": "Crítica",
                "Mitigação Implementada": "Commit SQL atômico com rollback automático + índice parcial único uq_one_active_model.",
            },
            {
                "ID": "R-ML-01",
                "Categoria": "Ciência de Dados / ML",
                "Risco / Falha na Implementação": "Data leakage via coluna id/Class ou extrapolação da base 50%/50% para prevalência bancária real.",
                "Severidade": "Crítica",
                "Mitigação Implementada": "Exclusão forçada de id/Class em extract_X_y, StandardScaler restrito ao treino e avisos metodológicos.",
            },
        ]
    )
    st.dataframe(risk_df, use_container_width=True)

st.divider()
st.subheader("⚙️ Ativação Transacional de Modelo (Exclusivo Administrador)")
if not selected_model["is_active"]:
    if st.button(f"Ativar Versão `{selected_version}` como Modelo Oficial", type="primary"):
        try:
            activated = api.activate_model(selected_model["id"])
            st.success(f"Modelo `{activated['version']}` ativado com sucesso!")
            st.rerun()
        except Exception as exc:
            st.error(f"Falha ao ativar modelo (verifique perfil Administrador): {exc}")
else:
    st.info(f"A versão `{selected_version}` já é o modelo único ativo no momento.")

