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

