import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.components.api_client import FraudGuardApiClient, render_sidebar_controls

st.set_page_config(
    page_title="FraudGuard AI — Visão Geral",
    page_icon="🛡️",
    layout="wide",
)

headers = render_sidebar_controls()
api = FraudGuardApiClient(headers=headers)

st.title("🛡️ FraudGuard AI — Visão Geral Operacional")
st.markdown(
    "Painel gerencial de monitoramento de inferências, transações importadas e alertas para revisão humana."
)

try:
    health = api.get_health()
    overview = api.get_overview()
except Exception as exc:
    st.error(f"Não foi possível conectar à API FastAPI ({api.base_url}): {exc}")
    st.stop()

col_h1, col_h2, col_h3, col_h4 = st.columns(4)
col_h1.metric("Status da API", health.get("status", "unknown").upper())
col_h2.metric("Banco de Dados", health.get("database", "unknown"))
col_h3.metric("Modelo Ativo", health.get("active_model_version") or "Nenhum")
col_h4.metric("Motor de Inferência", "Pronto" if health.get("model_ready") else "Indisponível")

st.divider()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total de Transações", f"{overview['total_transactions']:,}")
k2.metric(
    "Fração de Fraude (Dataset Importado)",
    f"{overview['imported_fraud_fraction'] * 100:.2f}%",
    help="Proporção do rótulo histórico actual_class=1 nas transações importadas do CSV.",
)
k3.metric("Total de Predições", f"{overview['total_predictions']:,}")
k4.metric("Predições Acima do Limiar", f"{overview['predicted_fraud_count']:,}")
k5.metric("Alertas Abertos (Revisão)", f"{overview['open_alerts']:,}")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Alertas por Estado de Revisão Humana")
    status_map = overview.get("alerts_by_status", {})
    status_df = pd.DataFrame(
        [
            {"Estado": "open (Aberto)", "Quantidade": status_map.get("open", 0)},
            {"Estado": "reviewing (Em análise)", "Quantidade": status_map.get("reviewing", 0)},
            {"Estado": "confirmed (Confirmado)", "Quantidade": status_map.get("confirmed", 0)},
            {"Estado": "dismissed (Descartado)", "Quantidade": status_map.get("dismissed", 0)},
        ]
    )
    fig_alerts = px.bar(
        status_df,
        x="Estado",
        y="Quantidade",
        color="Estado",
        text_auto=True,
    )
    st.plotly_chart(fig_alerts, use_container_width=True)

with col_right:
    st.subheader("Predições por Período de Ingestão (`created_at`)")
    st.caption(
        "Nota: `created_at` representa a data/hora de ingestão no sistema, não o timestamp original da compra."
    )
    period_data = overview.get("predictions_by_ingestion_date", [])
    if period_data:
        period_df = pd.DataFrame(period_data)
        fig_period = px.bar(
            period_df,
            x="ingestion_date",
            y=["total", "flagged"],
            barmode="group",
            labels={
                "ingestion_date": "Data de Ingestão",
                "value": "Contagem de Predições",
                "variable": "Série",
            },
        )
        st.plotly_chart(fig_period, use_container_width=True)
    else:
        st.info("Nenhuma predição realizada ainda. Utilize a página Simulador ou Transações.")

active_model = overview.get("active_model")
if active_model:
    st.divider()
    st.subheader(
        f"Resumo do Modelo Ativo — `{active_model['version']}` ({active_model['algorithm']})"
    )
    metrics = active_model.get("metrics", {})
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Limiar Operacional", f"{active_model['threshold']:.2f}")
    m2.metric("Precision (Teste)", f"{metrics.get('precision', 0.0):.4f}")
    m3.metric("Recall (Teste)", f"{metrics.get('recall', 0.0):.4f}")
    m4.metric("F1-Score (Teste)", f"{metrics.get('f1', 0.0):.4f}")
    m5.metric("ROC-AUC (Teste)", f"{metrics.get('roc_auc', 0.0):.4f}")
    m6.metric("PR-AUC / Avg Precision", f"{metrics.get('average_precision', 0.0):.4f}")

