import pandas as pd
import streamlit as st

from frontend.components.api_client import FraudGuardApiClient, render_sidebar_controls

st.set_page_config(page_title="Transações — FraudGuard AI", page_icon="💳", layout="wide")
headers = render_sidebar_controls()
api = FraudGuardApiClient(headers=headers)

st.title("💳 Consulta de Transações e Classificação Sob Demanda")
st.markdown(
    "Diferenciação explícita entre **rótulo histórico (`actual_class`)**, **previsão do modelo (`predicted_class`)** e **decisão de revisão (`requires_review`)**."
)

f1, f2, f3, f4 = st.columns(4)
page = int(f1.number_input("Página", min_value=1, value=1, step=1))
page_size = int(f2.selectbox("Registros por página", options=[10, 20, 50, 100], index=1))
class_filter_label = f3.selectbox(
    "Filtro por Rótulo Histórico (actual_class)",
    options=["Todos", "0 — Legítima", "1 — Fraude"],
)
source_filter_label = f4.selectbox(
    "Origem (source)",
    options=["Todas", "dataset", "manual", "api"],
)

actual_class = None
if class_filter_label.startswith("0"):
    actual_class = 0
elif class_filter_label.startswith("1"):
    actual_class = 1

source = None if source_filter_label == "Todas" else source_filter_label

try:
    result = api.list_transactions(
        page=page,
        page_size=page_size,
        actual_class=actual_class,
        source=source,
    )
except Exception as exc:
    st.error(f"Erro ao consultar transações: {exc}")
    st.stop()

items = result.get("items", [])
total = result.get("total", 0)
st.caption(f"Total de registros encontrados: **{total:,}** (Exibindo página {page})")

if not items:
    st.info("Nenhuma transação encontrada para os filtros selecionados.")
else:
    rows = []
    for tx in items:
        latest = tx.get("latest_prediction")
        rows.append(
            {
                "ID da Transação": tx["id"],
                "ID Origem (CSV)": tx.get("source_record_id"),
                "Origem": tx["source"],
                "Amount": tx["amount"],
                "Rótulo Histórico (actual_class)": (
                    "Desconhecido"
                    if tx.get("actual_class") is None
                    else f"{tx['actual_class']} ({'Fraude' if tx['actual_class'] == 1 else 'Legítima'})"
                ),
                "Última Previsão (predicted_class)": (
                    "Não classificada"
                    if not latest
                    else f"{latest['predicted_class']} (Score={latest['fraud_score']:.4f})"
                ),
                "Ingestão (created_at)": tx["created_at"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()
    st.subheader("🔍 Inspecionar Detalhes e Executar Inferência")
    selected_tx_id = st.selectbox(
        "Selecione uma transação da página atual",
        options=[tx["id"] for tx in items],
    )
    selected_tx = next(tx for tx in items if tx["id"] == selected_tx_id)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.json(
            {
                "id": selected_tx["id"],
                "source": selected_tx["source"],
                "source_record_id": selected_tx.get("source_record_id"),
                "amount": selected_tx["amount"],
                "actual_class": selected_tx.get("actual_class"),
                "ingested_at": selected_tx["created_at"],
                "features_V1_to_V28": selected_tx["features"],
            },
            expanded=False,
        )
    with c2:
        if st.button("⚡ Classificar esta Transação via API", type="primary"):
            try:
                pred = api.create_prediction({"transaction_id": selected_tx["id"]})
                st.success(f"Predição registrada! ID: `{pred['prediction_id']}`")
                st.metric("Fraud Score", f"{pred['fraud_score']:.4f}")
                st.metric("Limiar Aplicado", f"{pred['threshold']:.2f}")
                st.write(
                    f"**Classe Prevista (`predicted_class`):** `{pred['predicted_class']}` | "
                    f"**Requer Revisão Humana (`requires_review`):** `{pred['requires_review']}`"
                )
                if pred.get("alert_id"):
                    st.warning(f"🚨 Alerta aberto gerado: `{pred['alert_id']}`")
            except Exception as exc:
                st.error(f"Falha ao classificar transação: {exc}")

