import pandas as pd
import streamlit as st

from frontend.components.api_client import FraudGuardApiClient, render_sidebar_controls

st.set_page_config(page_title="Alertas — FraudGuard AI", page_icon="🚨", layout="wide")
headers = render_sidebar_controls()
api = FraudGuardApiClient(headers=headers)

st.title("🚨 Gestão e Revisão Humana de Alertas")
st.markdown(
    "Alertas gerados automaticamente para predições com `fraud_score >= threshold`. "
    "Permite ao analista revisar falsos alarmes (`dismissed`) ou confirmar suspeitas (`confirmed`)."
)

c1, c2, c3 = st.columns(3)
status_filter = c1.selectbox(
    "Filtrar por Status",
    options=["Todos", "open", "reviewing", "confirmed", "dismissed"],
)
page = int(c2.number_input("Página", min_value=1, value=1, step=1))
page_size = int(c3.selectbox("Tamanho da Página", options=[10, 20, 50], index=1))

try:
    res = api.list_alerts(
        page=page,
        page_size=page_size,
        status=None if status_filter == "Todos" else status_filter,
    )
except Exception as exc:
    st.error(f"Erro ao carregar alertas: {exc}")
    st.stop()

items = res.get("items", [])
total = res.get("total", 0)
st.caption(f"Total de alertas: **{total:,}**")

if not items:
    st.info("Nenhum alerta encontrado para o filtro selecionado.")
else:
    df = pd.DataFrame(
        [
            {
                "Alert ID": a["id"],
                "Status": a["status"],
                "Fraud Score": round(a["fraud_score"], 4),
                "Threshold": round(a["threshold"], 2),
                "Amount": a["amount"],
                "Rótulo Histórico": a.get("actual_class"),
                "Modelo": a["model_version"],
                "Notas": a.get("notes") or "",
                "Criado em": a["created_at"],
                "Revisado em": a.get("reviewed_at") or "-",
            }
            for a in items
        ]
    )
    st.dataframe(df, use_container_width=True)

    st.divider()
    st.subheader("📝 Atualizar Estado de Alerta")
    selected_id = st.selectbox("Selecione o Alerta", options=[a["id"] for a in items])
    current_alert = next(a for a in items if a["id"] == selected_id)

    with st.form("review_alert_form"):
        st.write(
            f"**Status Atual:** `{current_alert['status']}` | "
            f"**Score:** `{current_alert['fraud_score']:.4f}` | "
            f"**Amount:** `{current_alert['amount']}`"
        )
        new_status = st.selectbox(
            "Novo Status",
            options=["reviewing", "confirmed", "dismissed", "open"],
        )
        notes = st.text_area(
            "Parecer / Observações da Revisão Humana",
            value=current_alert.get("notes") or "",
        )
        save_btn = st.form_submit_button("Salvar Revisão", type="primary")

    if save_btn:
        try:
            updated = api.update_alert(selected_id, status=new_status, notes=notes)
            st.success(
                f"Alerta `{updated['id']}` atualizado para `{updated['status']}` em `{updated['reviewed_at']}`!"
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Transição inválida ou erro na atualização: {exc}")

