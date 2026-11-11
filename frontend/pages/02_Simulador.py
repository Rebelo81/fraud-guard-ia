import uuid

import streamlit as st

from frontend.components.api_client import FraudGuardApiClient, render_sidebar_controls

st.set_page_config(page_title="Simulador — FraudGuard AI", page_icon="🧪", layout="wide")
headers = render_sidebar_controls()
api = FraudGuardApiClient(headers=headers)

st.title("🧪 Simulador de Classificação em Tempo Real")
st.markdown(
    "Envie `Amount` e os 28 atributos numéricos anonimizados (`V1`…`V28`) para `POST /api/v1/predictions`. "
    "Conforme PRD §2.2, `id` e `Class` nunca entram no payload de inferência."
)

# Real profiles extracted from creditcard_2023.csv for rapid testing
PRESETS = {
    "Padrão Legítimo (Baixo Risco)": {
        "amount": 124.50,
        "V": {
            "V1": -0.05, "V2": -0.18, "V3": 0.65, "V4": -0.52, "V5": 0.12, "V6": -0.08,
            "V7": 0.45, "V8": -0.11, "V9": 0.32, "V10": 0.58, "V11": -0.45, "V12": 0.72,
            "V13": -0.10, "V14": 0.68, "V15": 0.05, "V16": 0.48, "V17": 0.42, "V18": 0.25,
            "V19": -0.15, "V20": -0.05, "V21": -0.12, "V22": -0.08, "V23": 0.02, "V24": 0.10,
            "V25": -0.04, "V26": -0.02, "V27": -0.05, "V28": -0.03,
        },
    },
    "Padrão Anômalo / Suspeito (Alto Risco)": {
        "amount": 6531.37,
        "V": {
            "V1": -1.85, "V2": 2.15, "V3": -2.40, "V4": 2.65, "V5": -1.60, "V6": -1.10,
            "V7": -2.35, "V8": 1.25, "V9": -1.95, "V10": -2.55, "V11": 2.45, "V12": -2.75,
            "V13": 0.15, "V14": -2.90, "V15": -0.20, "V16": -2.10, "V17": -2.65, "V18": -1.75,
            "V19": 1.10, "V20": 0.55, "V21": 0.85, "V22": 0.20, "V23": -0.15, "V24": -0.35,
            "V25": 0.25, "V26": 0.30, "V27": 0.65, "V28": 0.40,
        },
    },
}

preset_name = st.selectbox("Carregar Perfil de Simulação", options=list(PRESETS.keys()))
selected_preset = PRESETS[preset_name]

with st.form("simulator_form"):
    top_col1, top_col2 = st.columns(2)
    amount = top_col1.number_input(
        "Amount (Valor numérico da transação)",
        min_value=0.0,
        value=float(selected_preset["amount"]),
        step=10.0,
        format="%.2f",
    )
    request_id = top_col2.text_input(
        "Request ID (Opcional — chave de idempotência)",
        value=f"sim-{uuid.uuid4().hex[:8]}",
    )

    st.caption("Atributos Anonimizados `V1` a `V28`")
    v_inputs: dict[str, float] = {}
    cols = st.columns(4)
    for i in range(1, 29):
        key = f"V{i}"
        col = cols[(i - 1) % 4]
        v_inputs[key] = float(
            col.number_input(
                key,
                value=float(selected_preset["V"][key]),
                step=0.1,
                format="%.4f",
            )
        )

    submitted = st.form_submit_button("🚀 Executar Predição e Persistir", type="primary")

if submitted:
    payload = {
        "amount": amount,
        "features": v_inputs,
        "request_id": request_id.strip() if request_id.strip() else None,
    }
    try:
        res = api.create_prediction(payload)
        st.divider()
        st.subheader("Resultado da Inferência Persistida")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Escore de Risco (fraud_score)", f"{res['fraud_score']:.4f}")
        r2.metric("Limiar Ativo (threshold)", f"{res['threshold']:.2f}")
        r3.metric(
            "Classe Prevista",
            "1 (Suspeita)" if res["predicted_class"] == 1 else "0 (Legítima)",
        )
        r4.metric("Modelo / Schema", f"{res['model_version']} ({res['feature_schema_version']})")

        if res["requires_review"]:
            st.warning(
                f"🚨 **Alerta para Revisão Humana Gerado!** "
                f"`alert_id`: `{res.get('alert_id')}` | `prediction_id`: `{res['prediction_id']}`"
            )
        else:
            st.success(
                f"✅ Transação abaixo do limiar operacional. `prediction_id`: `{res['prediction_id']}`"
            )
        st.json(res)
    except Exception as exc:
        st.error(f"Erro na predição: {exc}")

