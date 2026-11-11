import pandas as pd
import streamlit as st

from frontend.components.api_client import FraudGuardApiClient, render_sidebar_controls

st.set_page_config(page_title="Importações — FraudGuard AI", page_icon="📥", layout="wide")
headers = render_sidebar_controls()
api = FraudGuardApiClient(headers=headers)

st.title("📥 Gestão de Importações de Dataset e Reconciliação (Administrador)")
st.markdown(
    "Execução controlada da importação em lotes do arquivo `creditcard_2023.csv` com verificação de SHA-256 "
    "e reconciliação de registros (`row_count` vs `inserted_rows`)."
)

with st.form("import_csv_form"):
    c1, c2, c3, c4 = st.columns(4)
    filename = c1.text_input("Arquivo CSV", value="creditcard_2023.csv")
    chunksize = int(c2.number_input("Chunksize", min_value=500, max_value=20000, value=10000, step=500))
    max_rows_val = int(
        c3.number_input(
            "Limite de Linhas (0 = todas as 568.630)",
            min_value=0,
            max_value=600000,
            value=0,
            step=5000,
        )
    )
    sync_mode = c4.checkbox("Aguardar conclusão síncrona", value=False)
    trigger = st.form_submit_button("Iniciar Importação Controlada", type="primary")

if trigger:
    try:
        res = api.create_import(
            filename=filename,
            chunksize=chunksize,
            max_rows=max_rows_val if max_rows_val > 0 else None,
            run_synchronously=sync_mode,
        )
        st.success(f"Importação registrada! ID: `{res['id']}` | Status: `{res['status']}`")
    except Exception as exc:
        st.error(f"Não foi possível iniciar importação: {exc}")

st.divider()
st.subheader("Histórico de Execuções de Importação")
if st.button("🔄 Atualizar Status das Importações"):
    st.rerun()

try:
    imports = api.list_imports()
    if not imports:
        st.info("Nenhuma importação registrada ainda.")
    else:
        st.dataframe(pd.DataFrame(imports), use_container_width=True)
except Exception as exc:
    st.warning(
        f"Acesso restrito ao perfil Administrador para listar importações (`GET /api/v1/imports`): {exc}"
    )

