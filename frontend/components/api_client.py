import os
from typing import Any

import httpx
import streamlit as st


def get_api_base_url() -> str:
    return os.getenv("STREAMLIT_API_BASE_URL", "http://localhost:8000/api/v1").rstrip("/")


def get_default_tokens() -> dict[str, str]:
    return {
        "analyst": os.getenv("ANALYST_API_TOKEN", "fg-dev-analyst-token-change-me"),
        "admin": os.getenv("ADMIN_API_TOKEN", "fg-dev-admin-token-change-me"),
    }


def render_sidebar_controls() -> dict[str, str]:
    st.sidebar.title("🛡️ FraudGuard AI")
    st.sidebar.caption("Sistema Demonstrativo de Detecção de Fraudes — Pós-IA")

    tokens = get_default_tokens()
    role_choice = st.sidebar.radio(
        "Perfil de Uso (RBAC)",
        options=["Analista", "Administrador"],
        index=0,
        help="Operações administrativas (importar CSV, ativar modelo) exigem credencial de Administrador.",
    )
    active_token = tokens["admin"] if role_choice == "Administrador" else tokens["analyst"]
    custom_token = st.sidebar.text_input(
        "Token de API (X-API-Key)",
        value=active_token,
        type="password",
    )

    st.sidebar.divider()
    st.sidebar.info(
        "**Aviso Metodológico (PRD §2.2):**\n"
        "- Base `creditcard_2023.csv` possui prevalência balanceada artificialmente (50% fraude / 50% legítima).\n"
        "- `created_at` refere-se ao momento de **ingestão**, não à data real da compra.\n"
        "- Sistema acadêmico: não bloqueia nem liquida pagamentos reais."
    )
    return {"X-API-Key": custom_token, "X-Role-Label": role_choice}


class FraudGuardApiClient:
    """
    HTTP client used by Streamlit to communicate strictly with FastAPI `/api/v1`
    (PRD Section 5.1 & 7.4: Streamlit never connects directly to the database).
    """

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.base_url = get_api_base_url()
        self.headers = headers or {}

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any] | list[Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        with httpx.Client(timeout=timeout) as client:
            resp = client.request(
                method=method,
                url=url,
                params=params,
                json=json_body,
                headers={"X-API-Key": self.headers.get("X-API-Key", "")},
            )
            if resp.status_code >= 400:
                try:
                    err_payload = resp.json()
                    msg = err_payload.get("error", {}).get("message", resp.text)
                    code = err_payload.get("error", {}).get("code", f"HTTP_{resp.status_code}")
                except Exception:
                    msg = resp.text
                    code = f"HTTP_{resp.status_code}"
                raise RuntimeError(f"[{code}] {msg}")
            return resp.json()

    def get_health(self) -> dict[str, Any]:
        return self._request("GET", "/health")  # type: ignore[return-value]

    def get_overview(self) -> dict[str, Any]:
        return self._request("GET", "/metrics/overview")  # type: ignore[return-value]

    def get_active_model(self) -> dict[str, Any]:
        return self._request("GET", "/models/active")  # type: ignore[return-value]

    def list_models(self) -> list[dict[str, Any]]:
        return self._request("GET", "/models")  # type: ignore[return-value]

    def activate_model(self, model_id: str) -> dict[str, Any]:
        return self._request("POST", f"/models/{model_id}/activate")  # type: ignore[return-value]

    def list_transactions(
        self,
        page: int = 1,
        page_size: int = 20,
        actual_class: int | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if actual_class is not None:
            params["actual_class"] = actual_class
        if source:
            params["source"] = source
        return self._request("GET", "/transactions", params=params)  # type: ignore[return-value]

    def get_transaction(self, tx_id: str) -> dict[str, Any]:
        return self._request("GET", f"/transactions/{tx_id}")  # type: ignore[return-value]

    def create_prediction(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/predictions", json_body=payload)  # type: ignore[return-value]

    def list_predictions(
        self,
        page: int = 1,
        page_size: int = 20,
        predicted_class: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if predicted_class is not None:
            params["predicted_class"] = predicted_class
        return self._request("GET", "/predictions", params=params)  # type: ignore[return-value]

    def list_alerts(
        self,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if status:
            params["status"] = status
        return self._request("GET", "/alerts", params=params)  # type: ignore[return-value]

    def update_alert(self, alert_id: str, status: str, notes: str | None) -> dict[str, Any]:
        return self._request(
            "PATCH",
            f"/alerts/{alert_id}",
            json_body={"status": status, "notes": notes},
        )  # type: ignore[return-value]

    def list_imports(self) -> list[dict[str, Any]]:
        return self._request("GET", "/imports")  # type: ignore[return-value]

    def create_import(
        self,
        filename: str = "creditcard_2023.csv",
        chunksize: int = 10_000,
        max_rows: int | None = None,
        run_synchronously: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "filename": filename,
            "chunksize": chunksize,
            "run_synchronously": run_synchronously,
        }
        if max_rows is not None:
            body["max_rows"] = max_rows
        return self._request("POST", "/imports", json_body=body, timeout=180.0)  # type: ignore[return-value]

