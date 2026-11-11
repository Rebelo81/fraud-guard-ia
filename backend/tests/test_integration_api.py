from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.db.session import build_engine
from backend.app.models.entities import FraudAlert, MLModel, Prediction, Transaction
from backend.tests.conftest import sample_valid_features


def test_alembic_migrations_upgrade_and_downgrade(tmp_path: Path):
    """PRD 13 Sprint 1 & 14.2: Migrações aplicam e revertem em banco isolado."""
    mig_db = tmp_path / "alembic_test.db"
    db_url = f"sqlite:///{mig_db}"

    alembic_cfg = Config("backend/alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    # Note: env.py reads get_settings().DATABASE_URL; temporarily override
    settings = get_settings()
    orig_url = settings.DATABASE_URL
    orig_direct = settings.DIRECT_DATABASE_URL
    try:
        settings.DATABASE_URL = db_url
        settings.DIRECT_DATABASE_URL = db_url

        command.upgrade(alembic_cfg, "head")
        eng = build_engine(db_url)
        inspector = inspect(eng)
        tables = set(inspector.get_table_names())
        expected_tables = {
            "dataset_imports",
            "ml_models",
            "training_runs",
            "transactions",
            "predictions",
            "fraud_alerts",
        }
        assert expected_tables.issubset(tables)
        eng.dispose()

        command.downgrade(alembic_cfg, "base")
        eng2 = build_engine(db_url)
        inspector2 = inspect(eng2)
        remaining = set(inspector2.get_table_names()) - {"alembic_version"}
        assert len(remaining) == 0
        eng2.dispose()
    finally:
        settings.DATABASE_URL = orig_url
        settings.DIRECT_DATABASE_URL = orig_direct


def test_health_and_active_model_endpoints(client: TestClient):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["database"] == "connected"
    assert body["active_model_version"] == "1.0.0"
    assert body["model_ready"] is True

    mod_res = client.get("/api/v1/models/active")
    assert mod_res.status_code == 200
    assert mod_res.json()["version"] == "1.0.0"
    assert mod_res.json()["is_active"] is True


def test_atomic_prediction_creates_transaction_prediction_and_alert(
    client: TestClient, db_session: Session
):
    """PRD 14.2: Criação de transação, predição e alerta em commit atômico."""
    # Positive shift on V1..V5 triggers high fraud_score in synthetic model
    high_risk_features = {
        f"V{i}": (3.5 * ((-1) ** (i - 1)) if i <= 5 else 0.1) for i in range(1, 29)
    }
    payload = {
        "amount": 6531.37,
        "features": high_risk_features,
        "request_id": "demo-atomic-001",
    }

    res = client.post("/api/v1/predictions", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["model_version"] == "1.0.0"
    assert data["feature_schema_version"] == "creditcard-2023-v1"
    assert 0.0 <= data["fraud_score"] <= 1.0
    assert data["predicted_class"] == 1
    assert data["requires_review"] is True
    assert data["alert_id"] is not None

    # Idempotent retry with same request_id and same payload returns same prediction_id
    retry_res = client.post("/api/v1/predictions", json=payload)
    assert retry_res.status_code == 201
    assert retry_res.json()["prediction_id"] == data["prediction_id"]

    # Repeating same request_id with DIFFERENT payload returns 409 Conflict (PRD 7.3)
    conflict_payload = dict(payload, amount=99.99)
    conflict_res = client.post("/api/v1/predictions", json=conflict_payload)
    assert conflict_res.status_code == 409
    assert conflict_res.json()["error"]["code"] == "REQUEST_ID_CONFLICT"

    # Lookup prediction without recalculating
    get_res = client.get(f"/api/v1/predictions/{data['prediction_id']}")
    assert get_res.status_code == 200
    assert get_res.json()["fraud_score"] == data["fraud_score"]

    # Review the generated alert
    alert_id = data["alert_id"]
    patch_res = client.patch(
        f"/api/v1/alerts/{alert_id}",
        json={"status": "confirmed", "notes": "Confirmed fraud in demo review"},
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["status"] == "confirmed"
    assert patch_res.json()["reviewed_at"] is not None


def test_admin_authentication_and_idempotent_csv_import(
    client: TestClient, db_session: Session
):
    """PRD 14.2: Paginação, filtros, autenticação administrativa e importação sem duplicar."""
    settings = get_settings()
    admin_headers = {"X-API-Key": settings.ADMIN_API_TOKEN}
    analyst_headers = {"X-API-Key": settings.ANALYST_API_TOKEN}

    # Anonymous or analyst attempt on admin endpoint must fail with 401/403
    unauth = client.post("/api/v1/imports", json={"filename": "creditcard_2023.csv"})
    assert unauth.status_code == 401

    forbidden = client.post(
        "/api/v1/imports",
        json={"filename": "creditcard_2023.csv"},
        headers=analyst_headers,
    )
    assert forbidden.status_code == 403

    # Admin imports CSV synchronously
    imp_res = client.post(
        "/api/v1/imports",
        json={
            "filename": "creditcard_2023.csv",
            "chunksize": 500,
            "run_synchronously": True,
        },
        headers=admin_headers,
    )
    assert imp_res.status_code == 202
    imp_body = imp_res.json()
    assert imp_body["status"] == "completed"
    assert imp_body["row_count"] == 300
    assert imp_body["inserted_rows"] == 300

    # Duplicate import of same file returns 409 Conflict and does not duplicate rows
    dup_res = client.post(
        "/api/v1/imports",
        json={"filename": "creditcard_2023.csv", "run_synchronously": True},
        headers=admin_headers,
    )
    assert dup_res.status_code == 409
    assert dup_res.json()["error"]["code"] == "DUPLICATE_IMPORT"

    # Verify paginated transactions and class filter
    tx_res = client.get("/api/v1/transactions?page=1&page_size=25&actual_class=1")
    assert tx_res.status_code == 200
    tx_body = tx_res.json()
    assert tx_body["total"] == 150
    assert len(tx_body["items"]) == 25
    assert all(item["actual_class"] == 1 for item in tx_body["items"])


def test_transactional_single_active_model_switch(client: TestClient, db_session: Session):
    """PRD RF-05 & 6.2: Ativar uma única versão de modelo por vez, de maneira transacional."""
    settings = get_settings()
    admin_headers = {"X-API-Key": settings.ADMIN_API_TOKEN}

    models_res = client.get("/api/v1/models")
    assert models_res.status_code == 200
    models = models_res.json()
    baseline = next(m for m in models if m["version"] == "0.9.0-baseline")

    act_res = client.post(
        f"/api/v1/models/{baseline['id']}/activate",
        headers=admin_headers,
    )
    assert act_res.status_code == 200
    assert act_res.json()["version"] == "0.9.0-baseline"
    assert act_res.json()["is_active"] is True

    # Ensure only ONE model has is_active == True in DB
    active_models = list(
        db_session.scalars(select(MLModel).where(MLModel.is_active.is_(True))).all()
    )
    assert len(active_models) == 1
    assert active_models[0].version == "0.9.0-baseline"


def test_model_failure_rolls_back_and_leaves_no_partial_data(
    client: TestClient, db_session: Session
):
    """PRD 14.2: Falha de banco ou modelo retorna erro apropriado e não deixa dados parciais."""
    active_model = db_session.scalar(select(MLModel).where(MLModel.is_active.is_(True)))
    assert active_model is not None
    original_uri = active_model.artifact_uri
    active_model.artifact_uri = "non_existent_corrupted_path.joblib"
    db_session.commit()

    tx_count_before = len(list(db_session.scalars(select(Transaction)).all()))
    pred_count_before = len(list(db_session.scalars(select(Prediction)).all()))
    alert_count_before = len(list(db_session.scalars(select(FraudAlert)).all()))

    try:
        res = client.post(
            "/api/v1/predictions",
            json={
                "amount": 500.0,
                "features": sample_valid_features(),
                "request_id": "should-rollback-001",
            },
        )
        assert res.status_code == 503
        assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"

        assert len(list(db_session.scalars(select(Transaction)).all())) == tx_count_before
        assert len(list(db_session.scalars(select(Prediction)).all())) == pred_count_before
        assert len(list(db_session.scalars(select(FraudAlert)).all())) == alert_count_before
    finally:
        active_model.artifact_uri = original_uri
        db_session.commit()

