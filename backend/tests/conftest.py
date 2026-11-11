import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import get_settings
from backend.app.db.base import Base
from backend.app.db.session import build_engine, get_db
from backend.app.main import create_app
from backend.app.models.entities import MLModel, TrainingRun
from backend.app.services.domain_services import runtime_registry
from ml.src.ingestion import compute_file_sha256
from ml.src.preprocessing import (
    EXPECTED_CSV_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    V_FEATURES,
    stratified_train_val_test_split,
)
from ml.src.training import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    train_and_save_model,
)


def make_synthetic_creditcard_df(n_rows: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    half = n_rows // 2
    classes = np.array([0] * half + [1] * (n_rows - half), dtype=int)
    rng.shuffle(classes)

    data: dict[str, np.ndarray] = {"id": np.arange(1, n_rows + 1, dtype=np.int64)}
    for idx, col in enumerate(V_FEATURES):
        # Give class 1 a clear shift on V1..V5 so models separate classes deterministically
        shift = classes * (1.8 if idx < 5 else 0.2) * ((-1) ** idx)
        data[col] = rng.normal(loc=shift, scale=1.0, size=n_rows).round(6)

    data["Amount"] = np.abs(rng.normal(loc=250.0 + classes * 150.0, scale=80.0, size=n_rows)).round(
        2
    )
    data["Class"] = classes
    return pd.DataFrame(data, columns=EXPECTED_CSV_COLUMNS)


@pytest.fixture()
def sample_csv_file(tmp_path: Path) -> Path:
    df = make_synthetic_creditcard_df(n_rows=300, seed=42)
    csv_path = tmp_path / "creditcard_2023.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def trained_artifacts_dir(tmp_path: Path, sample_csv_file: Path) -> Path:
    df = pd.read_csv(sample_csv_file)
    split = stratified_train_val_test_split(df, random_state=42)
    ds_sha = compute_file_sha256(sample_csv_file)
    out_dir = tmp_path / "artifacts"

    lr_pipe = build_logistic_regression_pipeline(random_state=42)
    train_and_save_model(
        pipeline=lr_pipe,
        version="0.9.0-baseline",
        algorithm="LogisticRegression",
        split=split,
        dataset_sha256=ds_sha,
        output_dir=out_dir,
        split_seed=42,
    )

    rf_pipe = build_random_forest_pipeline(random_state=42, n_estimators=20, max_depth=6)
    train_and_save_model(
        pipeline=rf_pipe,
        version="1.0.0",
        algorithm="RandomForestClassifier",
        split=split,
        dataset_sha256=ds_sha,
        output_dir=out_dir,
        split_seed=42,
    )
    return out_dir


@pytest.fixture()
def db_session(tmp_path: Path, trained_artifacts_dir: Path) -> Session:
    db_file = tmp_path / "test_fraudguard.db"
    test_engine = build_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(
        bind=test_engine, autocommit=False, autoflush=False, class_=Session
    )
    session = TestingSessionLocal()

    # Seed both models (0.9.0-baseline inactive, 1.0.0 active)
    for ver in ("0.9.0-baseline", "1.0.0"):
        meta_path = trained_artifacts_dir / ver / "metadata.json"
        art_path = trained_artifacts_dir / ver / "pipeline.joblib"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        m = MLModel(
            version=ver,
            algorithm=meta["algorithm"],
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            artifact_uri=str(art_path),
            artifact_sha256=meta["artifact_sha256"],
            metrics=meta["metrics"],
            threshold=float(meta["threshold"]),
            is_active=(ver == "1.0.0"),
            activated_at=datetime.now(UTC) if ver == "1.0.0" else None,
        )
        session.add(m)
        session.flush()
        tr = TrainingRun(
            model_id=m.id,
            dataset_sha256=meta["dataset_sha256"],
            split_seed=42,
            parameters=meta["parameters"],
            metrics=meta["metrics"],
            status="completed",
        )
        session.add(tr)
    session.commit()

    try:
        yield session
    finally:
        session.close()
        test_engine.dispose()


@pytest.fixture()
def client(db_session: Session, sample_csv_file: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    settings = get_settings()
    monkeypatch.setattr(settings, "CSV_DATA_DIR", str(sample_csv_file.parent))
    runtime_registry._engine = None

    app = create_app()

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def sample_valid_features(multiplier: float = 1.0) -> dict[str, float]:
    return {f"V{i}": round((0.25 * multiplier) * ((-1) ** i), 4) for i in range(1, 29)}

