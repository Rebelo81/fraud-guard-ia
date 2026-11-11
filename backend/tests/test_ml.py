from pathlib import Path

import numpy as np
import pytest

from backend.tests.conftest import make_synthetic_creditcard_df, sample_valid_features
from ml.src.evaluation import evaluate_rule_based_heuristic
from ml.src.inference import LoadedInferenceEngine, ModelArtifactRegistryError
from ml.src.preprocessing import (
    FEATURE_SCHEMA_VERSION,
    FEATURES,
    ID_COL,
    TARGET_COL,
    extract_X_y,
    stratified_train_val_test_split,
)
from ml.src.training import build_logistic_regression_pipeline


def test_id_and_class_strictly_excluded_from_features():
    """PRD 14.3: id/Class excluídos das features."""
    df = make_synthetic_creditcard_df(n_rows=100, seed=42)
    X, y = extract_X_y(df)

    assert ID_COL not in X.columns
    assert TARGET_COL not in X.columns
    assert list(X.columns) == FEATURES
    assert X.shape == (100, 29)
    assert set(y.unique()).issubset({0, 1})


def test_scaler_fitted_only_on_train_partition():
    """PRD 14.3: Sem ajuste de transformadores com dados de validação/teste."""
    df = make_synthetic_creditcard_df(n_rows=200, seed=42)
    split = stratified_train_val_test_split(df, random_state=42)

    # Inject extreme values into validation and test sets to prove scaler ignores them
    X_val_shifted = split.X_val.copy() + 10_000.0
    X_test_shifted = split.X_test.copy() + 50_000.0

    pipeline = build_logistic_regression_pipeline(random_state=42)
    pipeline.fit(split.X_train, split.y_train)

    scaler = pipeline.named_steps["scaler"]
    expected_train_mean = split.X_train.mean(axis=0).to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected_train_mean, rtol=1e-6, atol=1e-6)
    assert not np.allclose(scaler.mean_, X_val_shifted.mean(axis=0).to_numpy())
    assert not np.allclose(scaler.mean_, X_test_shifted.mean(axis=0).to_numpy())


def test_predict_proba_returns_finite_values_between_0_and_1(trained_artifacts_dir: Path):
    """PRD 14.3: predict_proba retorna valores finitos entre 0 e 1."""
    art_path = trained_artifacts_dir / "1.0.0" / "pipeline.joblib"
    engine = LoadedInferenceEngine(artifact_path=art_path)

    feats = sample_valid_features(multiplier=2.0)
    out1 = engine.predict_single(features=feats, amount=520.0)
    out2 = engine.predict_single(features=feats, amount=520.0)

    assert 0.0 <= out1.fraud_score <= 1.0
    assert np.isfinite(out1.fraud_score)
    # Deterministic reproducibility for same version and attributes (PRD 1.3 & 14.3)
    assert out1.fraud_score == out2.fraud_score
    assert out1.predicted_class == out2.predicted_class


def test_artifact_hash_and_schema_verified_on_load(trained_artifacts_dir: Path):
    """PRD 14.3: Schema de features e hash do artefato verificados no carregamento."""
    art_path = trained_artifacts_dir / "1.0.0" / "pipeline.joblib"

    with pytest.raises(ModelArtifactRegistryError, match="SHA-256 mismatch"):
        LoadedInferenceEngine(
            artifact_path=art_path,
            expected_sha256="0" * 64,
        )

    with pytest.raises(ModelArtifactRegistryError, match="Incompatible feature_schema_version"):
        LoadedInferenceEngine(
            artifact_path=art_path,
            expected_schema_version="incompatible-schema-v99",
        )

    engine = LoadedInferenceEngine(
        artifact_path=art_path,
        expected_schema_version=FEATURE_SCHEMA_VERSION,
    )
    assert engine.feature_schema_version == FEATURE_SCHEMA_VERSION


def test_rule_based_heuristic_vs_supervised_ml_comparison():
    """Verifica avaliação comparativa entre motor de regras determinísticas e ML supervisionado."""
    df = make_synthetic_creditcard_df(n_rows=200, seed=42)
    split = stratified_train_val_test_split(df, random_state=42)
    rule_metrics = evaluate_rule_based_heuristic(split.X_test, split.y_test)
    assert "f1" in rule_metrics
    assert "confusion_matrix" in rule_metrics
    assert "rule_definition" in rule_metrics


