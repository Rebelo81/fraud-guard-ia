import math

import pytest
from pydantic import ValidationError

from backend.app.schemas.api_schemas import PredictionCreateRequest
from backend.app.services.domain_services import (
    compute_predicted_class,
    validate_alert_transition,
)
from backend.tests.conftest import sample_valid_features
from ml.src.preprocessing import (
    FEATURES,
    V_FEATURES,
    build_feature_Vector_df,
    validate_feature_dict,
)


def test_schema_requires_all_28_v_features_and_amount():
    """PRD 14.1: Schema com 28 features obrigatórias e Amount."""
    feats = sample_valid_features()
    req = PredictionCreateRequest(amount=150.75, features=feats)
    assert req.amount == 150.75
    assert len(req.features) == 28

    # Missing V28 must fail validation
    incomplete = dict(feats)
    incomplete.pop("V28")
    with pytest.raises(ValidationError):
        PredictionCreateRequest(amount=150.75, features=incomplete)

    # Extra unknown feature or id/Class must fail validation
    with_id = dict(feats)
    with_id["id"] = 999
    with pytest.raises(ValidationError):
        PredictionCreateRequest(amount=150.75, features=with_id)

    with_class = dict(feats)
    with_class["Class"] = 1
    with pytest.raises(ValidationError):
        PredictionCreateRequest(amount=150.75, features=with_class)

    # Top-level Class or id must also be rejected (extra="forbid")
    with pytest.raises(ValidationError):
        PredictionCreateRequest.model_validate(
            {"amount": 100.0, "features": feats, "Class": 1}
        )


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf, "not-a-number", True])
def test_rejects_nan_infinity_and_non_numeric(bad_value):
    """PRD 14.1: Rejeição de NaN, infinito e dados não numéricos."""
    feats = sample_valid_features()
    feats["V5"] = bad_value  # type: ignore[assignment]
    with pytest.raises((ValidationError, ValueError)):
        PredictionCreateRequest(amount=100.0, features=feats)

    valid_feats = sample_valid_features()
    with pytest.raises((ValidationError, ValueError)):
         validate_feature_dict(valid_feats, bad_value)


def test_predicted_class_threshold_rule():
    """PRD 14.1: Regra predicted_class = fraud_score >= threshold."""
    assert compute_predicted_class(0.70, 0.70) == 1
    assert compute_predicted_class(0.85, 0.70) == 1
    assert compute_predicted_class(0.699999, 0.70) == 0
    assert compute_predicted_class(0.0, 0.50) == 0
    assert compute_predicted_class(1.0, 0.50) == 1


def test_alert_status_transitions():
    """PRD 14.1: Cálculo de alertas e transições de status permitidas."""
    assert validate_alert_transition("open", "reviewing") is True
    assert validate_alert_transition("open", "confirmed") is True
    assert validate_alert_transition("open", "dismissed") is True
    assert validate_alert_transition("reviewing", "confirmed") is True
    assert validate_alert_transition("reviewing", "dismissed") is True
    assert validate_alert_transition("confirmed", "dismissed") is False
    assert validate_alert_transition("dismissed", "confirmed") is False
    assert validate_alert_transition("confirmed", "open") is False


def test_canonical_29_feature_vector_independent_of_json_order():
    """PRD 14.1: Construção estável do vetor de 29 features, independente da ordem JSON."""
    forward_features = {f"V{i}": float(i) for i in range(1, 29)}
    reversed_features = {f"V{i}": float(i) for i in range(28, 0, -1)}

    df_forward = build_feature_Vector_df(forward_features, amount=999.99)
    df_reversed = build_feature_Vector_df(reversed_features, amount=999.99)

    assert list(df_forward.columns) == FEATURES
    assert list(df_reversed.columns) == FEATURES
    assert df_forward.shape == (1, 29)
    assert (df_forward.to_numpy() == df_reversed.to_numpy()).all()
    assert df_forward.iloc[0]["V1"] == 1.0
    assert df_forward.iloc[0]["V28"] == 28.0
    assert df_forward.iloc[0]["Amount"] == 999.99
    assert len(V_FEATURES) == 28

