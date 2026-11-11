import math
import uuid
from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ml.src.preprocessing import FORBIDDEN_FEATURE_COLS, V_FEATURES

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int


class HealthResponse(BaseModel):
    status: str
    environment: str
    database: str
    active_model_version: str | None = None
    model_ready: bool = False


def _validate_v_features_dict(v: dict[str, Any]) -> dict[str, float]:
    if not isinstance(v, dict):
        raise ValueError("features must be a JSON object mapping V1..V28 to finite numbers.")
    forbidden = FORBIDDEN_FEATURE_COLS.intersection(v.keys())
    if forbidden:
        raise ValueError(f"Forbidden fields in features: {sorted(forbidden)}")

    expected_keys = set(V_FEATURES)
    actual_keys = set(v.keys())
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(f"features must contain strictly V1..V28 ({', '.join(details)}).")

    cleaned: dict[str, float] = {}
    for key in V_FEATURES:
        raw = v[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"Feature '{key}' must be a number.")
        val = float(raw)
        if not math.isfinite(val):
            raise ValueError(f"Feature '{key}' must be finite (no NaN or Infinity).")
        cleaned[key] = val
    return cleaned


class TransactionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(..., description="Transaction numeric amount (>= 0)")
    features: dict[str, float] = Field(..., description="Anonymized features V1..V28")
    actual_class: int | None = Field(
        default=None, description="Optional historical class (0 or 1) for demo/manual entry"
    )
    source: Literal["manual", "api"] = Field(default="manual")

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if isinstance(v, bool) or not math.isfinite(v) or v < 0.0:
            raise ValueError("amount must be a finite non-negative number.")
        return float(v)

    @field_validator("features", mode="before")
    @classmethod
    def validate_features(cls, v: Any) -> dict[str, float]:
        return _validate_v_features_dict(v)

    @field_validator("actual_class")
    @classmethod
    def validate_actual_class(cls, v: int | None) -> int | None:
        if v is not None and v not in (0, 1):
            raise ValueError("actual_class must be 0, 1, or null.")
        return v


class PredictionSummarySchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    model_id: uuid.UUID
    fraud_score: float
    threshold: float
    predicted_class: int
    created_at: datetime
    request_id: str | None = None


class TransactionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    dataset_import_id: uuid.UUID | None = None
    source_record_id: int | None = None
    amount: float
    features: dict[str, float]
    actual_class: int | None = None
    source: str
    created_at: datetime
    latest_prediction: PredictionSummarySchema | None = None


class PredictionCreateRequest(BaseModel):
    """
    PRD Section 7.1:
    Validates strict 28 features V1..V28 and amount, without extra fields (`id` or `Class` rejected).
    Also supports optional `transaction_id` when classifying an already-persisted transaction
    from the Transactions table (PRD Section 9).
    """

    model_config = ConfigDict(extra="forbid")

    amount: float | None = Field(default=None, description="Transaction numeric amount (>= 0)")
    features: dict[str, float] | None = Field(
        default=None, description="All 28 anonymized attributes V1..V28"
    )
    request_id: str | None = Field(
        default=None, max_length=128, description="Optional idempotency key"
    )
    transaction_id: uuid.UUID | None = Field(
        default=None,
        description="Optional existing transaction UUID to classify without duplicating transaction",
    )

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: float | None) -> float | None:
        if v is not None:
            if isinstance(v, bool) or not math.isfinite(v) or v < 0.0:
                raise ValueError("amount must be a finite non-negative number.")
            return float(v)
        return None

    @field_validator("features", mode="before")
    @classmethod
    def validate_features(cls, v: Any) -> dict[str, float] | None:
        if v is not None:
            return _validate_v_features_dict(v)
        return None

    @model_validator(mode="after")
    def check_payload_mode(self) -> "PredictionCreateRequest":
        if self.transaction_id is None and (self.amount is None or self.features is None):
            raise ValueError(
                "Both 'amount' and 'features' (V1..V28) are required when 'transaction_id' is not provided."
            )
        return self


class PredictionResponse(BaseModel):
    """PRD Section 7.2 response contract."""

    prediction_id: uuid.UUID
    transaction_id: uuid.UUID
    model_version: str
    feature_schema_version: str
    fraud_score: float
    threshold: float
    predicted_class: int
    requires_review: bool
    alert_id: uuid.UUID | None = None
    created_at: datetime | None = None
    request_id: str | None = None


class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    prediction_id: uuid.UUID
    transaction_id: uuid.UUID
    model_version: str
    fraud_score: float
    threshold: float
    predicted_class: int
    amount: float
    actual_class: int | None = None
    status: str
    notes: str | None = None
    reviewed_at: datetime | None = None
    created_at: datetime


class AlertUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["open", "reviewing", "dismissed", "confirmed"]
    notes: str | None = Field(default=None, max_length=2000)


class MLModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    version: str
    algorithm: str
    feature_schema_version: str
    artifact_uri: str
    artifact_sha256: str
    metrics: dict[str, Any]
    threshold: float
    is_active: bool
    created_at: datetime
    activated_at: datetime | None = None


class ImportCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(default="creditcard_2023.csv", min_length=1, max_length=255)
    chunksize: int = Field(default=10_000, ge=500, le=20_000)
    max_rows: int | None = Field(
        default=None,
        ge=1,
        description="Optional limit for quick demo import; omit to import all 568,630 rows",
    )
    run_synchronously: bool = Field(
        default=False,
        description="If true, waits for completion before returning; otherwise runs in background (202 Accepted)",
    )


class DatasetImportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    sha256: str
    status: str
    row_count: int
    inserted_rows: int
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class MetricsOverviewResponse(BaseModel):
    total_transactions: int
    dataset_transactions: int
    manual_or_api_transactions: int
    imported_fraud_count: int
    imported_legit_count: int
    imported_fraud_fraction: float
    total_predictions: int
    predicted_fraud_count: int
    predicted_legit_count: int
    total_alerts: int
    open_alerts: int
    alerts_by_status: dict[str, int]
    predictions_by_ingestion_date: list[dict[str, Any]]
    active_model: MLModelResponse | None = None

