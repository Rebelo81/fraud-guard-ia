import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import status
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.exceptions import AppError
from backend.app.models.entities import (
    DatasetImport,
    FraudAlert,
    MLModel,
    Prediction,
    Transaction,
)
from backend.app.repositories.data_repositories import (
    AlertRepository,
    ImportRepository,
    ModelRepository,
    PredictionRepository,
    TransactionRepository,
    compute_overview_aggregates,
)
from backend.app.schemas.api_schemas import (
    AlertResponse,
    AlertUpdateRequest,
    MLModelResponse,
    MetricsOverviewResponse,
    PredictionCreateRequest,
    PredictionResponse,
    PredictionSummarySchema,
    TransactionCreateRequest,
    TransactionResponse,
)
from ml.src.inference import LoadedInferenceEngine, ModelArtifactRegistryError
from ml.src.ingestion import compute_file_sha256, import_csv_batches_to_db

logger = logging.getLogger(__name__)

# Allowed status transitions for human alert review (PRD Section 6.1 & 14.1)
ALLOWED_ALERT_TRANSITIONS: dict[str, set[str]] = {
    "open": {"open", "reviewing", "dismissed", "confirmed"},
    "reviewing": {"reviewing", "open", "dismissed", "confirmed"},
    "dismissed": {"dismissed", "reviewing"},
    "confirmed": {"confirmed", "reviewing"},
}


def validate_alert_transition(current_status: str, target_status: str) -> bool:
    allowed = ALLOWED_ALERT_TRANSITIONS.get(current_status, set())
    return target_status in allowed


def compute_predicted_class(fraud_score: float, threshold: float) -> int:
    """PRD Section 7.2 & 14.1: predicted_class = 1 when fraud_score >= threshold."""
    if not (math.isfinite(fraud_score) and 0.0 <= fraud_score <= 1.0):
        raise ValueError("fraud_score must be a finite float in [0, 1].")
    if not (math.isfinite(threshold) and 0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be a finite float in [0, 1].")
    return 1 if fraud_score >= threshold else 0


class InferenceRuntimeRegistry:
    """
    Manages the in-memory LoadedInferenceEngine synchronized with the active MLModel in DB.
    """

    def __init__(self) -> None:
        self._engine: LoadedInferenceEngine | None = None
        self._loaded_model_id: uuid.UUID | None = None
        self._loaded_sha256: str | None = None

    def resolve_artifact_path(self, artifact_uri: str) -> Path:
        raw_path = Path(artifact_uri)
        if raw_path.is_file():
            return raw_path
        root_candidate = get_settings().project_root / artifact_uri
        if root_candidate.is_file():
            return root_candidate
        return raw_path

    def ensure_loaded(self, active_model: MLModel) -> LoadedInferenceEngine:
        if (
            self._engine is not None
            and self._loaded_model_id == active_model.id
            and self._loaded_sha256 == active_model.artifact_sha256
        ):
            return self._engine

        resolved_path = self.resolve_artifact_path(active_model.artifact_uri)
        try:
            engine = LoadedInferenceEngine(
                artifact_path=resolved_path,
                expected_sha256=active_model.artifact_sha256,
                expected_schema_version=active_model.feature_schema_version,
                model_version=active_model.version,
                threshold=active_model.threshold,
            )
        except ModelArtifactRegistryError as exc:
            logger.error(
                "Failed to load active model artifact",
                extra={"model_version": active_model.version, "error": str(exc)},
            )
            raise AppError(
                code="MODEL_UNAVAILABLE",
                message=f"Active model '{active_model.version}' failed verification: {exc}",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            ) from exc

        self._engine = engine
        self._loaded_model_id = active_model.id
        self._loaded_sha256 = active_model.artifact_sha256
        return engine

    def reload(self, active_model: MLModel) -> LoadedInferenceEngine:
        self._engine = None
        self._loaded_model_id = None
        self._loaded_sha256 = None
        return self.ensure_loaded(active_model)


runtime_registry = InferenceRuntimeRegistry()


class ModelService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = ModelRepository(db)

    def get_active_model(self) -> MLModel:
        model = self.repo.get_active()
        if model is None:
            raise AppError(
                code="ACTIVE_MODEL_NOT_FOUND",
                message="No active ML model is registered.",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return model

    def list_models(self) -> list[MLModel]:
        return self.repo.list_all()

    def activate_model(self, model_id: uuid.UUID) -> MLModel:
        """
        PRD Section 7.4 & 8.4:
        Activates a single model version transactionally ONLY if its artifact,
        SHA-256 checksum, feature schema version, and metrics are validated.
        """
        model = self.repo.get_by_id(model_id)
        if model is None:
            raise AppError(
                code="MODEL_NOT_FOUND",
                message=f"Model {model_id} not found.",
                status_code=status.HTTP_404_NOT_FOUND,
            )

        if not model.metrics or not isinstance(model.metrics, dict):
            raise AppError(
                code="INVALID_MODEL_METRICS",
                message="Model cannot be activated without recorded evaluation metrics.",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        resolved_path = runtime_registry.resolve_artifact_path(model.artifact_uri)
        try:
            LoadedInferenceEngine(
                artifact_path=resolved_path,
                expected_sha256=model.artifact_sha256,
                expected_schema_version=model.feature_schema_version,
                model_version=model.version,
                threshold=model.threshold,
            )
        except ModelArtifactRegistryError as exc:
            raise AppError(
                code="INVALID_MODEL_ARTIFACT",
                message=f"Cannot activate model '{model.version}': {exc}",
                status_code=status.HTTP_400_BAD_REQUEST,
            ) from exc

        try:
            activated = self.repo.activate_model(model)
            self.db.commit()
            self.db.refresh(activated)
            runtime_registry.reload(activated)
            logger.info(
                "Activated ML model version",
                extra={"model_id": str(activated.id), "model_version": activated.version},
            )
            return activated
        except Exception as exc:
            self.db.rollback()
            raise AppError(
                code="MODEL_ACTIVATION_FAILED",
                message=f"Failed to activate model transactionally: {exc}",
                status_code=status.HTTP_409_CONFLICT,
            ) from exc


class TransactionService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = TransactionRepository(db)

    @staticmethod
    def _to_response(tx: Transaction) -> TransactionResponse:
        latest_pred = None
        if tx.predictions:
            sorted_preds = sorted(tx.predictions, key=lambda p: p.created_at, reverse=True)
            latest_pred = PredictionSummarySchema.model_validate(sorted_preds[0])
        return TransactionResponse(
            id=tx.id,
            dataset_import_id=tx.dataset_import_id,
            source_record_id=tx.source_record_id,
            amount=tx.amount,
            features=tx.features,
            actual_class=tx.actual_class,
            source=tx.source,
            created_at=tx.created_at,
            latest_prediction=latest_pred,
        )

    def create_transaction(self, req: TransactionCreateRequest) -> TransactionResponse:
        tx = self.repo.create(
            amount=req.amount,
            features=req.features,
            source=req.source,
            actual_class=req.actual_class,
        )
        self.db.commit()
        self.db.refresh(tx)
        return self._to_response(tx)

    def get_transaction(self, tx_id: uuid.UUID) -> TransactionResponse:
        tx = self.repo.get_by_id(tx_id)
        if tx is None:
            raise AppError(
                code="TRANSACTION_NOT_FOUND",
                message=f"Transaction {tx_id} not found.",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return self._to_response(tx)

    def list_transactions(
        self,
        page: int = 1,
        page_size: int = 20,
        actual_class: int | None = None,
        source: str | None = None,
    ) -> tuple[list[TransactionResponse], int]:
        items, total = self.repo.list_paginated(
            page=page,
            page_size=page_size,
            actual_class=actual_class,
            source=source,
        )
        return [self._to_response(tx) for tx in items], total


class PredictionService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.pred_repo = PredictionRepository(db)
        self.tx_repo = TransactionRepository(db)
        self.model_service = ModelService(db)

    @staticmethod
    def _build_response(pred: Prediction) -> PredictionResponse:
        model_ver = pred.model.version if pred.model else "unknown"
        schema_ver = (
            pred.model.feature_schema_version
            if pred.model
            else get_settings().MODEL_FEATURE_SCHEMA_VERSION
        )
        alert_id = pred.alert.id if pred.alert else None
        requires_review = alert_id is not None or pred.predicted_class == 1
        return PredictionResponse(
            prediction_id=pred.id,
            transaction_id=pred.transaction_id,
            model_version=model_ver,
            feature_schema_version=schema_ver,
            fraud_score=pred.fraud_score,
            threshold=pred.threshold,
            predicted_class=pred.predicted_class,
            requires_review=requires_review,
            alert_id=alert_id,
            created_at=pred.created_at,
            request_id=pred.request_id,
        )

    @staticmethod
    def _payload_matches_existing(pred: Prediction, req: PredictionCreateRequest) -> bool:
        tx = pred.transaction
        if tx is None:
            return False
        if req.transaction_id is not None:
            return pred.transaction_id == req.transaction_id
        if req.amount is None or req.features is None:
            return False
        if not math.isclose(tx.amount, req.amount, rel_tol=1e-9, abs_tol=1e-9):
            return False
        for k, v in req.features.items():
            if k not in tx.features or not math.isclose(
                float(tx.features[k]), float(v), rel_tol=1e-9, abs_tol=1e-9
            ):
                return False
        return True

    def create_prediction(self, req: PredictionCreateRequest) -> PredictionResponse:
        """
        PRD Section 6.2, 7.1, 7.2, 7.3:
        1. Deduplicates `request_id` if provided (returns 409 if payload differs).
        2. Loads active model and runs deterministic Scikit-learn inference.
        3. Persists Transaction, Prediction, and FraudAlert (when score >= threshold)
           within a SINGLE atomic SQL transaction.
        """
        if req.request_id:
            existing = self.pred_repo.get_by_request_id(req.request_id)
            if existing is not None:
                if self._payload_matches_existing(existing, req):
                    return self._build_response(existing)
                raise AppError(
                    code="REQUEST_ID_CONFLICT",
                    message=f"request_id '{req.request_id}' was already used with a different payload.",
                    status_code=status.HTTP_409_CONFLICT,
                )

        active_model = self.model_service.get_active_model()
        engine = runtime_registry.ensure_loaded(active_model)

        try:
            if req.transaction_id is not None:
                tx = self.tx_repo.get_by_id(req.transaction_id)
                if tx is None:
                    raise AppError(
                        code="TRANSACTION_NOT_FOUND",
                        message=f"Transaction {req.transaction_id} not found.",
                        status_code=status.HTTP_404_NOT_FOUND,
                    )
                amount = tx.amount
                features = tx.features
            else:
                assert req.amount is not None and req.features is not None
                amount = req.amount
                features = req.features
                tx = Transaction(
                    amount=amount,
                    features=features,
                    source="api",
                    actual_class=None,
                )
                self.db.add(tx)
                self.db.flush()

            inference_out = engine.predict_single(
                features=features,
                amount=amount,
                override_threshold=active_model.threshold,
            )

            predicted_class = compute_predicted_class(
                fraud_score=inference_out.fraud_score,
                threshold=active_model.threshold,
            )

            prediction = Prediction(
                transaction_id=tx.id,
                model_id=active_model.id,
                fraud_score=inference_out.fraud_score,
                threshold=active_model.threshold,
                predicted_class=predicted_class,
                request_id=req.request_id,
            )
            self.db.add(prediction)
            self.db.flush()

            alert: FraudAlert | None = None
            if predicted_class == 1:
                alert = FraudAlert(
                    prediction_id=prediction.id,
                    status="open",
                )
                self.db.add(alert)
                self.db.flush()

            self.db.commit()

            logger.info(
                "Prediction created",
                extra={
                    "request_id": req.request_id,
                    "prediction_id": str(prediction.id),
                    "transaction_id": str(tx.id),
                    "model_version": active_model.version,
                    "fraud_score": prediction.fraud_score,
                    "threshold": prediction.threshold,
                    "predicted_class": prediction.predicted_class,
                    "alert_created": alert is not None,
                    "latency_ms": inference_out.latency_ms,
                },
            )

            return PredictionResponse(
                prediction_id=prediction.id,
                transaction_id=tx.id,
                model_version=active_model.version,
                feature_schema_version=active_model.feature_schema_version,
                fraud_score=prediction.fraud_score,
                threshold=prediction.threshold,
                predicted_class=prediction.predicted_class,
                requires_review=alert is not None,
                alert_id=alert.id if alert is not None else None,
                created_at=prediction.created_at,
                request_id=prediction.request_id,
            )
        except AppError:
            self.db.rollback()
            raise
        except Exception as exc:
            self.db.rollback()
            logger.error("Atomic prediction persistence failed", exc_info=True)
            raise AppError(
                code="PREDICTION_PERSISTENCE_ERROR",
                message=f"Could not complete prediction transaction: {exc}",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            ) from exc

    def get_prediction(self, pred_id: uuid.UUID) -> PredictionResponse:
        """PRD Section 7.4: Never recalculate score on history lookup; return saved prediction."""
        pred = self.pred_repo.get_by_id(pred_id)
        if pred is None:
            raise AppError(
                code="PREDICTION_NOT_FOUND",
                message=f"Prediction {pred_id} not found.",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return self._build_response(pred)

    def list_predictions(
        self,
        page: int = 1,
        page_size: int = 20,
        predicted_class: int | None = None,
    ) -> tuple[list[PredictionResponse], int]:
        items, total = self.pred_repo.list_paginated(
            page=page,
            page_size=page_size,
            predicted_class=predicted_class,
        )
        return [self._build_response(p) for p in items], total


class AlertService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = AlertRepository(db)

    @staticmethod
    def _to_response(alert: FraudAlert) -> AlertResponse:
        pred = alert.prediction
        tx = pred.transaction
        model = pred.model
        return AlertResponse(
            id=alert.id,
            prediction_id=alert.prediction_id,
            transaction_id=pred.transaction_id,
            model_version=model.version if model else "unknown",
            fraud_score=pred.fraud_score,
            threshold=pred.threshold,
            predicted_class=pred.predicted_class,
            amount=tx.amount if tx else 0.0,
            actual_class=tx.actual_class if tx else None,
            status=alert.status,
            notes=alert.notes,
            reviewed_at=alert.reviewed_at,
            created_at=alert.created_at,
        )

    def list_alerts(
        self,
        page: int = 1,
        page_size: int = 20,
        status_filter: str | None = None,
    ) -> tuple[list[AlertResponse], int]:
        items, total = self.repo.list_paginated(
            page=page,
            page_size=page_size,
            status=status_filter,
        )
        return [self._to_response(a) for a in items], total

    def update_alert(self, alert_id: uuid.UUID, req: AlertUpdateRequest) -> AlertResponse:
        alert = self.repo.get_by_id(alert_id)
        if alert is None:
            raise AppError(
                code="ALERT_NOT_FOUND",
                message=f"Alert {alert_id} not found.",
                status_code=status.HTTP_404_NOT_FOUND,
            )

        if not validate_alert_transition(alert.status, req.status):
            raise AppError(
                code="INVALID_ALERT_TRANSITION",
                message=f"Transition from '{alert.status}' to '{req.status}' is not permitted.",
                status_code=status.HTTP_400_BAD_REQUEST,
                details={
                    "current_status": alert.status,
                    "requested_status": req.status,
                    "allowed_targets": sorted(ALLOWED_ALERT_TRANSITIONS.get(alert.status, set())),
                },
            )

        alert.status = req.status
        if req.notes is not None:
            alert.notes = req.notes
        alert.reviewed_at = datetime.now(timezone.utc)
        self.db.add(alert)
        self.db.commit()
        self.db.refresh(alert)
        return self._to_response(alert)


class ImportService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = ImportRepository(db)

    def resolve_csv_path(self, filename: str) -> Path:
        # Prevent path traversal
        safe_name = Path(filename).name
        if safe_name != filename:
            raise AppError(
                code="INVALID_FILENAME",
                message="Filename must not contain directory traversal segments.",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        settings = get_settings()
        candidates = [
            Path(settings.CSV_DATA_DIR) / safe_name,
            settings.project_root / safe_name,
            settings.project_root / "ml" / "data" / safe_name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise AppError(
            code="CSV_FILE_NOT_FOUND",
            message=f"CSV file '{safe_name}' not found in configured data directories.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    def initiate_import(self, filename: str) -> tuple[DatasetImport, Path]:
        csv_path = self.resolve_csv_path(filename)
        sha256 = compute_file_sha256(csv_path)

        existing = self.repo.get_by_sha256(sha256)
        if existing is not None:
            if existing.status in ("completed", "running"):
                raise AppError(
                    code="DUPLICATE_IMPORT",
                    message=f"File '{filename}' (sha256={sha256[:12]}...) has already been imported.",
                    status_code=status.HTTP_409_CONFLICT,
                    details={"import_id": str(existing.id), "status": existing.status},
                )
            return existing, csv_path

        record = DatasetImport(
            filename=csv_path.name,
            sha256=sha256,
            status="pending",
            row_count=0,
            inserted_rows=0,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record, csv_path

    def get_import(self, import_id: uuid.UUID) -> DatasetImport:
        rec = self.repo.get_by_id(import_id)
        if rec is None:
            raise AppError(
                code="IMPORT_NOT_FOUND",
                message=f"Dataset import {import_id} not found.",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return rec

    def list_imports(self) -> list[DatasetImport]:
        return self.repo.list_all()


def run_import_job_in_session(
    session_factory: Any,
    import_id: uuid.UUID,
    csv_path: Path,
    chunksize: int = 10_000,
    max_rows: int | None = None,
) -> None:
    db = session_factory()
    try:
        import_csv_batches_to_db(
            db=db,
            csv_path=csv_path,
            chunksize=chunksize,
            max_rows=max_rows,
            existing_import_id=import_id,
        )
    except Exception:
        logger.error("Background CSV import job failed", exc_info=True)
    finally:
        db.close()


class MetricsService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.model_repo = ModelRepository(db)

    def get_overview(self) -> MetricsOverviewResponse:
        agg = compute_overview_aggregates(self.db)
        active = self.model_repo.get_active()
        active_resp = MLModelResponse.model_validate(active) if active is not None else None
        return MetricsOverviewResponse(
            **agg,
            active_model=active_resp,
        )

