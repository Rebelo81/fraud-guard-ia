import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, func, select, update
from sqlalchemy.orm import Session, joinedload

from backend.app.models.entities import (
    DatasetImport,
    FraudAlert,
    MLModel,
    Prediction,
    Transaction,
)


class TransactionRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_id(self, tx_id: uuid.UUID) -> Transaction | None:
        stmt = (
            select(Transaction)
            .options(joinedload(Transaction.predictions))
            .where(Transaction.id == tx_id)
        )
        return self.db.scalars(stmt).first()

    def list_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        actual_class: int | None = None,
        source: str | None = None,
    ) -> tuple[list[Transaction], int]:
        filters = []
        if actual_class is not None:
            filters.append(Transaction.actual_class == actual_class)
        if source is not None:
            filters.append(Transaction.source == source)

        count_stmt = select(func.count(Transaction.id)).where(*filters)
        total = int(self.db.scalar(count_stmt) or 0)

        stmt = (
            select(Transaction)
            .options(joinedload(Transaction.predictions))
            .where(*filters)
            .order_by(desc(Transaction.created_at))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        items = list(self.db.scalars(stmt).unique().all())
        return items, total

    def create(
        self,
        amount: float,
        features: dict[str, float],
        source: str = "api",
        actual_class: int | None = None,
        dataset_import_id: uuid.UUID | None = None,
        source_record_id: int | None = None,
    ) -> Transaction:
        tx = Transaction(
            amount=amount,
            features=features,
            source=source,
            actual_class=actual_class,
            dataset_import_id=dataset_import_id,
            source_record_id=source_record_id,
        )
        self.db.add(tx)
        self.db.flush()
        return tx


class ModelRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_active(self) -> MLModel | None:
        stmt = select(MLModel).where(MLModel.is_active.is_(True))
        return self.db.scalar(stmt)

    def get_by_id(self, model_id: uuid.UUID) -> MLModel | None:
        return self.db.get(MLModel, model_id)

    def get_by_version(self, version: str) -> MLModel | None:
        stmt = select(MLModel).where(MLModel.version == version)
        return self.db.scalar(stmt)

    def list_all(self) -> list[MLModel]:
        stmt = select(MLModel).order_by(desc(MLModel.created_at))
        return list(self.db.scalars(stmt).all())

    def deactivate_all(self) -> None:
        stmt = (
            update(MLModel)
            .where(MLModel.is_active.is_(True))
            .values(is_active=False)
        )
        self.db.execute(stmt)
        self.db.flush()

    def activate_model(self, model: MLModel) -> MLModel:
        self.deactivate_all()
        model.is_active = True
        model.activated_at = datetime.now(UTC)
        self.db.add(model)
        self.db.flush()
        return model


class PredictionRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_id(self, pred_id: uuid.UUID) -> Prediction | None:
        stmt = (
            select(Prediction)
            .options(
                joinedload(Prediction.model),
                joinedload(Prediction.alert),
                joinedload(Prediction.transaction),
            )
            .where(Prediction.id == pred_id)
        )
        return self.db.scalars(stmt).first()

    def get_by_request_id(self, request_id: str) -> Prediction | None:
        stmt = (
            select(Prediction)
            .options(
                joinedload(Prediction.model),
                joinedload(Prediction.alert),
                joinedload(Prediction.transaction),
            )
            .where(Prediction.request_id == request_id)
        )
        return self.db.scalars(stmt).first()

    def list_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        predicted_class: int | None = None,
    ) -> tuple[list[Prediction], int]:
        filters = []
        if predicted_class is not None:
            filters.append(Prediction.predicted_class == predicted_class)

        count_stmt = select(func.count(Prediction.id)).where(*filters)
        total = int(self.db.scalar(count_stmt) or 0)

        stmt = (
            select(Prediction)
            .options(
                joinedload(Prediction.model),
                joinedload(Prediction.alert),
                joinedload(Prediction.transaction),
            )
            .where(*filters)
            .order_by(desc(Prediction.created_at))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        items = list(self.db.scalars(stmt).unique().all())
        return items, total


class AlertRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_id(self, alert_id: uuid.UUID) -> FraudAlert | None:
        stmt = (
            select(FraudAlert)
            .options(
                joinedload(FraudAlert.prediction).joinedload(Prediction.transaction),
                joinedload(FraudAlert.prediction).joinedload(Prediction.model),
            )
            .where(FraudAlert.id == alert_id)
        )
        return self.db.scalars(stmt).first()

    def list_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
    ) -> tuple[list[FraudAlert], int]:
        filters = []
        if status is not None:
            filters.append(FraudAlert.status == status)

        count_stmt = select(func.count(FraudAlert.id)).where(*filters)
        total = int(self.db.scalar(count_stmt) or 0)

        stmt = (
            select(FraudAlert)
            .options(
                joinedload(FraudAlert.prediction).joinedload(Prediction.transaction),
                joinedload(FraudAlert.prediction).joinedload(Prediction.model),
            )
            .where(*filters)
            .order_by(desc(FraudAlert.created_at))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        items = list(self.db.scalars(stmt).unique().all())
        return items, total


class ImportRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_id(self, import_id: uuid.UUID) -> DatasetImport | None:
        return self.db.get(DatasetImport, import_id)

    def get_by_sha256(self, sha256: str) -> DatasetImport | None:
        stmt = select(DatasetImport).where(DatasetImport.sha256 == sha256)
        return self.db.scalar(stmt)

    def list_all(self) -> list[DatasetImport]:
        stmt = select(DatasetImport).order_by(desc(DatasetImport.created_at))
        return list(self.db.scalars(stmt).all())


def compute_overview_aggregates(db: Session) -> dict[str, Any]:
    total_tx = int(db.scalar(select(func.count(Transaction.id))) or 0)
    dataset_tx = int(
        db.scalar(select(func.count(Transaction.id)).where(Transaction.source == "dataset")) or 0
    )
    manual_api_tx = total_tx - dataset_tx

    imported_fraud = int(
        db.scalar(
            select(func.count(Transaction.id)).where(
                Transaction.source == "dataset", Transaction.actual_class == 1
            )
        )
        or 0
    )
    imported_legit = int(
        db.scalar(
            select(func.count(Transaction.id)).where(
                Transaction.source == "dataset", Transaction.actual_class == 0
            )
        )
        or 0
    )
    denom = imported_fraud + imported_legit
    fraud_fraction = round(imported_fraud / denom, 6) if denom > 0 else 0.0

    total_preds = int(db.scalar(select(func.count(Prediction.id))) or 0)
    pred_fraud = int(
        db.scalar(select(func.count(Prediction.id)).where(Prediction.predicted_class == 1)) or 0
    )
    pred_legit = total_preds - pred_fraud

    total_alerts = int(db.scalar(select(func.count(FraudAlert.id))) or 0)
    alerts_by_status: dict[str, int] = {
        "open": 0,
        "reviewing": 0,
        "dismissed": 0,
        "confirmed": 0,
    }
    status_rows = db.execute(
        select(FraudAlert.status, func.count(FraudAlert.id)).group_by(FraudAlert.status)
    ).all()
    for st_name, cnt in status_rows:
        alerts_by_status[str(st_name)] = int(cnt)

    # Group recent predictions by ingestion date (YYYY-MM-DD)
    date_expr = func.date(Prediction.created_at)
    period_rows = db.execute(
        select(
            date_expr.label("ingestion_date"),
            func.count(Prediction.id).label("total"),
            func.sum(Prediction.predicted_class).label("flagged"),
        )
        .group_by(date_expr)
        .order_by(date_expr)
    ).all()
    predictions_by_date = [
        {
            "ingestion_date": str(r.ingestion_date),
            "total": int(r.total or 0),
            "flagged": int(r.flagged or 0),
        }
        for r in period_rows
    ]

    return {
        "total_transactions": total_tx,
        "dataset_transactions": dataset_tx,
        "manual_or_api_transactions": manual_api_tx,
        "imported_fraud_count": imported_fraud,
        "imported_legit_count": imported_legit,
        "imported_fraud_fraction": fraud_fraction,
        "total_predictions": total_preds,
        "predicted_fraud_count": pred_fraud,
        "predicted_legit_count": pred_legit,
        "total_alerts": total_alerts,
        "open_alerts": alerts_by_status.get("open", 0),
        "alerts_by_status": alerts_by_status,
        "predictions_by_ingestion_date": predictions_by_date,
    }

