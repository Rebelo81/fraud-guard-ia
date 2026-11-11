import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
     JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from backend.app.db.base import Base

JsonVariant = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DatasetImport(Base):
    __tablename__ = "dataset_imports"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_dataset_imports_status",
        ),
        CheckConstraint("row_count >= 0", name="ck_dataset_imports_row_count"),
        CheckConstraint("inserted_rows >= 0", name="ck_dataset_imports_inserted_rows"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    inserted_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    transactions: Mapped[list["Transaction"]] = relationship(
        "Transaction", back_populates="dataset_import"
    )


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (
        UniqueConstraint(
            "dataset_import_id",
            "source_record_id",
            name="uq_transactions_import_source_record",
        ),
        CheckConstraint(
            "actual_class IS NULL OR actual_class IN (0, 1)",
            name="ck_transactions_actual_class",
        ),
        CheckConstraint(
            "source IN ('dataset', 'manual', 'api')",
            name="ck_transactions_source",
        ),
        Index("ix_transactions_created_at_desc", text("created_at DESC")),
        Index("ix_transactions_actual_class", "actual_class"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_import_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("dataset_imports.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_record_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    amount: Mapped[float] = mapped_column(Double, nullable=False)
    features: Mapped[dict[str, float]] = mapped_column(JsonVariant, nullable=False)
    actual_class: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    source: Mapped[str] = mapped_column(Text, nullable=False, default="api")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    dataset_import: Mapped[DatasetImport | None] = relationship(
        "DatasetImport", back_populates="transactions"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="transaction", cascade="all, delete-orphan"
    )


class MLModel(Base):
    __tablename__ = "ml_models"
    __table_args__ = (
        CheckConstraint(
            "threshold >= 0.0 AND threshold <= 1.0",
            name="ck_ml_models_threshold_range",
        ),
        Index(
            "uq_one_active_model",
            "is_active",
            unique=True,
            postgresql_where=text("is_active = TRUE"),
            sqlite_where=text("is_active = 1"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    algorithm: Mapped[str] = mapped_column(Text, nullable=False)
    feature_schema_version: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JsonVariant, nullable=False)
    threshold: Mapped[float] = mapped_column(Double, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    training_runs: Mapped[list["TrainingRun"]] = relationship(
        "TrainingRun", back_populates="model"
    )
    predictions: Mapped[list["Prediction"]] = relationship("Prediction", back_populates="model")


class TrainingRun(Base):
    __tablename__ = "training_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_training_runs_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("ml_models.id", ondelete="SET NULL"),
        nullable=True,
    )
    dataset_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    split_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JsonVariant, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JsonVariant, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="completed")

    model: Mapped[MLModel | None] = relationship("MLModel", back_populates="training_runs")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint(
            "fraud_score >= 0.0 AND fraud_score <= 1.0",
            name="ck_predictions_fraud_score_range",
        ),
        CheckConstraint(
            "threshold >= 0.0 AND threshold <= 1.0",
            name="ck_predictions_threshold_range",
        ),
        CheckConstraint(
            "predicted_class IN (0, 1)",
            name="ck_predictions_predicted_class",
        ),
        Index("ix_predictions_tx_created_desc", "transaction_id", text("created_at DESC")),
        Index("ix_predictions_model_id", "model_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("transactions.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("ml_models.id", ondelete="RESTRICT"),
        nullable=False,
    )
    fraud_score: Mapped[float] = mapped_column(Double, nullable=False)
    threshold: Mapped[float] = mapped_column(Double, nullable=False)
    predicted_class: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    request_id: Mapped[str | None] = mapped_column(Text, nullable=True, unique=True)

    transaction: Mapped[Transaction] = relationship("Transaction", back_populates="predictions")
    model: Mapped[MLModel] = relationship("MLModel", back_populates="predictions")
    alert: Mapped["FraudAlert | None"] = relationship(
        "FraudAlert", back_populates="prediction", uselist=False, cascade="all, delete-orphan"
    )


class FraudAlert(Base):
    __tablename__ = "fraud_alerts"
    __table_args__ = (
        CheckConstraint(
            "status IN ('open', 'reviewing', 'dismissed', 'confirmed')",
            name="ck_fraud_alerts_status",
        ),
        Index("ix_fraud_alerts_status_created_desc", "status", text("created_at DESC")),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default="open")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    prediction: Mapped[Prediction] = relationship("Prediction", back_populates="alert")

