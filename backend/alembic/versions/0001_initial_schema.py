"""Initial schema for FraudGuard AI

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-11-11 16:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

JsonVariant = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "dataset_imports",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False, unique=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("row_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("inserted_rows", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_dataset_imports_status",
        ),
        sa.CheckConstraint("row_count >= 0", name="ck_dataset_imports_row_count"),
        sa.CheckConstraint("inserted_rows >= 0", name="ck_dataset_imports_inserted_rows"),
    )

    op.create_table(
        "ml_models",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("version", sa.Text(), nullable=False, unique=True),
        sa.Column("algorithm", sa.Text(), nullable=False),
        sa.Column("feature_schema_version", sa.Text(), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("metrics", JsonVariant, nullable=False),
        sa.Column("threshold", sa.Double(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "threshold >= 0.0 AND threshold <= 1.0",
            name="ck_ml_models_threshold_range",
        ),
    )

    op.create_index(
        "uq_one_active_model",
        "ml_models",
        ["is_active"],
        unique=True,
        postgresql_where=sa.text("is_active = TRUE"),
        sqlite_where=sa.text("is_active = 1"),
    )

    op.create_table(
        "training_runs",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "model_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("ml_models.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("dataset_sha256", sa.String(length=64), nullable=False),
        sa.Column("split_seed", sa.Integer(), nullable=False),
        sa.Column("parameters", JsonVariant, nullable=False),
        sa.Column("metrics", JsonVariant, nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="completed"),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_training_runs_status",
        ),
    )

    op.create_table(
        "transactions",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "dataset_import_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("dataset_imports.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("source_record_id", sa.BigInteger(), nullable=True),
        sa.Column("amount", sa.Double(), nullable=False),
        sa.Column("features", JsonVariant, nullable=False),
        sa.Column("actual_class", sa.SmallInteger(), nullable=True),
        sa.Column("source", sa.Text(), nullable=False, server_default="api"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "dataset_import_id",
            "source_record_id",
            name="uq_transactions_import_source_record",
        ),
        sa.CheckConstraint(
            "actual_class IS NULL OR actual_class IN (0, 1)",
            name="ck_transactions_actual_class",
        ),
        sa.CheckConstraint(
            "source IN ('dataset', 'manual', 'api')",
            name="ck_transactions_source",
        ),
    )

    op.create_index(
        "ix_transactions_created_at_desc",
        "transactions",
        [sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_transactions_actual_class",
        "transactions",
        ["actual_class"],
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "transaction_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("transactions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "model_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("ml_models.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("fraud_score", sa.Double(), nullable=False),
        sa.Column("threshold", sa.Double(), nullable=False),
        sa.Column("predicted_class", sa.SmallInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("request_id", sa.Text(), nullable=True, unique=True),
        sa.CheckConstraint(
            "fraud_score >= 0.0 AND fraud_score <= 1.0",
            name="ck_predictions_fraud_score_range",
        ),
        sa.CheckConstraint(
            "threshold >= 0.0 AND threshold <= 1.0",
            name="ck_predictions_threshold_range",
        ),
        sa.CheckConstraint(
            "predicted_class IN (0, 1)",
            name="ck_predictions_predicted_class",
        ),
    )

    op.create_index(
        "ix_predictions_tx_created_desc",
        "predictions",
        ["transaction_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_predictions_model_id",
        "predictions",
        ["model_id"],
    )

    op.create_table(
        "fraud_alerts",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "prediction_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("predictions.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("status", sa.Text(), nullable=False, server_default="open"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "status IN ('open', 'reviewing', 'dismissed', 'confirmed')",
            name="ck_fraud_alerts_status",
        ),
    )

    op.create_index(
        "ix_fraud_alerts_status_created_desc",
        "fraud_alerts",
        ["status", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_fraud_alerts_status_created_desc", table_name="fraud_alerts")
    op.drop_table("fraud_alerts")
    op.drop_index("ix_predictions_model_id", table_name="predictions")
    op.drop_index("ix_predictions_tx_created_desc", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_transactions_actual_class", table_name="transactions")
    op.drop_index("ix_transactions_created_at_desc", table_name="transactions")
    op.drop_table("transactions")
    op.drop_table("training_runs")
    op.drop_index("uq_one_active_model", table_name="ml_models")
    op.drop_table("ml_models")
    op.drop_table("dataset_imports")

