import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from backend.app.models.entities import DatasetImport, Transaction
from ml.src.preprocessing import (
    AMOUNT_COL,
    EXPECTED_CSV_COLUMNS,
    FEATURES,
    ID_COL,
    TARGET_COL,
    V_FEATURES,
)


@dataclass(frozen=True)
class CsvValidationSummary:
    filename: str
    sha256: str
    row_count: int
    class_counts: dict[int, int]
    duplicate_ids: int
    missing_values: int
    columns: list[str]


def compute_file_sha256(file_path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_csv_in_chunks(
    csv_path: str | Path,
    chunksize: int = 10_000,
) -> CsvValidationSummary:
    """
    Streams the CSV in chunks (`5_000` to `20_000`) to validate schema,
    check finite numeric values, verify unique `id`s, and reconcile class counts
    without loading the whole dataset into memory at once (PRD Section 2.1 & 6.3).
    """
    if not (100 <= chunksize <= 50_000):
        raise ValueError("chunksize must be between 100 and 50,000.")

    path = Path(csv_path)
    sha256 = compute_file_sha256(path)

    total_rows = 0
    class_counts: dict[int, int] = {0: 0, 1: 0}
    seen_ids: set[int] = set()
    duplicate_ids = 0
    missing_values = 0
    columns: list[str] = []

    for chunk_idx, chunk in enumerate(pd.read_csv(path, chunksize=chunksize)):
        if chunk_idx == 0:
            columns = list(chunk.columns)
            if set(columns) != set(EXPECTED_CSV_COLUMNS) or len(columns) != len(
                EXPECTED_CSV_COLUMNS
            ):
                raise ValueError(
                    f"Invalid CSV schema. Expected columns {EXPECTED_CSV_COLUMNS}, got {columns}"
                )

        chunk_na = int(chunk.isna().sum().sum())
        if chunk_na > 0:
            missing_values += chunk_na
            raise ValueError(f"CSV chunk {chunk_idx} contains {chunk_na} missing (NaN) values.")

        numeric_matrix = chunk[FEATURES].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(numeric_matrix).all():
            raise ValueError(f"CSV chunk {chunk_idx} contains non-finite values (NaN or Inf).")

        if (chunk[AMOUNT_COL] < 0).any():
            raise ValueError(f"CSV chunk {chunk_idx} contains negative Amount values.")

        classes = chunk[TARGET_COL].unique()
        invalid_classes = set(classes) - {0, 1}
        if invalid_classes:
            raise ValueError(f"CSV chunk {chunk_idx} has invalid Class values: {invalid_classes}")

        ids_series = chunk[ID_COL].astype(np.int64)
        chunk_id_list = ids_series.tolist()
        for rec_id in chunk_id_list:
            if rec_id in seen_ids:
                duplicate_ids += 1
            else:
                seen_ids.add(rec_id)

        if duplicate_ids > 0:
            raise ValueError(f"Duplicate id values detected in CSV ({duplicate_ids} duplicates).")

        vc = chunk[TARGET_COL].value_counts()
        for cls_val in (0, 1):
            class_counts[cls_val] += int(vc.get(cls_val, 0))

        total_rows += len(chunk)

    if total_rows == 0:
        raise ValueError("CSV file is empty.")

    return CsvValidationSummary(
        filename=path.name,
        sha256=sha256,
        row_count=total_rows,
        class_counts=class_counts,
        duplicate_ids=duplicate_ids,
        missing_values=missing_values,
        columns=columns,
    )


def convert_csv_to_parquet(
    csv_path: str | Path,
    parquet_path: str | Path,
    chunksize: int = 20_000,
) -> CsvValidationSummary:
    """
    Validates CSV in chunks and writes a verified local Parquet file for offline ML training.
    """
    summary = validate_csv_in_chunks(csv_path, chunksize=chunksize)
    out_path = Path(parquet_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df.to_parquet(out_path, index=False)
    return summary


def import_csv_batches_to_db(
    db: Session,
    csv_path: str | Path,
    chunksize: int = 5_000,
    max_rows: int | None = None,
    existing_import_id: uuid.UUID | None = None,
) -> DatasetImport:
    """
    Imports validated CSV rows in batches into `dataset_imports` and `transactions`
    (PRD Section 6.1, 6.2, 6.3).
    Safe for re-execution and idempotent on `(dataset_import_id, source_record_id)`.
    """
    path = Path(csv_path)
    sha256 = compute_file_sha256(path)

    if existing_import_id is not None:
        dataset_import = db.get(DatasetImport, existing_import_id)
        if dataset_import is None:
            raise ValueError(f"DatasetImport {existing_import_id} not found.")
    else:
        existing = db.scalar(select(DatasetImport).where(DatasetImport.sha256 == sha256))
        if existing is not None and existing.status == "completed":
            return existing
        if existing is not None:
            dataset_import = existing
        else:
            dataset_import = DatasetImport(
                filename=path.name,
                sha256=sha256,
                status="pending",
                row_count=0,
                inserted_rows=0,
            )
            db.add(dataset_import)
            db.commit()
            db.refresh(dataset_import)

    try:
        dataset_import.status = "running"
        dataset_import.error_message = None
        db.commit()

        dialect_name = db.bind.dialect.name if db.bind is not None else "postgresql"
        total_seen = 0
        total_inserted = 0

        for chunk in pd.read_csv(path, chunksize=chunksize):
            if set(chunk.columns) != set(EXPECTED_CSV_COLUMNS):
                raise ValueError(
                    f"Invalid CSV columns: expected {EXPECTED_CSV_COLUMNS}, got {list(chunk.columns)}"
                )

            if max_rows is not None and total_seen >= max_rows:
                break
            if max_rows is not None and total_seen + len(chunk) > max_rows:
                chunk = chunk.iloc[: max_rows - total_seen]

            if chunk.isna().any().any():
                raise ValueError("CSV contains NaN values.")

            feat_matrix = chunk[FEATURES].to_numpy(dtype=np.float64, copy=False)
            if not np.isfinite(feat_matrix).all():
                raise ValueError("CSV contains non-finite values.")

            records: list[dict[str, Any]] = []
            v_values = chunk[V_FEATURES].to_numpy(dtype=np.float64, copy=False)
            ids = chunk[ID_COL].to_numpy(dtype=np.int64, copy=False)
            amounts = chunk[AMOUNT_COL].to_numpy(dtype=np.float64, copy=False)
            classes = chunk[TARGET_COL].to_numpy(dtype=np.int16, copy=False)
            now_ts = datetime.now(timezone.utc)

            for i in range(len(chunk)):
                feature_dict = {V_FEATURES[j]: float(v_values[i, j]) for j in range(28)}
                records.append(
                    {
                        "id": uuid.uuid4(),
                        "dataset_import_id": dataset_import.id,
                        "source_record_id": int(ids[i]),
                        "amount": float(amounts[i]),
                        "features": feature_dict,
                        "actual_class": int(classes[i]),
                        "source": "dataset",
                        "created_at": now_ts,
                    }
                )

            if records:
                if dialect_name == "postgresql":
                    stmt = pg_insert(Transaction).values(records)
                    stmt = stmt.on_conflict_do_nothing(
                        index_elements=["dataset_import_id", "source_record_id"]
                    )
                    res = db.execute(stmt)
                    inserted_count = (
                        res.rowcount
                        if res.rowcount is not None and res.rowcount >= 0
                        else len(records)
                    )
                elif dialect_name == "sqlite":
                    stmt = sqlite_insert(Transaction).values(records)
                    stmt = stmt.on_conflict_do_nothing(
                        index_elements=["dataset_import_id", "source_record_id"]
                    )
                    res = db.execute(stmt)
                    inserted_count = (
                        res.rowcount
                        if res.rowcount is not None and res.rowcount >= 0
                        else len(records)
                    )
                else:
                    db.bulk_insert_mappings(Transaction, records)
                    inserted_count = len(records)

                total_seen += len(records)
                total_inserted += int(inserted_count)
                dataset_import.row_count = total_seen
                dataset_import.inserted_rows = total_inserted
                db.commit()

        dataset_import.status = "completed"
        dataset_import.completed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(dataset_import)
        return dataset_import

    except Exception as exc:
        db.rollback()
        dataset_import.status = "failed"
        dataset_import.error_message = str(exc)
        dataset_import.completed_at = datetime.now(timezone.utc)
        db.add(dataset_import)
        db.commit()
        raise

