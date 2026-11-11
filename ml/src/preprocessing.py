import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_SCHEMA_VERSION = "creditcard-2023-v1"
ID_COL = "id"
TARGET_COL = "Class"
AMOUNT_COL = "Amount"
V_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]
FEATURES: list[str] = [*V_FEATURES, AMOUNT_COL]
EXPECTED_CSV_COLUMNS: list[str] = [ID_COL, *V_FEATURES, AMOUNT_COL, TARGET_COL]
FORBIDDEN_FEATURE_COLS: set[str] = {ID_COL, TARGET_COL}


@dataclass(frozen=True)
class DataSplitResult:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    quality_report: dict[str, Any]


def validate_feature_dict(features: dict[str, Any], amount: Any) -> dict[str, float]:
    """
    Validates that `features` contains exactly V1..V28 with finite numeric values,
    never contains `id` or `Class`, and `amount` is a finite non-negative number.
    Returns an ordered dictionary of all 29 features following `FEATURES`.
    """
    if not isinstance(features, dict):
        raise ValueError("features must be a dictionary mapping V1..V28 to finite numbers.")

    forbidden_found = FORBIDDEN_FEATURE_COLS.intersection(features.keys())
    if forbidden_found:
        raise ValueError(f"Forbidden columns present in features: {sorted(forbidden_found)}")

    expected_v_set = set(V_FEATURES)
    provided_keys = set(features.keys())
    missing = sorted(expected_v_set - provided_keys)
    extra = sorted(provided_keys - expected_v_set)
    if missing or extra:
        msg_parts = []
        if missing:
            msg_parts.append(f"missing={missing}")
        if extra:
            msg_parts.append(f"extra={extra}")
        raise ValueError(f"Invalid feature keys ({', '.join(msg_parts)}). Expected V1..V28.")

    ordered: dict[str, float] = {}
    for col in V_FEATURES:
        raw_val = features[col]
        if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float, np.number)):
            raise ValueError(f"Feature '{col}' must be a finite number.")
        val = float(raw_val)
        if not math.isfinite(val):
            raise ValueError(f"Feature '{col}' must be finite (no NaN or Infinity).")
        ordered[col] = val

    if isinstance(amount, bool) or not isinstance(amount, (int, float, np.number)):
        raise ValueError("Amount must be a finite number.")
    amount_val = float(amount)
    if not math.isfinite(amount_val) or amount_val < 0.0:
        raise ValueError("Amount must be a finite non-negative number.")

    ordered[AMOUNT_COL] = amount_val
    return ordered


def build_feature_Vector_df(features: dict[str, Any], amount: Any) -> pd.DataFrame:
    """
    Constructs a 1-row DataFrame with exact column order `FEATURES` (29 columns),
    independent of input JSON key order (PRD Section 6.2 & 8.1).
    """
    ordered = validate_feature_dict(features=features, amount=amount)
    row = [[ordered[col] for col in FEATURES]]
    return pd.DataFrame(row, columns=FEATURES, dtype=np.float64)


def extract_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extracts X (strictly the 29 `FEATURES` in canonical order) and y (`Class`).
    Explicitly guarantees `id` and `Class` are never in X.
    """
    missing_cols = [c for c in [*FEATURES, TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    X = df.loc[:, FEATURES].copy().astype(np.float64)
    assert ID_COL not in X.columns, "id must never be included in X"
    assert TARGET_COL not in X.columns, "Class must never be included in X"
    assert list(X.columns) == FEATURES, "X columns must strictly match canonical FEATURES order"

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite values (NaN or Inf) detected in feature matrix X.")

    y = df[TARGET_COL].astype(int).copy()
    invalid_targets = set(y.unique()) - {0, 1}
    if invalid_targets:
        raise ValueError(f"Invalid target class values detected: {invalid_targets}")

    return X, y


def stratified_train_val_test_split(
    df: pd.DataFrame,
    random_state: int = 42,
    deduplicate_exact_features: bool = False,
) -> DataSplitResult:
    """
    Performs duplicate/quality audit and splits dataset into:
    70% train, 15% validation, 15% test with stratification by Class and random_state=42.
    (PRD Section 2.2 & 2.3).
    """
    total_rows = int(len(df))
    duplicate_id_count = int(df[ID_COL].duplicated().sum()) if ID_COL in df.columns else 0
    exact_feature_duplicates = int(df.duplicated(subset=FEATURES).sum())
    exact_feature_and_class_duplicates = int(
        df.duplicated(subset=[*FEATURES, TARGET_COL]).sum()
    )

    working_df = df
    excluded_duplicates = 0
    if deduplicate_exact_features and exact_feature_duplicates > 0:
        working_df = df.drop_duplicates(subset=FEATURES, keep="first").reset_index(drop=True)
        excluded_duplicates = total_rows - len(working_df)

    X, y = extract_X_y(working_df)

    # First split: 70% train, 30% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    # Second split: divide the 30% equally into 15% validation and 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    quality_report = {
        "total_rows_input": total_rows,
        "rows_after_deduplication": int(len(working_df)),
        "excluded_exact_feature_duplicates": int(excluded_duplicates),
        "duplicate_id_count": duplicate_id_count,
        "exact_feature_duplicates_detected": exact_feature_duplicates,
        "exact_feature_and_class_duplicates_detected": exact_feature_and_class_duplicates,
        "missing_values_count": int(df[[*FEATURES, TARGET_COL]].isna().sum().sum()),
        "class_distribution_input": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
        "splits": {
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "train_class_counts": {
                str(k): int(v) for k, v in y_train.value_counts().sort_index().items()
            },
            "val_class_counts": {
                str(k): int(v) for k, v in y_val.value_counts().sort_index().items()
            },
            "test_class_counts": {
                str(k): int(v) for k, v in y_test.value_counts().sort_index().items()
            },
        },
        "random_state": random_state,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": len(FEATURES),
    }

    return DataSplitResult(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        quality_report=quality_report,
    )

