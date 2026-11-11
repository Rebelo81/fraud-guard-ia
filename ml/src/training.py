import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.src.evaluation import (
    evaluate_predictions,
    measure_inference_latency,
    select_optimal_threshold,
)
from ml.src.ingestion import compute_file_sha256, validate_csv_in_chunks
from ml.src.preprocessing import (
    FEATURE_SCHEMA_VERSION,
    FEATURES,
    DataSplitResult,
    stratified_train_val_test_split,
)


@dataclass(frozen=True)
class TrainedModelArtifact:
    version: str
    algorithm: str
    feature_schema_version: str
    artifact_path: Path
    metadata_path: Path
    artifact_sha256: str
    threshold: float
    parameters: dict[str, Any]
    metrics: dict[str, Any]
    started_at: datetime
    finished_at: datetime


def build_logistic_regression_pipeline(random_state: int = 42) -> Pipeline:
    """
    PRD Section 8.2: Baseline Pipeline(StandardScaler, LogisticRegression).
    StandardScaler is fitted strictly on the training fold inside the Pipeline.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_random_forest_pipeline(
    random_state: int = 42,
    n_estimators: int = 40,
    max_depth: int = 12,
    min_samples_leaf: int = 4,
    max_samples: float | None = 0.25,
) -> Pipeline:
    """
    PRD Section 8.2: Candidate RandomForestClassifier wrapped in a Pipeline
    with StandardScaler for uniform preprocessing and zero leakage.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_samples=max_samples,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_and_save_model(
    pipeline: Pipeline,
    version: str,
    algorithm: str,
    split: DataSplitResult,
    dataset_sha256: str,
    output_dir: str | Path,
    split_seed: int = 42,
) -> TrainedModelArtifact:
    started_at = datetime.now(UTC)

    # Fit ONLY on training partition (70%)
    pipeline.fit(split.X_train, split.y_train)

    # Select operational threshold strictly on validation partition (15%)
    val_scores = pipeline.predict_proba(split.X_val)[:, 1]
    selected_threshold, threshold_selection_info = select_optimal_threshold(
        y_val=split.y_val,
        val_scores=val_scores,
    )
    val_metrics = evaluate_predictions(
        y_true=split.y_val,
        scores=val_scores,
        threshold=selected_threshold,
    )

    # Apply frozen threshold ONCE on the untouched test partition (15%)
    test_scores = pipeline.predict_proba(split.X_test)[:, 1]
    test_metrics = evaluate_predictions(
        y_true=split.y_test,
        scores=test_scores,
        threshold=selected_threshold,
    )

    latency_metrics = measure_inference_latency(pipeline=pipeline, X_sample=split.X_test)
    finished_at = datetime.now(UTC)

    version_dir = Path(output_dir) / version
    version_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = version_dir / "pipeline.joblib"
    metadata_path = version_dir / "metadata.json"

    joblib.dump(pipeline, artifact_path, compress=3)
    artifact_sha256 = compute_file_sha256(artifact_path)

    clf = pipeline.named_steps["classifier"]
    raw_params = clf.get_params()
    serializable_params = {
        k: v
        for k, v in raw_params.items()
        if isinstance(v, (str, int, float, bool)) or v is None
    }

    combined_metrics: dict[str, Any] = {
        # Top-level metrics reflect the untouched test set for direct API consumption
        **test_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "threshold_selection": threshold_selection_info,
        "latency": latency_metrics,
        "data_quality": split.quality_report,
        "methodological_notes": [
            "Class distribution in creditcard_2023.csv is 50% legitimate / 50% fraud (artificial balance).",
            "Do not extrapolate precision, false positive rate, or probability calibration to real banking prevalence.",
            "id and Class were strictly excluded from feature matrix X.",
            "No temporal timestamp exists in the dataset; stratified 70/15/15 split with random_state=42 was used.",
        ],
    }

    metadata_payload = {
        "version": version,
        "algorithm": algorithm,
        "sklearn_version": sklearn.__version__,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": FEATURES,
        "artifact_filename": artifact_path.name,
        "artifact_sha256": artifact_sha256,
        "dataset_sha256": dataset_sha256,
        "split_seed": split_seed,
        "threshold": selected_threshold,
        "parameters": serializable_params,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "metrics": combined_metrics,
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    return TrainedModelArtifact(
        version=version,
        algorithm=algorithm,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        artifact_sha256=artifact_sha256,
        threshold=selected_threshold,
        parameters=serializable_params,
        metrics=combined_metrics,
        started_at=started_at,
        finished_at=finished_at,
    )


def run_experiments(
    csv_path: str | Path,
    output_dir: str | Path = "ml/artifacts",
    parquet_path: str | Path | None = "ml/data/creditcard_2023.parquet",
    random_state: int = 42,
    rf_n_estimators: int = 40,
    rf_max_depth: int = 12,
) -> dict[str, TrainedModelArtifact]:
    """
    Runs the complete reproducible offline experiment workflow (PRD Section 8):
    1. Validates CSV in chunks and reconciles counts.
    2. Saves verified Parquet for training.
    3. Audits duplicates and performs stratified 70/15/15 split.
    4. Trains Baseline LogisticRegression (`0.9.0-baseline`) and Candidate RandomForest (`1.0.0`).
    5. Saves versioned artifacts, SHA-256 checksums, and full comparison report.
    """
    summary = validate_csv_in_chunks(csv_path, chunksize=20_000)
    df = pd.read_csv(csv_path)

    if parquet_path is not None:
        pq_file = Path(parquet_path)
        pq_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq_file, index=False)

    split = stratified_train_val_test_split(df=df, random_state=random_state)

    baseline_pipeline = build_logistic_regression_pipeline(random_state=random_state)
    baseline_artifact = train_and_save_model(
        pipeline=baseline_pipeline,
        version="0.9.0-baseline",
        algorithm="LogisticRegression",
        split=split,
        dataset_sha256=summary.sha256,
        output_dir=output_dir,
        split_seed=random_state,
    )

    rf_pipeline = build_random_forest_pipeline(
        random_state=random_state,
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
    )
    rf_artifact = train_and_save_model(
        pipeline=rf_pipeline,
        version="1.0.0",
        algorithm="RandomForestClassifier",
        split=split,
        dataset_sha256=summary.sha256,
        output_dir=output_dir,
        split_seed=random_state,
    )

    comparison_report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "filename": summary.filename,
            "sha256": summary.sha256,
            "row_count": summary.row_count,
            "class_counts": {str(k): v for k, v in summary.class_counts.items()},
            "duplicate_ids": summary.duplicate_ids,
            "missing_values": summary.missing_values,
        },
        "models": {
            baseline_artifact.version: {
                "algorithm": baseline_artifact.algorithm,
                "artifact_sha256": baseline_artifact.artifact_sha256,
                "threshold": baseline_artifact.threshold,
                "test_metrics": baseline_artifact.metrics["test"],
                "validation_metrics": baseline_artifact.metrics["validation"],
                "latency": baseline_artifact.metrics["latency"],
            },
            rf_artifact.version: {
                "algorithm": rf_artifact.algorithm,
                "artifact_sha256": rf_artifact.artifact_sha256,
                "threshold": rf_artifact.threshold,
                "test_metrics": rf_artifact.metrics["test"],
                "validation_metrics": rf_artifact.metrics["validation"],
                "latency": rf_artifact.metrics["latency"],
            },
        },
    }
    report_path = Path(output_dir) / "experiment_comparison.json"
    report_path.write_text(json.dumps(comparison_report, indent=2), encoding="utf-8")

    return {
        baseline_artifact.version: baseline_artifact,
        rf_artifact.version: rf_artifact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FraudGuard AI Offline ML Training Pipeline")
    parser.add_argument("--csv-path", type=str, default="creditcard_2023.csv")
    parser.add_argument("--output-dir", type=str, default="ml/artifacts")
    parser.add_argument("--parquet-path", type=str, default="ml/data/creditcard_2023.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifacts = run_experiments(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        parquet_path=args.parquet_path,
        random_state=args.seed,
    )
    for ver, art in artifacts.items():
        print(
            f"[OK] Version={ver} ({art.algorithm}) | SHA256={art.artifact_sha256[:12]}... "
            f"| Threshold={art.threshold} | Test F1={art.metrics['f1']} | Test PR-AUC={art.metrics['average_precision']}"
        )


if __name__ == "__main__":
    main()

