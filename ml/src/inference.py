import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from ml.src.ingestion import compute_file_sha256
from ml.src.preprocessing import (
    FEATURE_SCHEMA_VERSION,
    FEATURES,
    build_feature_Vector_df,
)


@dataclass(frozen=True)
class InferenceOutput:
    fraud_score: float
    threshold: float
    predicted_class: int
    requires_review: bool
    model_version: str
    feature_schema_version: str
    latency_ms: float


class ModelArtifactRegistryError(RuntimeError):
    """Raised when a model artifact is missing, corrupted, or incompatible."""


class LoadedInferenceEngine:
    """
    Loads an approved Scikit-learn Pipeline artifact, verifies its SHA-256 checksum
    and feature schema version, and produces deterministic predictions (PRD Section 5.1 & 8.4).
    """

    def __init__(
        self,
        artifact_path: str | Path,
        expected_sha256: str | None = None,
        expected_schema_version: str = FEATURE_SCHEMA_VERSION,
        model_version: str = "1.0.0",
        threshold: float = 0.50,
    ) -> None:
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.is_file():
            raise ModelArtifactRegistryError(
                f"Model artifact not found at: {self.artifact_path}"
            )

        actual_sha256 = compute_file_sha256(self.artifact_path)
        metadata_path = self.artifact_path.parent / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        target_sha256 = expected_sha256 or metadata.get("artifact_sha256")
        if target_sha256 and actual_sha256.lower() != target_sha256.lower():
            raise ModelArtifactRegistryError(
                f"Artifact SHA-256 mismatch for {self.artifact_path}: "
                f"expected {target_sha256}, got {actual_sha256}"
            )

        schema_ver = metadata.get("feature_schema_version", expected_schema_version)
        if schema_ver != expected_schema_version:
            raise ModelArtifactRegistryError(
                f"Incompatible feature_schema_version: expected {expected_schema_version}, got {schema_ver}"
            )

        meta_features = metadata.get("features")
        if meta_features is not None and list(meta_features) != FEATURES:
            raise ModelArtifactRegistryError(
                "Artifact feature list does not match canonical FEATURES order."
            )

        self.artifact_sha256 = actual_sha256
        self.feature_schema_version = schema_ver
        self.model_version = metadata.get("version", model_version)
        self.threshold = float(metadata.get("threshold", threshold))
        if not (0.0 <= self.threshold <= 1.0):
            raise ModelArtifactRegistryError(f"Invalid threshold: {self.threshold}")

        self.pipeline = joblib.load(self.artifact_path)
        if not hasattr(self.pipeline, "predict_proba"):
            raise ModelArtifactRegistryError(
                "Loaded artifact does not implement predict_proba."
            )

    def predict_single(
        self,
        features: dict[str, Any],
        amount: float,
        override_threshold: float | None = None,
    ) -> InferenceOutput:
        t0 = time.perf_counter()
        X_df = build_feature_Vector_df(features=features, amount=amount)
        proba = self.pipeline.predict_proba(X_df)
        raw_score = float(proba[0, 1])

        if not math.isfinite(raw_score) or raw_score < 0.0 or raw_score > 1.0:
            raise ModelArtifactRegistryError(
                f"Model returned invalid probability score: {raw_score}"
            )

        applied_threshold = (
            float(override_threshold) if override_threshold is not None else self.threshold
        )
        fraud_score = round(raw_score, 6)
        # Use raw_score >= applied_threshold or fraud_score >= applied_threshold consistently
        # PRD Section 7.2 & 14.1: predicted_class = 1 when fraud_score >= threshold
        predicted_class = 1 if fraud_score >= applied_threshold else 0
        requires_review = predicted_class == 1
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 4)

        return InferenceOutput(
            fraud_score=fraud_score,
            threshold=applied_threshold,
            predicted_class=predicted_class,
            requires_review=requires_review,
            model_version=self.model_version,
            feature_schema_version=self.feature_schema_version,
            latency_ms=latency_ms,
        )

