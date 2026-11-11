import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.base import Base
from backend.app.db.session import SessionLocal, engine
from backend.app.models.entities import MLModel, TrainingRun
from ml.src.ingestion import import_csv_batches_to_db
from ml.src.training import run_experiments


def register_artifacts_in_db(db: Session, artifacts_dir: str | Path = "ml/artifacts") -> None:
    base_dir = Path(artifacts_dir)
    for version in ("0.9.0-baseline", "1.0.0"):
        meta_path = base_dir / version / "metadata.json"
        artifact_path = base_dir / version / "pipeline.joblib"
        if not meta_path.is_file() or not artifact_path.is_file():
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        existing = db.scalar(select(MLModel).where(MLModel.version == version))
        if existing is None:
            model = MLModel(
                version=version,
                algorithm=meta["algorithm"],
                feature_schema_version=meta["feature_schema_version"],
                artifact_uri=str(artifact_path.as_posix()),
                artifact_sha256=meta["artifact_sha256"],
                metrics=meta["metrics"],
                threshold=float(meta["threshold"]),
                is_active=(version == "1.0.0"),
                activated_at=datetime.now(UTC) if version == "1.0.0" else None,
            )
            db.add(model)
            db.flush()

            run = TrainingRun(
                model_id=model.id,
                dataset_sha256=meta["dataset_sha256"],
                split_seed=int(meta["split_seed"]),
                parameters=meta["parameters"],
                metrics=meta["metrics"],
                started_at=datetime.fromisoformat(meta["started_at"]),
                finished_at=datetime.fromisoformat(meta["finished_at"]),
                status="completed",
            )
            db.add(run)
        else:
            existing.artifact_uri = str(artifact_path.as_posix())
            existing.artifact_sha256 = meta["artifact_sha256"]
            existing.metrics = meta["metrics"]
            existing.threshold = float(meta["threshold"])
            db.add(existing)

    db.commit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train models (if needed), register artifacts in DB, and import CSV batches."
    )
    parser.add_argument("--csv-path", type=str, default="creditcard_2023.csv")
    parser.add_argument("--artifacts-dir", type=str, default="ml/artifacts")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--max-import-rows", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=10_000)
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)

    artifacts_path = Path(args.artifacts_dir)
    if not args.skip_train or not (artifacts_path / "1.0.0" / "pipeline.joblib").is_file():
        print("[1/3] Running ML training & evaluation pipeline on dataset...")
        run_experiments(
            csv_path=args.csv_path,
            output_dir=args.artifacts_dir,
        )

    with SessionLocal() as db:
        print("[2/3] Registering versioned ML artifacts in database...")
        register_artifacts_in_db(db=db, artifacts_dir=args.artifacts_dir)

        if not args.skip_import:
            print(f"[3/3] Importing CSV batches from {args.csv_path}...")
            imp = import_csv_batches_to_db(
                db=db,
                csv_path=args.csv_path,
                chunksize=args.chunksize,
                max_rows=args.max_import_rows,
            )
            print(
                f"[OK] Import {imp.id} status={imp.status} | row_count={imp.row_count} | inserted_rows={imp.inserted_rows}"
            )


if __name__ == "__main__":
    main()

