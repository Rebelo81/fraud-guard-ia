import uuid
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, Query, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.core.security import (
    AuthenticatedPrincipal,
    require_admin,
    require_analyst,
)
from backend.app.db.session import SessionLocal, get_db
from backend.app.repositories.data_repositories import ModelRepository
from backend.app.schemas.api_schemas import (
    AlertResponse,
    AlertUpdateRequest,
    DatasetImportResponse,
    HealthResponse,
    ImportCreateRequest,
    MLModelResponse,
    MetricsOverviewResponse,
    PaginatedResponse,
    PredictionCreateRequest,
    PredictionResponse,
    TransactionCreateRequest,
    TransactionResponse,
)
from backend.app.services.domain_services import (
    AlertService,
    ImportService,
    MetricsService,
    ModelService,
    PredictionService,
    TransactionService,
    run_import_job_in_session,
    runtime_registry,
)
from ml.src.ingestion import import_csv_batches_to_db

api_v1_router = APIRouter()


@api_v1_router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Saúde da API e prontidão do banco/modelo",
)
def get_health(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    db_status = "disconnected"
    active_version: str | None = None
    model_ready = False

    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
        active_model = ModelRepository(db).get_active()
        if active_model is not None:
            active_version = active_model.version
            try:
                runtime_registry.ensure_loaded(active_model)
                model_ready = True
            except Exception:
                model_ready = False
    except Exception:
        db_status = "disconnected"

    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        environment=settings.APP_ENV,
        database=db_status,
        active_model_version=active_version,
        model_ready=model_ready,
    )


@api_v1_router.get(
    "/models/active",
    response_model=MLModelResponse,
    tags=["Models"],
    summary="Versão e métricas do modelo ativo",
)
def get_active_model(
    _principal: AuthenticatedPrincipal = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> MLModelResponse:
    service = ModelService(db)
    model = service.get_active_model()
    return MLModelResponse.model_validate(model)


@api_v1_router.get(
    "/models",
    response_model=list[MLModelResponse],
    tags=["Models"],
    summary="Listar versões de modelos registrados para comparação",
)
def list_models(
    _principal: AuthenticatedPrincipal = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> list[MLModelResponse]:
    service = ModelService(db)
    return [MLModelResponse.model_validate(m) for m in service.list_models()]


@api_v1_router.post(
    "/models/{id}/activate",
    response_model=MLModelResponse,
    tags=["Models"],
    summary="Ativar modelo aprovado de maneira transacional (Administrador)",
)
def activate_model(
    id: uuid.UUID,
    _principal: AuthenticatedPrincipal = Depends(require_admin),
    db: Session = Depends(get_db),
) -> MLModelResponse:
    service = ModelService(db)
    activated = service.activate_model(id)
    return MLModelResponse.model_validate(activated)


@api_v1_router.get(
    "/transactions",
    response_model=PaginatedResponse[TransactionResponse],
    tags=["Transactions"],
    summary="Listagem paginada de transações com filtro por classe",
)
def list_transactions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    actual_class: int | None = Query(default=None, ge=0, le=1),
    source: Literal["dataset", "manual", "api"] | None = Query(default=None),
    _principal: AuthenticatedPrincipal = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> PaginatedResponse[TransactionResponse]:
    service = TransactionService(db)
    items, total = service.list_transactions(
        page=page,
        page_size=page_size,
        actual_class=actual_class,
        source=source,
    )
    return PaginatedResponse[TransactionResponse](
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@api_v1_router.get(
    "/transactions/{id}",
    response_model=TransactionResponse,
    tags=["Transactions"],
    summary="Detalhes de uma transação",
)
def get_transaction(
    id: uuid.UUID,
    _principal: AuthenticatedPrincipal = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> TransactionResponse:
    service = TransactionService(db)
    return service.get_transaction(id)


@api_v1_router.post(
    "/transactions",
    response_model=TransactionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Transactions"],
    summary="Persistência de uma transação de demonstração",
)
def create_transaction(
    payload: TransactionCreateRequest,
    _principal: AuthenticatedPrincipal = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> TransactionResponse:
    service = TransactionService(db)
    return service.create_transaction(payload)


@api_v1_router.post(
    "/predictions",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Predictions"],
    summary="Classificar transação, persistir predição e gerar alerta se aplicável",
)
def create_prediction(
    payload: PredictionCreateRequest,
    _principal: AuthenticatedPrincipal = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    service = PredictionService(db)
    return service.create_prediction(payload)


@api_v1_router.get(
    "/predictions",
    response_model=PaginatedResponse[PredictionResponse],
    tags=["Predictions"],
    summary="Listar histórico de predições persistidas",
)
def list_predictions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    predicted_class: int | None = Query(default=None, ge=0, le=1),
    _principal: AuthenticatedPrincipal = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> PaginatedResponse[PredictionResponse]:
    service = PredictionService(db)
    items, total = service.list_predictions(
        page=page,
        page_size=page_size,
        predicted_class=predicted_class,
    )
    return PaginatedResponse[PredictionResponse](
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@api_v1_router.get(
    "/predictions/{id}",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Recuperar resultado de predição salvo sem recalcular escore",
)
def get_prediction(
    id: uuid.UUID,
    _principal: AuthenticatedPrincipal = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    service = PredictionService(db)
    return service.get_prediction(id)


@api_v1_router.get(
    "/alerts",
    response_model=PaginatedResponse[AlertResponse],
    tags=["Alerts"],
    summary="Listar alertas de fraude com paginação e filtros",
)
def list_alerts(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status_filter: Literal["open", "reviewing", "dismissed", "confirmed"] | None = Query(
        default=None, alias="status"
    ),
    _principal: AuthenticatedPrincipal = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> PaginatedResponse[AlertResponse]:
    service = AlertService(db)
    items, total = service.list_alerts(
        page=page,
        page_size=page_size,
        status_filter=status_filter,
    )
    return PaginatedResponse[AlertResponse](
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@api_v1_router.patch(
    "/alerts/{id}",
    response_model=AlertResponse,
    tags=["Alerts"],
    summary="Revisar e atualizar estado de alerta",
)
def update_alert(
    id: uuid.UUID,
    payload: AlertUpdateRequest,
    _principal: AuthenticatedPrincipal = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> AlertResponse:
    service = AlertService(db)
    return service.update_alert(id, payload)


@api_v1_router.get(
    "/metrics/overview",
    response_model=MetricsOverviewResponse,
    tags=["Metrics"],
    summary="Indicadores operacionais e gerenciais",
)
def get_metrics_overview(
    _principal: AuthenticatedPrincipal = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> MetricsOverviewResponse:
    service = MetricsService(db)
    return service.get_overview()


@api_v1_router.post(
    "/imports",
    response_model=DatasetImportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Imports"],
    summary="Importação controlada de CSV em lotes (Administrador)",
)
def create_import(
    payload: ImportCreateRequest,
    background_tasks: BackgroundTasks,
    _principal: AuthenticatedPrincipal = Depends(require_admin),
    db: Session = Depends(get_db),
) -> DatasetImportResponse:
    service = ImportService(db)
    record, csv_path = service.initiate_import(payload.filename)

    if payload.run_synchronously:
        updated = import_csv_batches_to_db(
            db=db,
            csv_path=csv_path,
            chunksize=payload.chunksize,
            max_rows=payload.max_rows,
            existing_import_id=record.id,
        )
        return DatasetImportResponse.model_validate(updated)

    session_factory = (
        db.get_bind() if hasattr(db, "get_bind") else None
    )
    from sqlalchemy.orm import sessionmaker

    bound_factory = (
        sessionmaker(bind=session_factory, autocommit=False, autoflush=False)
        if session_factory is not None
        else SessionLocal
    )
    background_tasks.add_task(
        run_import_job_in_session,
        bound_factory,
        record.id,
        csv_path,
        payload.chunksize,
        payload.max_rows,
    )
    return DatasetImportResponse.model_validate(record)


@api_v1_router.get(
    "/imports",
    response_model=list[DatasetImportResponse],
    tags=["Imports"],
    summary="Listar execuções de importação de dataset (Administrador)",
)
def list_imports(
    _principal: AuthenticatedPrincipal = Depends(require_admin),
    db: Session = Depends(get_db),
) -> list[DatasetImportResponse]:
    service = ImportService(db)
    return [DatasetImportResponse.model_validate(r) for r in service.list_imports()]


@api_v1_router.get(
    "/imports/{id}",
    response_model=DatasetImportResponse,
    tags=["Imports"],
    summary="Acompanhar processamento e reconciliação de importação (Administrador)",
)
def get_import(
    id: uuid.UUID,
    _principal: AuthenticatedPrincipal = Depends(require_admin),
    db: Session = Depends(get_db),
) -> DatasetImportResponse:
    service = ImportService(db)
    record = service.get_import(id)
    return DatasetImportResponse.model_validate(record)

