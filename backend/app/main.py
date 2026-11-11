import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1.router import api_v1_router
from backend.app.core.config import get_settings
from backend.app.core.exceptions import register_exception_handlers
from backend.app.core.logging import configure_logging
from backend.app.db.session import SessionLocal
from backend.app.repositories.data_repositories import ModelRepository
from backend.app.services.domain_services import runtime_registry

settings = get_settings()
configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Loads active model artifact on startup if available in database."""
    try:
        with SessionLocal() as db:
            active_model = ModelRepository(db).get_active()
            if active_model is not None:
                runtime_registry.ensure_loaded(active_model)
                logger.info(
                    "Active ML model loaded on startup",
                    extra={"model_version": active_model.version},
                )
    except Exception as exc:
        logger.warning(
            "Startup model preload skipped or database not yet migrated",
            extra={"reason": str(exc)},
        )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "FraudGuard AI — Sistema Inteligente de Detecção de Fraudes Financeiras "
            "(Mini Capstone Pós-Graduação em IA). Aplicação acadêmica e demonstrativa."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def structured_logging_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        response.headers["X-Request-ID"] = req_id
        logger.info(
            "HTTP request processed",
            extra={
                "request_id": req_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        return response

    register_exception_handlers(app)
    app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)
    return app


app = create_app()

