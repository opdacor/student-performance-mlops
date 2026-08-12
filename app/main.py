"""
Servicio de predicción de rendimiento estudiantil (nota final G3).

Endpoints:
    GET  /health          -> estado del servicio y del modelo
    POST /predict          -> predicción para un estudiante
    POST /predict/batch     -> predicción para varios estudiantes
    GET  /model/schema      -> features esperadas y métricas del modelo
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.predictor import ModelNotAvailableError, predictor
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelSchemaResponse,
    PredictionResponse,
    StudentFeatures,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Se ejecuta UNA sola vez al arrancar el contenedor/servidor.
    try:
        predictor.load()
    except ModelNotAvailableError:
        # No tumbamos el arranque: /health reportará "degraded" en vez
        # de que el contenedor entero falle al levantar.
        pass
    yield


app = FastAPI(
    title="Student Performance Predictor",
    description="Predice la nota final (G3, escala 0-20) de un estudiante.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    # Mensaje accionable: qué campo falló y por qué, nunca un 500 genérico.
    errors = [
        {"field": ".".join(str(p) for p in e["loc"][1:]), "issue": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": "Entrada inválida", "errors": errors})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if predictor.is_ready else "degraded",
        model_loaded=predictor.is_ready,
        model_version=predictor.metadata.get("trained_at") if predictor.is_ready else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: StudentFeatures) -> PredictionResponse:
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="El modelo no está disponible todavía.")
    value = predictor.predict(features)
    return PredictionResponse(
        predicted_g3=value,
        model_version=predictor.metadata.get("trained_at", "unknown"),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="El modelo no está disponible todavía.")
    values = predictor.predict_batch(request.items)
    version = predictor.metadata.get("trained_at", "unknown")
    return BatchPredictionResponse(
        predictions=[PredictionResponse(predicted_g3=v, model_version=version) for v in values]
    )


@app.get("/model/schema", response_model=ModelSchemaResponse)
def model_schema() -> ModelSchemaResponse:
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="El modelo no está disponible todavía.")
    meta = predictor.metadata
    return ModelSchemaResponse(
        features={
            "numeric": meta["numeric_features"],
            "categorical": meta["categorical_features"],
        },
        target=meta["target"],
        model_type=meta["model_type"],
        metrics=meta["metrics"],
    )
