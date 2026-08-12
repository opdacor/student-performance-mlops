"""
Contratos de la API, validados con Pydantic.

Los límites (ge/le) y categorías (Literal) vienen directamente de los
valores observados en data/student-por.csv. Si llega una entrada fuera
de rango, Pydantic la rechaza antes de que toque el modelo, con un 422
y un mensaje que indica exactamente qué campo falló.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StudentFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "school": "GP",
                "sex": "F",
                "age": 17,
                "address": "U",
                "famsize": "GT3",
                "Pstatus": "T",
                "Medu": 3,
                "Fedu": 2,
                "Mjob": "services",
                "Fjob": "other",
                "reason": "course",
                "guardian": "mother",
                "traveltime": 1,
                "studytime": 2,
                "failures": 0,
                "schoolsup": "no",
                "famsup": "yes",
                "paid": "no",
                "activities": "yes",
                "nursery": "yes",
                "higher": "yes",
                "internet": "yes",
                "romantic": "no",
                "famrel": 4,
                "freetime": 3,
                "goout": 3,
                "Dalc": 1,
                "Walc": 2,
                "health": 4,
                "absences": 4,
                "G1": 12,
                "G2": 13,
            }
        }
    )

    # --- Categóricas (valores tal cual aparecen en el dataset) ---
    school: Literal["GP", "MS"]
    sex: Literal["F", "M"]
    address: Literal["U", "R"]
    famsize: Literal["GT3", "LE3"]
    Pstatus: Literal["A", "T"]
    Mjob: Literal["at_home", "health", "other", "services", "teacher"]
    Fjob: Literal["at_home", "health", "other", "services", "teacher"]
    reason: Literal["course", "home", "other", "reputation"]
    guardian: Literal["father", "mother", "other"]
    schoolsup: Literal["yes", "no"]
    famsup: Literal["yes", "no"]
    paid: Literal["yes", "no"]
    activities: Literal["yes", "no"]
    nursery: Literal["yes", "no"]
    higher: Literal["yes", "no"]
    internet: Literal["yes", "no"]
    romantic: Literal["yes", "no"]

    # --- Numéricas (rangos observados en el dataset de entrenamiento) ---
    age: int = Field(ge=15, le=22, description="Edad del estudiante")
    Medu: int = Field(ge=0, le=4, description="Nivel educacional de la madre")
    Fedu: int = Field(ge=0, le=4, description="Nivel educacional del padre")
    traveltime: int = Field(ge=1, le=4)
    studytime: int = Field(ge=1, le=4)
    failures: int = Field(ge=0, le=4, description="Número de asignaturas reprobadas previamente")
    famrel: int = Field(ge=1, le=5)
    freetime: int = Field(ge=1, le=5)
    goout: int = Field(ge=1, le=5)
    Dalc: int = Field(ge=1, le=5, description="Consumo de alcohol en días de semana")
    Walc: int = Field(ge=1, le=5, description="Consumo de alcohol en fin de semana")
    health: int = Field(ge=1, le=5)
    absences: int = Field(ge=0, le=93)
    G1: int = Field(ge=0, le=20, description="Nota primer período")
    G2: int = Field(ge=0, le=20, description="Nota segundo período")


class PredictionResponse(BaseModel):
    predicted_g3: float = Field(description="Nota final predicha, escala 0-20")
    model_version: str


class BatchPredictionRequest(BaseModel):
    items: list[StudentFeatures] = Field(min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None


class ModelSchemaResponse(BaseModel):
    features: dict
    target: str
    model_type: str
    metrics: dict
