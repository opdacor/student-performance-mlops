"""
Carga del artefacto entrenado y lógica de inferencia.

El modelo se carga UNA VEZ (ver main.py, evento de startup) y se
reutiliza en cada request. Cargarlo en cada predicción sería lento
y es exactamente lo que la pauta prohíbe explícitamente.
"""

import json

import joblib
import pandas as pd

from app.config import METADATA_PATH, MODEL_PATH
from app.schemas import StudentFeatures


class ModelNotAvailableError(Exception):
    """El modelo no pudo cargarse o no está listo para predecir."""


class Predictor:
    def __init__(self) -> None:
        self.pipeline = None
        self.metadata: dict = {}
        self.feature_order: list[str] = []

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise ModelNotAvailableError(f"No se encontró el modelo en {MODEL_PATH}")
        if not METADATA_PATH.exists():
            raise ModelNotAvailableError(f"No se encontró metadata en {METADATA_PATH}")

        self.pipeline = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_order = (
            self.metadata["numeric_features"] + self.metadata["categorical_features"]
        )

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None

    def _to_dataframe(self, items: list[StudentFeatures]) -> pd.DataFrame:
        rows = [item.model_dump() for item in items]
        df = pd.DataFrame(rows)
        # Reordena las columnas exactamente como se entrenó el pipeline.
        return df[self.feature_order]

    def predict(self, features: StudentFeatures) -> float:
        return self.predict_batch([features])[0]

    def predict_batch(self, items: list[StudentFeatures]) -> list[float]:
        if not self.is_ready:
            raise ModelNotAvailableError("El modelo no está cargado todavía.")
        df = self._to_dataframe(items)
        predictions = self.pipeline.predict(df)
        return [round(float(p), 2) for p in predictions]


# Instancia única compartida por toda la app (patrón singleton simple).
predictor = Predictor()
