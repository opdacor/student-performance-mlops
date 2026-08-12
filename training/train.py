"""
Entrenamiento del modelo de predicción de rendimiento estudiantil (G3).

Uso:
    python training/train.py

Genera en models/:
    - model.joblib      -> Pipeline completo (preprocesador + RandomForest)
    - metadata.json      -> features esperadas, tipos, métricas, fecha, semilla
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --- Configuración reproducible ---------------------------------------
SEED = 42
TEST_SIZE = 0.2
TARGET = "G3"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student-por.csv"
MODELS_DIR = BASE_DIR / "models"

# --- Definición explícita del contrato de features ---------------------
# Estas listas son la fuente de verdad: se reutilizan en metadata.json
# y de ahí las va a leer la API (GET /model/schema).
NUMERIC_FEATURES = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
]

CATEGORICAL_FEATURES = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f"[train.py] No se encontró el dataset en {path}. "
            "Coloca student-por.csv en la carpeta data/ antes de entrenar."
        )
    df = pd.read_csv(path)
    missing_cols = set(ALL_FEATURES + [TARGET]) - set(df.columns)
    if missing_cols:
        sys.exit(f"[train.py] Faltan columnas esperadas en el CSV: {missing_cols}")
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=SEED,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    print(f"[train.py] Cargando datos desde {DATA_PATH}")
    df = load_data(DATA_PATH)

    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    print(f"[train.py] Train: {len(X_train)} filas | Test: {len(X_test)} filas")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[train.py] MAE  (test): {mae:.3f} puntos (escala G3: 0-20)")
    print(f"[train.py] R^2  (test): {r2:.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"[train.py] Modelo guardado en {model_path}")

    metadata = {
        "model_type": "RandomForestRegressor",
        "target": TARGET,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "n_features": len(ALL_FEATURES),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "seed": SEED,
        "test_size": TEST_SIZE,
        "metrics": {"mae": round(mae, 4), "r2": round(r2, 4)},
        "trained_at": datetime.now(UTC).isoformat(),
        "sklearn_pipeline_steps": [name for name, _ in pipeline.steps],
    }
    metadata_path = MODELS_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[train.py] Metadata guardada en {metadata_path}")


if __name__ == "__main__":
    main()
