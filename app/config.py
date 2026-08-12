"""
Configuración del servicio, leída desde variables de entorno.

Ninguna ruta ni valor queda "hardcodeado" en el código: todo tiene un
default razonable para desarrollo local, pero puede sobreescribirse
con variables de entorno (documentadas en .env.example).
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "model.joblib"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", BASE_DIR / "models" / "metadata.json"))

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
