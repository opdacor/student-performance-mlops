# Imagen base slim: solo lo esencial de Python, sin herramientas de
# compilación ni paquetes de sistema que no usamos. Reduce tamaño y
# superficie de ataque.
FROM python:3.12-slim

# Evita que Python genere .pyc y fuerza logs sin buffer (aparecen al
# instante en `docker logs`, útil para depurar).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Copiamos solo requirements primero: si el código cambia pero las
# dependencias no, Docker reutiliza esta capa cacheada y el build es
# mucho más rápido.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Solo lo que el SERVICIO necesita en tiempo de ejecución.
# training/ y data/ quedan fuera a propósito (ver .dockerignore):
# no sirven para nada en producción y solo inflan la imagen.
COPY app/ ./app/
COPY models/ ./models/

# Usuario no-root: si alguien compromete el contenedor, no queda con
# privilegios de administrador dentro de él.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /srv
USER appuser

EXPOSE 8000

# Docker (y el orquestador que sea) puede preguntar "¿sigues vivo?"
# sin depender de que nosotros lo revisemos a mano.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
