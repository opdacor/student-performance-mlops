"""
Pruebas del servicio. Corren sin red y sin credenciales: TestClient
llama directamente al código de la app en memoria, no a un puerto real.

Cobertura exigida por la pauta:
    - Contrato de la API            -> test_health_*, test_model_schema_*
    - Validación de entradas         -> test_predict_invalid_*
    - Casos borde                    -> test_predict_edge_*
    - Errores esperados               -> test_predict_missing_field, batch vacío
"""

import copy


# --- Contrato de la API -------------------------------------------------

def test_health_returns_ok_and_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_schema_exposes_features_and_metrics(client):
    response = client.get("/model/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["target"] == "G3"
    assert "age" in body["features"]["numeric"]
    assert "sex" in body["features"]["categorical"]
    assert "mae" in body["metrics"]


# --- Camino feliz ---------------------------------------------------------

def test_predict_valid_returns_200_and_range(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["predicted_g3"] <= 20
    assert "model_version" in body


def test_predict_batch_valid_returns_same_count(client, valid_payload):
    payload = {"items": [valid_payload, valid_payload, valid_payload]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 3
    for prediction in body["predictions"]:
        assert 0 <= prediction["predicted_g3"] <= 20


# --- Validación de entradas: categorías fuera de dominio -------------------

def test_predict_invalid_categorical_value_returns_422(client, valid_payload):
    payload = copy.deepcopy(valid_payload)
    payload["sex"] = "X"  # solo se aceptan "F" o "M"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    fields_with_errors = [e["field"] for e in body["errors"]]
    assert "sex" in fields_with_errors


def test_predict_invalid_numeric_out_of_range_returns_422(client, valid_payload):
    payload = copy.deepcopy(valid_payload)
    payload["age"] = 999  # rango válido observado: 15-22
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_wrong_type_returns_422(client, valid_payload):
    payload = copy.deepcopy(valid_payload)
    payload["G1"] = "doce"  # debe ser int, no string
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# --- Errores esperados: campos faltantes / batch vacío ---------------------

def test_predict_missing_required_field_returns_422(client, valid_payload):
    payload = copy.deepcopy(valid_payload)
    del payload["G2"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    fields_with_errors = [e["field"] for e in body["errors"]]
    assert "G2" in fields_with_errors


def test_predict_batch_empty_list_returns_422(client):
    response = client.post("/predict/batch", json={"items": []})
    assert response.status_code == 422


# --- Casos borde: límites exactos observados en el dataset ------------------

def test_predict_edge_values_at_boundaries(client, valid_payload):
    payload = copy.deepcopy(valid_payload)
    payload.update({"age": 15, "failures": 0, "absences": 0, "G1": 0, "G2": 0})
    response_low = client.post("/predict", json=payload)
    assert response_low.status_code == 200

    payload.update({"age": 22, "failures": 4, "absences": 93, "G1": 20, "G2": 20})
    response_high = client.post("/predict", json=payload)
    assert response_high.status_code == 200
