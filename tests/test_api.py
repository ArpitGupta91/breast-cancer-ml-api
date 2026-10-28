
from fastapi.testclient import TestClient
from app.main import app, FEATURE_NAMES

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction():
    features = {name: 0.0 for name in FEATURE_NAMES}

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in ["benign", "malignant"]
    assert "probabilities" in result
    assert "0" in result["probabilities"]
    assert "1" in result["probabilities"]


def test_missing_features_rejected():
    response = client.post(
        "/predict",
        json={"features": {}}
    )

    assert response.status_code == 422
