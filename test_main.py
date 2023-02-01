from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict():
    response = client.post("/predict", json={"feature1": 1.2, "feature2": 3.4})
    assert response.status_code == 200
    assert "prediction" in response.json()
