import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from fastapi.testclient import TestClient
from app import app  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")
client = TestClient(app)
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    logger.info(f"Response JSON: {data}")
    assert "message" in data
    assert data["message"] == "Welcome to the Bank Churn Prediction API"

if __name__ == "__main__":
    test_read_root()
    print("Test passed!")