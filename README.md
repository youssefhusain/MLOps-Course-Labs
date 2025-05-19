# MLOps Course Labs



This FastAPI app serves a machine learning model and provides endpoints to interact with it.

## 🔧 Requirements

- Python 3.10
- FastAPI
- Uvicorn
- Scikit-learn
- Joblib
- Pydantic

## 📁 File Structure

```
├── api.py              # Main FastAPI app
├── model.pkl      # Pretrained ML model (example: DecisionTreeClassifier)
├── test_api.py         # Test script (not shown here)
├── requirements.txt    # Python dependencies
```

## 🚀 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the API:

```bash
uvicorn api:app --reload
```

3. Access the Swagger UI:

[http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📦 API Endpoints

### `GET /`

Returns a welcome message.

**Response:**
```json
{
  "message": "Welcome to the ML prediction API"
}
```

---

### `GET /health`

Checks if the model was loaded correctly.

**Response:**
```json
{
  "status": "ok"
}
```
or
```json
{
  "status": "model not loaded"
}
```

---

### `POST /predict`

Takes customer features and returns the prediction.

**Request Body Example:**
```json
{
  "CreditScore": 600,
  "Geography": "France",
  "Gender": "Male",
  "Age": 40,
  "Tenure": 3,
  "Balance": 60000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 50000
}
```

**Response:**
```json
{
  "prediction": [1]
}
```

---

## 🧪 Testing

Make sure to include at least one test function in `test_api.py`. Example test:

```python
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the ML prediction API"}
```

---

## 🪵 Logging

Logging is enabled using Python’s built-in `logging` module to track:

- Model load status
- Requests received
- Prediction results
- Health checks

---

## 🌐 Screenshot

Insert a screenshot of your Swagger UI here:

![Swagger UI](./screenshot.png)

---

## ✅ Bonus (Optional)

- ✅ Use [HyperDX](https://hyperdx.io) for live logs
- ✅ Follow commit message convention (e.g., `feat: add predict endpoint`)
- ✅ Use `feature/api` branch then merge to `main`
- ✅ Write 3 test functions in total

---

## 🔗 Repo

[GitHub Repository Link](https://github.com/your-username/your-repo-name)


