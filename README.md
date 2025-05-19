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




