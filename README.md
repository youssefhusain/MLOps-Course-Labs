# ==========================
# Dockerfile for FastAPI App
# ==========================
# Step 1:the Dockerfile
```bash
FROM python:3.10-slim

WORKDIR /app


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

# Step 2: Run the Docker Container


```bash

docker build -f Dockerfile -t youssef_app .
```
