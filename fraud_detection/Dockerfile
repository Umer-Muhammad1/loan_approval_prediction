FROM python:3.10-slim
WORKDIR /app

# 1. Install system dependencies (needed for some ML libs)
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the ENTIRE project (important for Kedro + API)
COPY . . 

# 4. Default environment variables
ENV MLFLOW_TRACKING_URI="http://mlflow-service:5000"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Default command (can be overridden in K8s)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]