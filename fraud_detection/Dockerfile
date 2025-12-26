FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app ./app

# Set environment variables for the cluster
ENV MLFLOW_TRACKING_URI="http://mlflow-service:5000"
ENV MODEL_NAME="loan_xgb_model"
ENV MODEL_ALIAS="production"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Use uvicorn to serve the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]