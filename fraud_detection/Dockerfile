FROM python:3.10-slim

RUN python -m pip install -U pip && pip install uv

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose ports for FastAPI (8000) and MLflow (5000)
#EXPOSE 8000
EXPOSE 5000