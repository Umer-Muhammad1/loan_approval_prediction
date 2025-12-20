FROM python:3.10-slim

RUN python -m pip install -U pip
RUN pip install uv

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

COPY . .

ENV KEDRO_ENV=docker
CMD ["kedro"]
