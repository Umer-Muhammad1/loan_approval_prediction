ARG BASE_IMAGE=python:3.10-slim
FROM $BASE_IMAGE as runtime-environment

# update pip and install uv
RUN python -m pip install -U "pip>=21.2"
RUN pip install uv

# install project requirements
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

# 👇 project root inside container
WORKDIR /app
USER root
RUN mkdir -p /app/mlruns /app/data && \
    chown -R ${KEDRO_UID}:${KEDRO_GID} /app
USER kedro_docker

FROM runtime-environment

ARG KEDRO_UID=999
ARG KEDRO_GID=0
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

EXPOSE 8888
ENV KEDRO_ENV=docker
CMD ["kedro", "run"]

# To build the docker image, run:
# docker build -t image_name .
# To run the docker image, run:
# docker run --rm -v %cd%\data:/app/data image_name
# to persist with mlflow experiments, run:
# docker run --rm -v %cd%\data:/app/data -v mlflow_data:/app/mlruns image_name
