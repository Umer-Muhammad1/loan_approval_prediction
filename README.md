**End-to-End MLOps Project: Loan Status Prediction**
---


A **production-grade MLOps system** that trains, validates, deploys, and serves a loan approval prediction model using Kubernetes, MLflow, and CI/CD automation. The goal is not only to train an accurate model, but to demonstrate production-grade machine learning engineering practices, including reproducibility, automation, deployment, and lifecycle management.

The system covers the complete ML lifecycle:

- Data ingestion and validation

- Feature engineering and model training

- Experiment tracking and model registry

- Containerization and CI/CD

- Kubernetes-based deployment

- Model serving via FastAPI


This repository is designed to reflect real-world MLOps workflows, not notebook-driven experimentation.

**High-Level Architecture**
---

The project follows a modular, decoupled architecture where training, model registry, and inference are independently managed.

End-to-end flow:

- Data is ingested and validated using Kedro pipelines

- Data quality checks and type casting prevent silent data issues

- Models are trained and logged to MLflow (tracking + registry)

- CI pipeline builds and pushes Docker images to DockerHub

- Kubernetes pulls the latest images and deploys services

- FastAPI dynamically loads the production model from MLflow Registry

Architecture
    
    A[Raw Loan Data] --> B[Data Validation using Great Expectations]
    B --> C[Feature Engineering Kedro EDA Pipeline]
    C --> D[Kedro Model Training Pipeline]

    D --> E[MLflow Tracking]
    E --> F[MLflow Model Registry]
    F --> |Production Alias| G[FastAPI Inference Service]

    G -->[Prediction API]



Training and inference are intentionally separated to mirror real production systems. 



**Data Validation & Quality Control**
---

To mitigate common data reliability issues, the pipeline integrates data validation and type enforcement:

- Great Expectations validates schema, ranges, and constraints

- Explicit type-casting node ensure consistent feature types

- Validation failures stop the pipeline early

- Prevents silent training-serving skew

These checks are executed as part of the Kedro pipeline before model training.


**Machine Learning Pipeline**
---
1. Training & Experimentation 
Pipelines are orchestrated using Kedro, includes EDA, modelling, and bussiness metrics. Conda environments ensure reproducible execution.

2. Experiment Tracking & Model Registry


flowchart LR

    A[Ingest Data] --> B[Schema Validation]
    B --> C[Range & Constraint Checks]
    C --> D[Type Casting]
    D --> E[EDA & Feature Engineering]
    E --> F[Model Training]
    F --> G[Evaluation & Business Metrics]
    G -->|Log Params & Metrics| H[MLflow Tracking]

    
MLflow Tracking logs parameters, metrics, and artifacts. Models are registered and versioned in MLflow Model Registry. Enables controlled promotion (using aliases) of models to production stages.


**CI/CD Pipeline**
--- 
The project uses GitHub Actions for continuous integration:

- Triggered on each push to the repository

- Builds Docker images for training and inference services

- Pushes versioned images to DockerHub

- Kubernetes automatically pulls the latest images during deployment

This ensures consistent and automated delivery from code to production.



**Containerization & Deployment**
---
1. **Docker**

- All services are containerized for environmental consistency

- The same images are used locally and in Kubernetes

2. **Kubernetes**

The cluster runs two main services:

- MLflow API: experiment tracking and model registry

- FastAPI Inference Service: serves predictions using the registered model

During deployment, Kubernetes pulls the latest image from DockerHub, and the Kedro experiments are executed inside the cluster. FastAPI loads the production model directly from MLflow Registry.



**Monitoring & Metrics**
---

- Model performance metrics are logged and tracked in MLflow

- Business-level metrics are computed via dedicated Kedro pipelines

- Enables traceability between data, experiments, and deployed models

**Technology Stack**
---

Pipeline Orchestration: Kedro

Data Validation: Great Expectations

Experiment Tracking & Registry: MLflow

API Service: FastAPI

Containerization: Docker, Docker Compose

CI/CD: GitHub Actions

Deployment: Kubernetes

Version Control: Git























