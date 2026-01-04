# End-to-End MLOps System: Loan Status Prediction

## Overview

This repository implements a **production-grade end-to-end MLOps system** for loan approval prediction.  
The project focuses on **engineering reliability, automation, and reproducibility**, rather than notebook-driven experimentation or model accuracy alone.

The system mirrors real-world machine learning platforms by **decoupling training, model registry, and inference**, enforcing data quality checks, and deploying services through **containerized CI/CD workflows on Kubernetes**.


---

## Project Goals

- Demonstrate production-oriented MLOps practices
- Enforce data validation and schema consistency
- Enable reproducible, modular ML pipelines
- Track experiments and manage model lifecycle
- Deploy and serve models via automated workflows

**Non-goals:** state-of-the-art modeling, notebook experimentation, manual deployments.

---

## Architecture Overview

The system follows a **modular, decoupled architecture** where training and inference are independently managed.


    A[Raw Loan Data] --> B[Data Validation using Great Expectations]
    B --> C[Feature Engineering Kedro EDA Pipeline]
    C --> D[Kedro Model Training Pipeline]

    D --> E[MLflow Tracking]
    E --> F[MLflow Model Registry]
    F --> |Production Alias| G[FastAPI Inference Service]

    G -->[Prediction API]


### Design Principles

- Training and inference are isolated
- Model registry is the contract between pipelines and serving
- Models are promoted via MLflow aliases
- CI/CD governs the path from code to production

---

## Data Validation & Quality Control

Data validation is enforced **before feature engineering and training**:

- Schema, ranges, and types validated using **Great Expectations**
- Explicit type casting ensures consistency
- Validation failures halt the pipeline early

This prevents schema drift, corrupted training data, and training–serving skew.

---

## Machine Learning Pipelines

All workflows are orchestrated using **Kedro**, enabling modular, reproducible pipelines.

- Feature engineering and training executed as Kedro pipelines
- Parameters, metrics, and artifacts logged to **MLflow**
- Models registered and versioned in MLflow Model Registry

Inference services dynamically load the current **Production** model from the registry.

---

## Containerization & Deployment

- All components are containerized using **Docker**
- The same images are used locally and in Kubernetes
- A Kubernetes cluster runs:
  - MLflow Tracking & Registry
  - FastAPI Inference Service

Training runs as containerized jobs; inference runs as a long-lived service.

---

## CI/CD Pipeline

**GitHub Actions** automates delivery:

1. Build Docker images for training and inference
2. Push versioned images to DockerHub
3. Kubernetes pulls and deploys updated images

This ensures a consistent and auditable deployment process.

---

## Model Serving

- **FastAPI** exposes a REST endpoint for predictions
- The service loads the production model dynamically from MLflow
- Model updates do not require API redeployment

---

## Monitoring & Metrics

- Model performance metrics tracked in **MLflow**
- Business metrics computed within the **Kedro** pipeline
.*

---

## Technology Stack

- Orchestration: Kedro  
- Data Validation: Great Expectations  
- Tracking & Registry: MLflow  
- API: FastAPI  
- Containerization: Docker  
- CI/CD: GitHub Actions  
- Deployment: Kubernetes  

---

## Why This Project Matters

This project demonstrates how machine learning systems can be built with **production constraints in mind**, emphasizing maintainability, automation, and operational safety rather than ad-hoc experimentation.

