**End-to-End MLOps Project: Loan Status Prediction**


1. **Project Overview**

This project is a full end-to-end MLOps system for predicting loan approval status using structured tabular data. The goal is not only to train an accurate model, but to demonstrate production-grade machine learning engineering practices including reproducibility, automation, deployment, and lifecycle management.

The system covers the complete ML lifecycle:

Data ingestion and validation

Feature engineering and model training

Experiment tracking and model registry

Containerization and CI/CD

Kubernetes-based deployment

Model serving via FastAPI

This repository is designed to reflect real-world MLOps workflows, not notebook-driven experimentation.

2. **Key Objectives**

Build reproducible ML pipelines using Kedro

Enforce data quality using Great Expectations

Track experiments and manage models with MLflow

Automate builds and deployments using GitHub Actions

Package services using Docker & Docker Compose

Deploy training and inference services on Kubernetes

Serve predictions through a FastAPI application

Maintain clear separation between training, registry, and serving

3. System Architecture

4. **Technology Stack**
- Machine Learning & Pipelines

- Python

- Kedro – modular, reproducible ML pipelines

- Scikit-learn model training

- Data Quality & Validation

- Great Expectations – schema validation and data quality checks

- Explicit type casting nodes to prevent silent data issues

- Experiment Tracking & Model Management

- MLflow Tracking – experiments and metrics
 
- MLflow Model Registry – versioning and lifecycle management
 
- Model promotion using Production alias
 
- API & Serving
- 
- FastAPI – model inference service

- Model loaded at application startup from MLflow registry

- DevOps & Infrastructure

- Docker – containerization

- Docker Compose – local multi-service setup

- GitHub Actions – CI pipeline (build, test, push images)

- Docker Hub – image registry

- Kubernetes – orchestration and deployment