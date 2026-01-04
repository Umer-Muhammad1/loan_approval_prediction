# Running the Fraud Detection Project

This document explains how to **run the `fraud_detection` Kedro project** locally and in Kubernetes.  
It covers **training pipelines, model promotion, and inference**, including MLflow usage.

---

## Prerequisites

Make sure the following are installed and configured:

- [Docker & Docker Compose](https://docs.docker.com/compose/install/)
- [Kubernetes](https://kubernetes.io/docs/tasks/tools/) (Docker Desktop or minikube)
- [kubectl](https://kubernetes.io/docs/tasks/tools/) configured to the cluster
- [Conda](https://docs.conda.io/en/latest/) (optional for local dev)
- [Git](https://git-scm.com/)

>  All container images are already built and pushed to DockerHub, so no local builds are required.

---

## 1. Start the Kubernetes Cluster

If using Docker Desktop:

1. Open Docker Desktop
2. Enable the Kubernetes cluster
3. Verify it is running: Execute following commands for logs
- kubectl cluster-info
- kubectl get nodes
- kubectl get pods
---

## 2. Deploy Services to Kubernetes

### a) Apply deployment manifest:

```bash
cd fraud_detection
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
```
- This will deploy:

    - MLflow Tracking & Model Registry

    - FastAPI Inference Service

- Check pods:
```bash
kubectl get pods
kubectl get svc
```
### b) Run Kedro Pipelines

Execute the Kedro training pipelines:

```bash
kedro run
```

- Validates input data with Great Expectations

-  Performs feature engineering

- Trains the model

- Logs parameters, metrics, and artifacts to MLflow

- Registers the trained model in MLflow Registry

# c) Access MLflow UI

```bash
kubectl port-forward svc/mlflow-service 5000:5000
```
- Open the MLflow UI at: http://localhost:5000

- Review experiments and metrics

- Promote the best model to the Production alias

> The FastAPI service automatically loads the latest production model.

# d) Start the API Service
```bash
kubectl port-forward svc/loan-api-service 8080:80
```
- Verify logs to ensure the service is running:
```bash
kubectl logs -l app=loan-api -f
```
- Restart deployment if needed:
```bash
kubectl rollout restart deployment deployment_name
```
- Access the API at http://localhost:8080/predict for model inference.

---
## 3. Visualization
- Kedro Viz for pipeline visualization:
```bash
kedro viz
```
- MLflow UI to visualize experiment metrics, parameters, and artifacts

## 4. Notes & Tips

- Training and inference are decoupled; FastAPI loads the Production model automatically.

- Use kubectl logs and kubectl describe pod to troubleshoot.

- **Port-forwarding** allows local access to services running inside Kubernetes.

- Model promotion in MLflow does not require API redeployment.

## References

- **Kedro Documentation** —> https://docs.kedro.org/en/stable/  
- **MLflow Documentation** —> https://mlflow.org/docs/latest/  
- **FastAPI Documentation** —> https://fastapi.tiangolo.com/  
- **Kubernetes Basics** —> https://kubernetes.io/docs/tutorials/kubernetes-basics/
