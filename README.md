# 🚀 End-to-End MLOps Pipeline for Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-brightgreen)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-teal)
![Docker](https://img.shields.io/badge/Docker-Containerization-blue)
![Prometheus](https://img.shields.io/badge/Monitoring-Prometheus-orange)
![Grafana](https://img.shields.io/badge/Observability-Grafana-yellow)
![CI/CD](https://img.shields.io/badge/GitHub-Actions%20CI%2FCD-lightgrey)

---

## 📌 Project Overview

This project implements an **end-to-end Machine Learning Operations (MLOps) pipeline** for **credit card fraud detection** using the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

The system demonstrates **modern MLOps best practices** including:
✅ Experiment tracking
✅ Model registry
✅ Containerized deployment
✅ Real-time inferencing
✅ Production monitoring

---

## ✨ Key Features

1. **Machine Learning**: Binary classification using XGBoost and Random Forest
2. **Experiment Tracking**: Full MLflow integration with model registry
3. **Real-time API**: FastAPI REST service for fraud prediction
4. **Containerization**: Docker containers for all services
5. **Cloud Deployment**: EC2 machine with CI/CD automation
6. **Monitoring**: Prometheus + Grafana observability stack
7. **CI/CD Pipeline**: GitHub Actions for automated deployment

---

## ⚙️ Local Development Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/onehungrybird/mlops_fraud2.git
cd mlops_fraud2
```

### 2️⃣ Run the automated setup script

```bash
chmod +x mlflow_setup.sh
./mlflow_setup.sh
```

This script will:

* Update system packages
* Install Python dependencies and build tools
* Create a virtual environment
* Install all required Python packages from `requirements.txt`
* Configure firewall rules (MLflow: 5000, FastAPI: 8000)
* Set up the complete MLflow environment

### 3️⃣ Download the dataset

Place the Kaggle dataset in:

```
data/raw/creditcard.csv
```

### 4️⃣ Activate environment & start services

```bash
source venv/bin/activate
```

Start MLflow server:

```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow_runs/mlflow.db \
    --default-artifact-root file:mlflow_runs \
    --host 0.0.0.0 \
    --port 5000
```

---

## 🐳 Docker Setup (Recommended for Production)

Build and start all services:

```bash
docker-compose up --build
```

Verify services:

```bash
docker-compose ps
```

### Available Services

* **MLflow UI** → [http://98.80.224.211:5000](http://98.80.224.211:5000)
* **FastAPI** → [http://98.80.224.211:8000](http://98.80.224.211:8000)
* **Prometheus** → [http://98.80.224.211:9090](http://98.80.224.211:9090)
* **Grafana** → [http://98.80.224.211:3000](http://98.80.224.211:3000) (user: `admin`, pass: `grafana_password`)

---

## 🎯 Model Training

Run training script:

```bash
python3 -m notebooks.train \
    --data-path data/raw/creditcard.csv \
    --experiment-name credit-card-fraud-detection
```

Predict via Swagger UI → [http://98.80.224.211:8000/docs](http://98.80.224.211:8000/docs)

---

## 📊 API Documentation

| Endpoint   | Method | Description                          | Response |
| ---------- | ------ | ------------------------------------ | -------- |
| `/`        | GET    | API root with service info           | JSON     |
| `/health`  | GET    | Health check (service + model state) | JSON     |
| `/predict` | POST   | Fraud prediction endpoint            | JSON     |
| `/metrics` | GET    | Prometheus metrics for monitoring    | Metrics  |
| `/docs`    | GET    | Swagger UI for API testing           | OpenAPI  |

---

## 🛑 Cleanup

```bash
docker compose down
```

---

## 📌 Implementation Matrix

| Feature                               | Status / Notes               |
| ------------------------------------- | ---------------------------- |
| Environment & reproducibility         | ✅ Done                       |
| Data versioning                       | ❌ No (S3 access unavailable) |
| Data preprocessing                    | ✅ Done                       |
| Model development & training (MLflow) | ✅ Done                       |
| Versioning, model registry & CI/CD    | ✅ Done                       |
| MLflow model promotion                | ✅ Done                       |
| Model serialization                   | ✅ Done                       |
| Real-time inferencing                 | ✅ Done                       |
| Cloud Deployment                      | ✅ Done                       |
| IaC (Infrastructure as Code)          | ❌ No (AWS access required)   |
| API Monitoring                        | ✅ Done                       |
| Model Monitoring                      | ✅ Done                       |
| Health checks                         | ✅ Done                       |
| Project README.md                     | ✅ Done                       |
| Testing                               | ✅ Done                       |

---

## ✅ Conclusion

This project showcases how **ML + DevOps + Cloud + Monitoring** can be combined into a **production-ready MLOps pipeline**.
It’s designed to **impress technical reviewers** by demonstrating **end-to-end ownership**: from dataset → training → deployment → monitoring.

---
