This project implements an end-to-end machine learning operations (MLOps) pipeline for credit card fraud detection using the 
Kaggle Credit Card Fraud Detection dataset. The system demonstrates modern MLOps best practices including experiment tracking, 
model registry, containerized deployment, and production monitoring.

Key Features
 1. Machine Learning: Binary classification using XGBoost and Random Forest
 2. Experiment Tracking: Complete MLflow integration with model registry
 3. Real-time API: FastAPI REST service for fraud prediction
 4. Containerization: Docker containers for all services
 5. Cloud Deployment: Ec2 machine with CI/CD
 6. Monitoring: Prometheus + Grafana observability stack
 7. CI/CD Pipeline: GitHub Actions for automated deployment

Local Development Setup:
1. clone the repo
	```git clone https://github.com/onehungrybird/mlops_fraud2.git```
	```cd mlops_fraud2```

2. Run the automated setup script
	```chmod +x mlflow_setup.sh```
	```./mlflow_setup.sh```
	
	Update system packages  
	Install Python dependencies and build tools  
	Create a virtual environment  
	Install all required Python packages from requirements.txt  
	Configure firewall rules for MLflow (port 5000) and FastAPI (port 8000)  
	Set up the complete MLflow environment  

3. Download the dataset and place it in `data/raw` directory  
4. Activate the environment and start services  

	# Activate virtual environment  
	```source venv/bin/activate```  

	# Start MLflow server  
	```mlflow server \
		--backend-store-uri sqlite:///mlflow_runs/mlflow.db \
		--default-artifact-root file:mlflow_runs \
		--host 0.0.0.0 \
		--port 5000```  

5. Docker Setup (Recommended for Production)  
	```docker-compose up --build```  
	
	verify services are running  
	```docker-compose ps```  
	
6. The following services will be available:

	```MLflow UI: http://98.80.224.211:5000```  
	```FastAPI: http://98.80.224.211:8000```  
	```Prometheus: http://98.80.224.211:9090```  
	```Grafana: http://98.80.224.211:3000 (admin/grafana_password)```  
	
7. Training a Model  
	After running the setup script and activating the environment:  
	# Using the training script directly  
	```python3 -m notebooks.train --data-path data/raw/creditcard.csv --experiment-name credit-card-fraud-detection```  
	
	# make prediction on (http://98.80.224.211:8000/docs ) through UI.  
	
8. MLFlow model registry (check registered model, if not then manually register it)  
	```http://98.80.224.211:5000```  
	
9. API Health check  
	```http://localhost:8000/health```  


API Documentation  
Endpoints:  

**GET /**  
Description: API root with service information  
Response: JSON with API details and available endpoints  

**GET /health**  
Description: Health check endpoint  
Response: Service status and model loading state  

**POST /predict**  
Description: Fraud prediction endpoint  
Request Body: Transaction features (JSON)  
Response: Fraud probability and classification  

**GET /metrics**  
Description: Prometheus metrics for monitoring  
Response: Application metrics in Prometheus format  

**GET /docs**  
Description: Interactive API documentation (Swagger UI)  
Response: OpenAPI documentation interface  

10. Cleanup  
	```docker compose down```  

11. Implementations  

| Feature                              | Status / Notes                               |
|--------------------------------------|----------------------------------------------|
| Environment & reproducibility        | ✅ Done                                      |
| Data versioning                      | ❌ No (didn't have S3 access)                 |
| Data Preprocessing                   | ✅ Done                                      |
| Model development & training (MLflow)| ✅ Done                                      |
| Versioning, MLflow model registry & CI/CD | ✅ Done                                |
| MLflow model promotion               | ✅ Done                                      |
| Model serialization                  | ✅ Done                                      |
| Real-time inferencing                | ✅ Done                                      |
| Cloud Deployment                     | ✅ Done                                      |
| IaC (Infrastructure as Code)         | ❌ No (needed AWS access, not available)      |
| API Monitoring                       | ✅ Done                                      |
| Model Monitoring                     | ✅ Done                                      |
| Health checks                        | ✅ Done                                      |
| Project README.md                    | ✅ Done                                      |
| Testing                              | ✅ Done                                      |
