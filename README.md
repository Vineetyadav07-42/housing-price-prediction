# California House Price Prediction

An end-to-end machine learning project that predicts California house prices using **XGBoost**, with custom feature engineering, data preprocessing, FastAPI, Docker, CI/CD, and cloud deployment.

## Live Demo

The trained model is deployed as a REST API using FastAPI and Docker on **AWS EC2**.

**Live API:** http://13.234.69.92:8000

**Interactive API Documentation:** http://13.234.69.92:8000/docs

The Swagger UI allows users to send house features and receive a predicted house value.

---

## Project Overview

The objective of this project is to build a machine learning regression system that predicts the **median house value** based on geographical, demographic, and housing-related features.

The project follows a complete machine learning and deployment workflow:

```text
Data
  ↓
Exploratory Data Analysis
  ↓
Feature Engineering
  ↓
Data Preprocessing
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Hyperparameter Tuning
  ↓
Final Pipeline
  ↓
FastAPI
  ↓
Docker
  ↓
CI/CD Pipeline
  ↓
AWS Deployment
```

---

## Dataset

The project uses the **California Housing dataset**.

The target variable is:

```text
median_house_value
```

### Input Features

* `longitude`
* `latitude`
* `housing_median_age`
* `total_rooms`
* `total_bedrooms`
* `population`
* `households`
* `median_income`
* `ocean_proximity`

---

## Feature Engineering

A custom Scikit-learn transformer was created to generate additional features.

The transformer is implemented in:

```text
src/FE.py
```

### Rooms per Household

```text
rooms_per_households = total_rooms / households
```

### Bedrooms-to-Room Ratio

```text
bedrooms_room_ratio = total_bedrooms / total_rooms
```

### Population per Household

```text
population_per_households = population / households
```

The custom `FeatureEngineering` transformer is integrated directly into the final machine learning pipeline, ensuring that the same feature transformations are applied during both training and inference.

---

## Data Preprocessing

The preprocessing pipeline uses Scikit-learn's `ColumnTransformer`.

### Numerical Features

Numerical features are processed using:

1. `SimpleImputer(strategy="median")`
2. `StandardScaler()`

### Categorical Features

The `ocean_proximity` feature is processed using:

1. `SimpleImputer(strategy="most_frequent")`
2. `OneHotEncoder()`

The preprocessing steps are included inside the final pipeline so that the same transformations are automatically applied when making predictions.

---

## Model Selection

Multiple regression models were evaluated using cross-validation.

| No. | Model                   |       CV RMSE |      CV R² |
| --: | ----------------------- | ------------: | ---------: |
|   1 | Linear Regression       |     68,276.93 |     0.6516 |
|   2 | Decision Tree Regressor |     70,835.44 |     0.6244 |
|   3 | Random Forest Regressor |     51,041.82 |     0.8050 |
|   4 | **XGBoost Regressor**   | **48,128.46** | **0.8268** |

### Final Model

**XGBoost Regressor** performed best, achieving the lowest cross-validation RMSE and highest cross-validation R² among the evaluated models.

---

## Hyperparameter Tuning

`GridSearchCV` was used to tune the XGBoost model's hyperparameters.

The selected configuration was:

```python
XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=1,
    colsample_bytree=0.8
)
```

The final machine learning pipeline consists of:

```text
Feature Engineering
        ↓
Data Preprocessing
        ↓
XGBoost Regressor
```

The complete pipeline is saved using Joblib:

```text
models/final_pipeline.pkl
```

Saving the entire pipeline ensures that feature engineering and preprocessing are automatically applied during inference.

---

## Project Structure

```text
Housing_Project_Resume/

│
├── data/
│   └── housing.csv
│
├── models/
│   └── final_pipeline.pkl
│
├── notebooks/
│   ├── best_model_selection.ipynb
│   └── fine_tune_bestmodel.ipynb
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── FE.py
│   ├── preprocessing_model.py
│   └── training.py
│
├── .github/
│   └── workflows/
│       └── deploy.yml
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## Technologies Used

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LinearRegression
* DecisionTreeRegressor
* RandomForestRegressor
* Joblib
* GridSearchCV

### API

* FastAPI
* Pydantic
* Uvicorn

### Deployment & DevOps

* Docker
* Git
* GitHub
* GitHub Actions
* CI/CD
* AWS ECR
* AWS EC2

---

# Running the Project Locally

## 1. Clone the Repository

```bash
git clone https://github.com/Vineetyadav07-42/housing-price-prediction.git
cd housing-price-prediction
```

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Train the Model

To train the model and generate the final pipeline:

```bash
python -m src.training
```

The trained pipeline will be saved to:

```text
models/final_pipeline.pkl
```

---

# Run the FastAPI Application

Start the API:

```bash
uvicorn src.api:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Open the interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Making Predictions

The prediction endpoint is:

```text
POST /predict
```

### Example Request

```json
{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
}
```

### Example Response

```json
{
    "predicted_house_value": 439603.90625
}
```

The prediction depends on the trained model and input features.

---

# Running with Docker

The application is containerized using Docker.

## Build the Docker Image

From the project root:

```bash
docker build -t housing-price-api .
```

## Run the Container

```bash
docker run -p 8000:8000 housing-price-api
```

The API will then be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

# CI/CD Pipeline

The project uses **GitHub Actions** to automate the build and deployment process.

The workflow configuration is located at:

```text
.github/workflows/deploy.yml
```

The CI/CD pipeline  automates the deployment workflow after changes are pushed to the GitHub repository.

The deployment process includes:

```text
Code Push to GitHub
        ↓
GitHub Actions
        ↓
Build Docker Image
        ↓
Push Docker Image to AWS ECR
        ↓
Connect to AWS EC2
        ↓
Pull Updated Docker Image
        ↓
Stop/Replace Previous Container
        ↓
Run Updated Container
        ↓
Live FastAPI Application
```

This eliminates the need to manually perform the Docker build, ECR push, and EC2 deployment steps after every code change.

---

# Cloud Deployment

The application is deployed using **AWS ECR and AWS EC2**.

The deployment architecture is:

```text
GitHub Repository
       ↓
GitHub Actions CI/CD
       ↓
Docker Build
       ↓
AWS ECR
       ↓
AWS EC2
       ↓
Docker Container
       ↓
FastAPI
       ↓
Saved XGBoost Pipeline
       ↓
Prediction
```

### AWS Components

* **AWS ECR** – Stores the Docker image.
* **AWS EC2** – Hosts and runs the Docker container.
* **GitHub Actions** – Automates the CI/CD deployment process.

### Live Application

**API:** http://13.234.69.92:8000

**Swagger UI:** http://13.234.69.92:8000/docs

The application uses an **AWS Elastic IP** to provide a stable public IP address for the deployed API.

---

# Key Machine Learning & Engineering Concepts

This project demonstrates practical knowledge of:

* Exploratory Data Analysis
* Regression
* Feature Engineering
* Missing Value Imputation
* Feature Scaling
* One-Hot Encoding
* Scikit-learn Pipelines
* Column Transformers
* Cross-Validation
* Hyperparameter Tuning
* XGBoost
* LinearRegression
* RandomForestRegressor
* DecisionTreeRegressor
* GridSearchCV
* Model Evaluation
* Model Serialization with Joblib
* REST API Development
* Pydantic Data Validation
* Docker Containerization
* CI/CD
* GitHub Actions
* AWS ECR
* AWS EC2
* Elastic IP
* Cloud Deployment
* Git & GitHub

---

# Future Improvements

Potential improvements include:

* Add automated unit and integration tests
* Add API logging
* Add model monitoring
* Add data validation
* Add model versioning
* Add authentication
* Build a web-based frontend
* Improve cloud infrastructure and scalability

---

# Author

**Vineet Yadav**

Machine Learning / ML Engineering Portfolio Project

**GitHub:** https://github.com/Vineetyadav07-42
