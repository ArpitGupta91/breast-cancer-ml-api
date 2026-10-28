# Breast Cancer Classification API

An end-to-end Machine Learning project that trains a classification model in Google Colab, serves predictions through FastAPI, packages the application using Docker, and automates testing and image building with GitHub Actions.

## Project Overview

This project demonstrates the complete machine learning deployment lifecycle:

1. Select and explore a public dataset from Kaggle.
2. Train and evaluate a machine learning classification model in Google Colab.
3. Save the trained model using Joblib.
4. Develop a REST API using FastAPI.
5. Test API endpoints using automated tests.
6. Containerize the application using Docker.
7. Configure a GitHub Actions Continuous Integration (CI) pipeline.

**Important:** This project is for educational purposes only. It is not a clinically validated medical diagnostic system and must not be used to make healthcare decisions.

## Dataset

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Data Set

**Source:** [Kaggle — Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

**Original research dataset:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

The dataset contains measurements computed from digitized images of fine needle aspirates of breast masses.

### Dataset Details

- Samples: 569
- Numerical input features: 30
- Target column: `diagnosis`
- Benign class: `B` (encoded as 0)
- Malignant class: `M` (encoded as 1)
- Task: Binary classification

The 30 features describe cell nuclei through measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

The dataset also includes an identifier column, which is excluded from model training.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Development environment | Google Colab |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Model serialization | Joblib |
| REST API | FastAPI |
| ASGI server | Uvicorn |
| API testing | Pytest, FastAPI TestClient |
| Containerization | Docker |
| Version control | Git and GitHub |
| Continuous Integration | GitHub Actions |

## Machine Learning Methodology

The model was developed in Google Colab using the following workflow:

1. Load the Kaggle CSV dataset.
2. Inspect the data, target distribution, and missing values.
3. Remove the identifier and unnecessary unnamed columns.
4. Separate the 30 numerical features from the target.
5. Encode the target labels as 0 and 1.
6. Split the dataset into training and testing sets using an 80:20 ratio and stratification.
7. Train a `RandomForestClassifier`.
8. Evaluate the model using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
9. Save the fitted model as `artifacts/model.joblib`.

### Model Evaluation

Record the actual results produced by your Colab notebook below before submission.

| Metric | Result |
|---|---|
| Accuracy | Add your measured result |
| ROC-AUC | Add your measured result |
| Precision | Add your measured classification report value |
| Recall | Add your measured classification report value |
| F1-score | Add your measured classification report value |

For this classification task, malignant-class recall and false negatives deserve particular attention.

## Project Structure

```text
breast-cancer-ml-api/
├── app/
│   └── main.py
├── artifacts/
│   └── model.joblib
├── tests/
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── .dockerignore
├── Dockerfile
├── requirements.txt
└── README.md
```

## API Endpoints

The FastAPI application loads the saved model and exposes the following endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API homepage and basic status |
| GET | `/health` | Check application and model availability |
| POST | `/predict` | Generate a classification prediction |
| GET | `/docs` | Interactive Swagger API documentation |

### Health Check

Request:

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Prediction Endpoint

Request:

```http
POST /predict
Content-Type: application/json
```

The request body must contain a `features` object with all 30 numerical features. The feature names must match those expected by the saved model.

Example structure:

```json
{
  "features": {
    "radius_mean": 0.0,
    "texture_mean": 0.0
  }
}
```

**Note:** This is an abbreviated illustration, not a complete valid request. The actual API requires all 30 feature names. Use the exact names generated from the trained model.

A successful response contains the predicted class, numeric class identifier, and class probabilities. Missing or unexpected feature names are rejected.

- `benign`: predicted class 0
- `malignant`: predicted class 1

The model output is an educational classification result, not a medical diagnosis.

## Running Locally

### Prerequisites

Install Python, Git, and Docker Desktop if you want to run the containerized version.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-ml-api.git
cd breast-cancer-ml-api
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the API

```bash
python -m uvicorn app.main:app --reload
```

Open the following URLs in your browser:

- Homepage: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- Interactive API documentation: http://127.0.0.1:8000/docs

The trained model must exist at `artifacts/model.joblib` before starting the API.

## Running Automated Tests

The project includes automated tests for the homepage, health endpoint, prediction endpoint, and invalid feature input.

Run the tests from the project root:

```bash
python -m pytest -v
```

The current local test suite contains four tests. All four passed in the development environment.

The prediction test uses placeholder numerical inputs to verify API behavior; it is not a measure of model accuracy.

## Running with Docker

Make sure Docker Desktop is running.

### Build the Docker image

```bash
docker build -t breast-cancer-ml-api .
```

### Start the container

```bash
docker run --name breast-cancer-api -p 8000:8000 breast-cancer-ml-api
```

### Verify the container

Open:

http://127.0.0.1:8000/health

To stop and remove the container, use another terminal:

```bash
docker stop breast-cancer-api
docker rm breast-cancer-api
```

The Docker image packages the API, Python dependencies, and serialized model together.

## Continuous Integration with GitHub Actions

The workflow is located at:

`.github/workflows/ci.yml`

It is configured to run on:

- Pushes to the `main` branch
- Pull requests targeting `main`
- Manual workflow dispatch

The pipeline performs these steps:

1. Checks out the repository.
2. Sets up Python.
3. Installs the dependencies listed in `requirements.txt`.
4. Runs the automated Pytest suite.
5. Builds the Docker image.

The workflow is considered successful when all required steps pass. Check the **Actions** tab in the GitHub repository to view the execution logs and final status.

## Reproducibility and Compatibility

The model was serialized with Joblib. The scikit-learn version used to load a serialized model should match the version used to train and save it.

Use compatible, explicitly pinned dependency versions for reproducible local and CI builds. The dependency configuration should be checked against the model's training environment before release.

Only load trusted Joblib model files.

## Limitations

- The dataset is relatively small and represents a specific diagnostic dataset.
- Model performance on this dataset does not establish clinical effectiveness.
- Predictions depend on the quality and consistency of input measurements.
- The API does not replace professional medical evaluation.
- A successful API response confirms that the application can return a prediction; it does not establish predictive accuracy.

## Future Improvements

- Add more comprehensive input validation and boundary checks.
- Compare Random Forest with other classification algorithms.
- Add test coverage for malformed requests and model-loading failures.
- Add model versioning and experiment tracking.
- Add API logging and monitoring.
- Publish a container image to a registry if required.

## Acknowledgements

Dataset: UCI Machine Learning Repository, *Breast Cancer Wisconsin (Diagnostic)*.

Kaggle dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## License

The dataset's original UCI record specifies a CC BY 4.0 license. Review the dataset page and attribution requirements before redistributing dataset contents.

The license for this project's source code should be specified separately by the repository owner.
