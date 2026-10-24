# SVMForge

**An end-to-end machine learning and MLOps platform for reproducible SVM experimentation, model serving, and operational monitoring.**

SVMForge is a portfolio project that demonstrates how to develop, evaluate, track, serve, and monitor a machine learning model using a practical MLOps workflow.

The project uses Support Vector Machines (SVMs) for binary classification and combines a Python-based machine learning backend with a separate Next.js frontend.

> **Project status:** In development. Features are implemented progressively throughout the 14-day roadmap.

## Project Objectives

- Build a reproducible machine learning workflow.
- Compare Linear and RBF Support Vector Machine models.
- Use stratified data splitting and leakage-safe preprocessing.
- Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
- Track experiments and parameters using MLflow.
- Develop a FastAPI inference service.
- Build a responsive frontend for model exploration and operational visibility.
- Introduce monitoring with Prometheus and Grafana.
- Automate testing and build checks with GitHub Actions.
- Containerize the application with Docker Compose.

## Technology Stack

| Area | Technologies |
|---|---|
| Machine learning | Python, scikit-learn, pandas, NumPy |
| Model evaluation | Cross-validation, classification metrics |
| Experiment tracking | MLflow, SQLite |
| Backend API | FastAPI, Uvicorn |
| Backend quality | pytest, HTTPX, Ruff |
| Frontend | Next.js, React, JavaScript |
| Monitoring | Prometheus, Grafana |
| Automation | Git, GitHub Actions |
| Containerization | Docker, Docker Compose |

## Machine Learning Workflow

1. Load the Breast Cancer Wisconsin Diagnostic dataset.
2. Validate the dataset and check for missing or non-finite values.
3. Create reproducible, stratified training, validation, and test splits.
4. Apply feature scaling within the model pipeline to prevent data leakage.
5. Train a Linear SVM baseline.
6. Compare Linear and RBF kernels.
7. Tune hyperparameters using stratified five-fold cross-validation.
8. Track experiment parameters and metrics in MLflow.
9. Prepare the selected model for API-based inference.

The current split uses approximately 70% training data, 15% validation data, and 15% test data. The test set is reserved for final evaluation and is not used for model selection.

## Dataset

SVMForge currently uses the built-in **Breast Cancer Wisconsin Diagnostic dataset** provided by scikit-learn.

- Samples: 569
- Numeric input features: 30
- Target classes: 2
- Class distribution: 212 samples in class 0 and 357 in class 1

This dataset is used as an educational machine learning benchmark. SVMForge is not a medical diagnostic system and must not be used to make clinical decisions.

## Local Development

### Prerequisites

- Python 3.10 or newer, compatible with the project's dependencies
- Node.js and npm
- Git
- Windows PowerShell or a compatible terminal

Docker is planned for a later project phase.

### Backend Setup

From the repository root:

```powershell
cd backend

py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the dataset validation:

```powershell
.\.venv\Scripts\python.exe scripts\validate_dataset.py
```

Run the data preparation check:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py
```

Run the Linear SVM baseline:

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py
```

Compare SVM kernels:

```powershell
.\.venv\Scripts\python.exe scripts\compare_svm.py
```

Run hyperparameter tuning:

```powershell
.\.venv\Scripts\python.exe scripts\tune_svm.py
```

Run the automated tests and lint checks:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
.\.venv\Scripts\python.exe -m ruff check svmforge tests scripts
```

### Frontend Setup

From the repository root:

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the frontend.

### MLflow

From the `backend` directory, after running the experiment tracking script:

```powershell
.\.venv\Scripts\python.exe scripts\track_experiment.py
```

Start the MLflow interface in a separate terminal:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri "sqlite:///mlflow.db" --host 127.0.0.1 --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

The MLflow database and experiment artifacts are local development data and should not be committed to Git.

## Frontend Vision

The planned frontend is organized around two main sections:

- **Model Studio:** predictions, model metrics, model comparison, and evaluation visualizations.
- **Operations Console:** API health, service status, request metrics, and monitoring information.

The interface will use a responsive design with lavender/violet accents and a light/dark theme toggle.

These capabilities will be marked as completed only after implementation and verification.

## MLOps Roadmap

- [x] Development environment and Git foundation
- [x] Dataset validation and quality tests
- [x] Reproducible stratified data splits
- [x] Leakage-safe feature scaling
- [x] Linear SVM baseline
- [x] Linear versus RBF comparison
- [x] Hyperparameter tuning with cross-validation
- [x] Initial MLflow experiment tracking
- [ ] Save and version the selected model artifact
- [ ] Build and test the FastAPI inference API
- [ ] Develop Model Studio and Operations Console
- [ ] Integrate frontend and backend
- [ ] Add Prometheus metrics and Grafana dashboards
- [ ] Add Docker Compose
- [ ] Add GitHub Actions CI checks
- [ ] Complete end-to-end tests and release documentation

## Engineering Practices

SVMForge follows these practices:

- Small, focused changes and Conventional Commit messages.
- Reproducible splits and controlled model evaluation.
- Automated tests and lint checks.
- Configuration separated from implementation where appropriate.
- Local experiment data and environment files excluded from version control.
- Test-set isolation during model selection.

## Disclaimer

This is an educational portfolio project for demonstrating machine learning and MLOps engineering. Its dataset and models are not intended for medical use or real-world clinical decision-making.

## License

No license has been selected yet. Add a `LICENSE` file before describing the repository as open source.

## Project Status

SVMForge is being developed as a 14-day, step-by-step MLOps project. Documentation, architecture, and feature status will be updated as implementation progresses.
