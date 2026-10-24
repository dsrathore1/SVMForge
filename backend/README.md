# SVMForge Backend

The SVMForge backend contains the machine learning workflow, dataset validation, preprocessing, SVM experiments, automated tests, and MLflow experiment tracking.

**Status:** Machine learning experimentation and initial MLflow tracking are implemented. FastAPI model serving and operational endpoints are planned for subsequent project tasks.

## Technology Stack

- Python
- scikit-learn
- pandas and NumPy
- MLflow with SQLite tracking
- FastAPI and Uvicorn — planned for inference serving
- pytest and HTTPX — for automated testing as API functionality is added
- Ruff — linting and code-quality checks

## Dataset

The current dataset is scikit-learn's Breast Cancer Wisconsin Diagnostic dataset.

| Property | Value |
|---|---|
| Samples | 569 |
| Numeric features | 30 |
| Target classes | 2 |
| Class 0 | 212 |
| Class 1 | 357 |

The dataset is used for educational binary-classification experiments, not clinical diagnosis.

## Data Preparation

The backend validates feature and target data before training. It checks for empty inputs, mismatched row counts, missing values, non-finite feature values, and invalid target-class counts.

The workflow creates reproducible, stratified splits:

- 70% training data
- 15% validation data
- 15% test data

A `StandardScaler` is fitted on training data only. Model pipelines apply scaling within each training or cross-validation fold, helping prevent data leakage.

## SVM Experiments

The current experiments include:

1. Linear SVM baseline
2. Linear versus RBF kernel comparison
3. Hyperparameter tuning with stratified five-fold cross-validation

The tuning search compares selected values of `C` and, for RBF, `gamma`. The primary selection metric is F1-score.

Metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The test set is reserved for final evaluation and is not used to select model parameters.

## Local Setup

From the repository root:

```powershell
cd backend

py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the virtual environment already exists, reuse it instead of creating another.

## Run the Workflow

Run dataset validation:

```powershell
.\.venv\Scripts\python.exe scripts\validate_dataset.py
```

Prepare and validate the data splits:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py
```

Train the Linear SVM baseline:

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py
```

Compare Linear and RBF kernels:

```powershell
.\.venv\Scripts\python.exe scripts\compare_svm.py
```

Tune SVM hyperparameters:

```powershell
.\.venv\Scripts\python.exe scripts\tune_svm.py
```

Track the tuned experiment using MLflow:

```powershell
.\.venv\Scripts\python.exe scripts\track_experiment.py
```

## MLflow

MLflow records experiment parameters and evaluation metrics in a local SQLite database.

Start the tracking interface from the backend directory:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri "sqlite:///mlflow.db" --host 127.0.0.1 --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

The local tracking database and artifacts are generated development data. Keep them out of version control.

## Testing and Linting

Run all backend tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Run Ruff:

```powershell
.\.venv\Scripts\python.exe -m ruff check svmforge tests scripts
```

Both checks should pass before committing changes.

## Planned Backend Work

- Save the selected model and associated metadata as an MLflow artifact.
- Implement a FastAPI prediction endpoint.
- Add request validation and consistent error responses.
- Expose model information and service health.
- Add API tests using pytest and HTTPX.
- Add Prometheus metrics for service monitoring.
- Integrate the backend with the Next.js frontend.
- Add containerization and automated CI checks.

## Development Guidelines

- Keep model preprocessing inside scikit-learn pipelines.
- Avoid fitting preprocessing on validation or test data.
- Never use the test set for hyperparameter selection.
- Add tests when introducing new functionality.
- Use small Conventional Commits, for example:
  - `feat(ml): add linear SVM baseline`
  - `test(ml): verify data split integrity`
  - `feat(mlops): track SVM experiments with MLflow`
- Do not commit secrets, virtual environments, local MLflow databases, or generated artifacts.

## Disclaimer

This backend is an educational machine learning and MLOps project. It is not a medical diagnostic service.
