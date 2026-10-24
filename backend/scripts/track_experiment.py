import sys
from pathlib import Path

import mlflow
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from svmforge.data import load_dataset
from svmforge.model import create_svm_search
from svmforge.preprocessing import split_dataset


def main() -> None:
    features, target = load_dataset()

    (
        x_train,
        x_validation,
        _x_test,
        y_train,
        y_validation,
        _y_test,
    ) = split_dataset(features, target, random_state=42)

    tracking_db = BACKEND_DIR / "mlflow.db"
    mlflow.set_tracking_uri(
        f"sqlite:///{tracking_db.as_posix()}"
    )
    mlflow.set_experiment("SVMForge-Experiments")

    search = create_svm_search()
    search.fit(x_train, y_train)

    model = search.best_estimator_
    predictions = model.predict(x_validation)
    decision_scores = model.decision_function(x_validation)

    with mlflow.start_run(run_name="tuned-svm-validation") as run:
        mlflow.log_param("dataset", "Breast Cancer Wisconsin Diagnostic")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("selection_metric", "f1")

        for name, value in search.best_params_.items():
            mlflow.log_param(name, str(value))

        mlflow.log_metric("best_cv_f1", float(search.best_score_))
        mlflow.log_metric(
            "validation_accuracy",
            float(accuracy_score(y_validation, predictions)),
        )
        mlflow.log_metric(
            "validation_precision",
            float(precision_score(y_validation, predictions, zero_division=0)),
        )
        mlflow.log_metric(
            "validation_recall",
            float(recall_score(y_validation, predictions, zero_division=0)),
        )
        mlflow.log_metric(
            "validation_f1",
            float(f1_score(y_validation, predictions, zero_division=0)),
        )
        mlflow.log_metric(
            "validation_roc_auc",
            float(roc_auc_score(y_validation, decision_scores)),
        )

        print("SVMForge MLflow Tracking")
        print("========================")
        print("Tracking status: PASSED")
        print("Experiment: SVMForge-Experiments")
        print(f"Run ID: {run.info.run_id}")
        print(f"Best parameters: {search.best_params_}")
        print(f"Cross-validation F1: {search.best_score_:.4f}")
        print(f"Validation F1: {f1_score(y_validation, predictions):.4f}")
        print(f"Tracking directory: {tracking_db}")
        print("Test set used for tuning: NO")


if __name__ == "__main__":
    main()
