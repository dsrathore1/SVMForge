import sys
from pathlib import Path
from time import perf_counter

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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

    search = create_svm_search()

    start = perf_counter()
    search.fit(x_train, y_train)
    elapsed = perf_counter() - start

    best_model = search.best_estimator_
    predictions = best_model.predict(x_validation)
    decision_scores = best_model.decision_function(x_validation)

    print("SVMForge Hyperparameter Tuning")
    print("==============================")
    print("Tuning status: PASSED")
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_validation)}")
    print("Cross-validation: Stratified 5-fold")
    print(f"Configurations evaluated: {len(search.cv_results_['params'])}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best mean CV F1-score: {search.best_score_:.4f}")
    print(f"Validation accuracy: {accuracy_score(y_validation, predictions):.4f}")
    print(
        "Validation precision: "
        f"{precision_score(y_validation, predictions, zero_division=0):.4f}"
    )
    print(
        "Validation recall: "
        f"{recall_score(y_validation, predictions, zero_division=0):.4f}"
    )
    print(f"Validation F1-score: {f1_score(y_validation, predictions):.4f}")
    print(f"Validation ROC-AUC: {roc_auc_score(y_validation, decision_scores):.4f}")
    print(f"Search duration (seconds): {elapsed:.2f}")
    print("Test set used during tuning: NO")


if __name__ == "__main__":
    main()
