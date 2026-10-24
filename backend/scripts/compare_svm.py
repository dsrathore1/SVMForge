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
from svmforge.model import (
    create_linear_svm,
    create_rbf_svm,
)
from svmforge.preprocessing import split_dataset


def evaluate_model(name, model, x_train, y_train, x_validation, y_validation):
    start = perf_counter()
    model.fit(x_train, y_train)
    training_seconds = perf_counter() - start

    predictions = model.predict(x_validation)
    probabilities = model.decision_function(x_validation)

    return {
        "model": name,
        "accuracy": accuracy_score(y_validation, predictions),
        "precision": precision_score(
            y_validation, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_validation, predictions, zero_division=0
        ),
        "f1": f1_score(y_validation, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_validation, probabilities),
        "training_seconds": training_seconds,
    }


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

    models = [
        ("Linear SVM", create_linear_svm()),
        ("RBF SVM", create_rbf_svm()),
    ]

    results = [
        evaluate_model(
            name,
            model,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )
        for name, model in models
    ]

    print("SVMForge Kernel Comparison")
    print("==========================")
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_validation)}")
    print("Test set used: NO")
    print()

    headers = (
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "Train (s)",
    )
    print(
        f"{headers[0]:<13} {headers[1]:>9} {headers[2]:>9} "
        f"{headers[3]:>9} {headers[4]:>9} {headers[5]:>9} "
        f"{headers[6]:>10}"
    )

    for result in results:
        print(
            f"{result['model']:<13} "
            f"{result['accuracy']:>9.4f} "
            f"{result['precision']:>9.4f} "
            f"{result['recall']:>9.4f} "
            f"{result['f1']:>9.4f} "
            f"{result['roc_auc']:>9.4f} "
            f"{result['training_seconds']:>10.4f}"
        )

    best_model = max(results, key=lambda result: result["f1"])
    print()
    print(f"Best validation F1-score: {best_model['model']}")
    print(f"F1-score: {best_model['f1']:.4f}")
    print("Selection based on validation F1-score only.")


if __name__ == "__main__":
    main()
