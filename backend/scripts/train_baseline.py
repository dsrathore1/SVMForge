import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from svmforge.data import load_dataset
from svmforge.model import create_linear_svm
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
    ) = split_dataset(features, target)

    model = create_linear_svm()
    model.fit(x_train, y_train)

    predictions = model.predict(x_validation)

    print("SVMForge Linear SVM Baseline")
    print("============================")
    print("Training status: PASSED")
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_validation)}")
    print("Kernel: linear")
    print("C: 1.0")
    print(f"Validation accuracy: {accuracy_score(y_validation, predictions):.4f}")
    print(f"Validation precision: {precision_score(y_validation, predictions, zero_division=0):.4f}")
    print(f"Validation recall: {recall_score(y_validation, predictions, zero_division=0):.4f}")
    print(f"Validation F1-score: {f1_score(y_validation, predictions, zero_division=0):.4f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_validation, predictions))
    print("Classification report:")
    print(
        classification_report(
            y_validation,
            predictions,
            zero_division=0,
        )
    )
    print("Test set used for model selection: NO")


if __name__ == "__main__":
    main()
