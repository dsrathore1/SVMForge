import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from svmforge.data import load_dataset
from svmforge.preprocessing import (
    fit_preprocessor,
    split_dataset,
)


def main() -> None:
    features, target = load_dataset()

    (
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test,
    ) = split_dataset(features, target)

    scaler = fit_preprocessor(x_train)

    # Transform every split using the scaler fitted on training data.
    x_train_scaled = scaler.transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_test)

    print("SVMForge Data Preparation")
    print("=========================")
    print("Preparation status: PASSED")
    print(f"Total samples: {len(features)}")
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_validation)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Feature count: {features.shape[1]}")
    print(f"Training classes: {y_train.value_counts().sort_index().to_dict()}")
    print(
        "Validation classes: "
        f"{y_validation.value_counts().sort_index().to_dict()}"
    )
    print(f"Test classes: {y_test.value_counts().sort_index().to_dict()}")
    print(f"Scaled training shape: {x_train_scaled.shape}")
    print(f"Scaled validation shape: {x_validation_scaled.shape}")
    print(f"Scaled test shape: {x_test_scaled.shape}")
    print("Preprocessing fitted on training data only: YES")


if __name__ == "__main__":
    main()
