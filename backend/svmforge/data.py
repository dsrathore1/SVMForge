import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


class DatasetValidationError(ValueError):
    """Raised when a dataset fails validation."""


def validate_dataset(
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Validate feature and target data before model training."""
    if not isinstance(features, pd.DataFrame):
        raise DatasetValidationError(
            "Features must be a pandas DataFrame."
        )

    if not isinstance(target, pd.Series):
        raise DatasetValidationError(
            "Target must be a pandas Series."
        )

    if features.empty or target.empty:
        raise DatasetValidationError(
            "Features and target must not be empty."
        )

    if len(features) != len(target):
        raise DatasetValidationError(
            "Feature and target row counts do not match."
        )

    if features.columns.has_duplicates:
        raise DatasetValidationError(
            "Feature column names must be unique."
        )

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in features.dtypes
    ):
        raise DatasetValidationError(
            "All input features must be numeric."
        )

    if features.isna().any().any():
        raise DatasetValidationError(
            "Features contain missing values."
        )

    if target.isna().any():
        raise DatasetValidationError(
            "Target contains missing values."
        )

    try:
        values = features.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError(
            "Features cannot be converted to numeric values."
        ) from exc

    if not np.isfinite(values).all():
        raise DatasetValidationError(
            "Features contain infinite or non-finite values."
        )

    if target.nunique() != 2:
        raise DatasetValidationError(
            "Expected exactly two target classes."
        )


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the built-in binary classification dataset."""
    dataset = load_breast_cancer(as_frame=True)

    features = dataset.data.copy()
    target = dataset.target.copy()

    validate_dataset(features, target)

    return features, target
