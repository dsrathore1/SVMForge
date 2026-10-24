from typing import TypeAlias

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DatasetSplit: TypeAlias = tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = 42,
) -> DatasetSplit:
    """Create reproducible, stratified 70/15/15 data splits."""
    if len(features) != len(target):
        raise ValueError("Feature and target row counts do not match.")

    if features.empty:
        raise ValueError("Dataset must not be empty.")

    if target.nunique() != 2:
        raise ValueError("Expected exactly two target classes.")

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=random_state,
        stratify=target,
    )

    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    return (
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test,
    )


def fit_preprocessor(
    training_features: pd.DataFrame,
) -> StandardScaler:
    """Fit feature scaling using training data only."""
    if training_features.empty:
        raise ValueError("Training features must not be empty.")

    scaler = StandardScaler()
    scaler.fit(training_features)

    return scaler
