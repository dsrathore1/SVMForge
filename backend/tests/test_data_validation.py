import numpy as np
import pandas as pd
import pytest

from svmforge.data import (
    DatasetValidationError,
    load_dataset,
    validate_dataset,
)


def test_builtin_dataset_has_expected_shape():
    features, target = load_dataset()

    assert features.shape == (569, 30)
    assert len(target) == 569
    assert target.nunique() == 2


def test_valid_dataset_passes_validation():
    features = pd.DataFrame(
        {"feature_a": [1.0, 2.0, 3.0, 4.0]}
    )
    target = pd.Series([0, 0, 1, 1])

    validate_dataset(features, target)


def test_missing_feature_value_is_rejected():
    features = pd.DataFrame(
        {"feature_a": [1.0, np.nan, 3.0, 4.0]}
    )
    target = pd.Series([0, 0, 1, 1])

    with pytest.raises(
        DatasetValidationError,
        match="missing values",
    ):
        validate_dataset(features, target)


def test_infinite_feature_value_is_rejected():
    features = pd.DataFrame(
        {"feature_a": [1.0, 2.0, np.inf, 4.0]}
    )
    target = pd.Series([0, 0, 1, 1])

    with pytest.raises(
        DatasetValidationError,
        match="non-finite",
    ):
        validate_dataset(features, target)


def test_mismatched_row_counts_are_rejected():
    features = pd.DataFrame(
        {"feature_a": [1.0, 2.0, 3.0, 4.0]}
    )
    target = pd.Series([0, 1])

    with pytest.raises(
        DatasetValidationError,
        match="row counts do not match",
    ):
        validate_dataset(features, target)


def test_non_binary_target_is_rejected():
    features = pd.DataFrame(
        {"feature_a": [1.0, 2.0, 3.0, 4.0]}
    )
    target = pd.Series([0, 1, 2, 3])

    with pytest.raises(
        DatasetValidationError,
        match="exactly two target classes",
    ):
        validate_dataset(features, target)
