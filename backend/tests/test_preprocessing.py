import numpy as np
import pandas as pd
import pytest

from svmforge.data import load_dataset
from svmforge.preprocessing import (
    fit_preprocessor,
    split_dataset,
)


def test_split_sizes_and_rows_are_preserved():
    features, target = load_dataset()

    splits = split_dataset(features, target)
    x_train, x_validation, x_test, y_train, y_validation, y_test = splits

    assert len(x_train) == 398
    assert len(x_validation) == 85
    assert len(x_test) == 86
    assert len(y_train) + len(y_validation) + len(y_test) == 569

    all_indices = list(x_train.index) + list(x_validation.index) + list(x_test.index)
    assert len(set(all_indices)) == 569


def test_splits_are_reproducible():
    features, target = load_dataset()

    first = split_dataset(features, target, random_state=42)
    second = split_dataset(features, target, random_state=42)

    for first_part, second_part in zip(first, second):
        if isinstance(first_part, pd.DataFrame):
            pd.testing.assert_frame_equal(first_part, second_part)
        else:
            pd.testing.assert_series_equal(first_part, second_part)


def test_class_proportions_are_approximately_preserved():
    features, target = load_dataset()

    x_train, x_validation, x_test, y_train, y_validation, y_test = split_dataset(
        features, target
    )

    original_ratio = target.mean()

    for split_target in (y_train, y_validation, y_test):
        assert abs(split_target.mean() - original_ratio) < 0.04

    assert len(x_train) + len(x_validation) + len(x_test) == len(features)


def test_scaler_is_fitted_on_training_data_only():
    features, target = load_dataset()

    x_train, x_validation, x_test, _, _, _ = split_dataset(
        features,
        target,
    )
    scaler = fit_preprocessor(x_train)

    np.testing.assert_allclose(
        scaler.mean_,
        x_train.mean().to_numpy(),
    )

    train_scaled = scaler.transform(x_train)
    validation_scaled = scaler.transform(x_validation)
    test_scaled = scaler.transform(x_test)

    assert train_scaled.shape == x_train.shape
    assert validation_scaled.shape == x_validation.shape
    assert test_scaled.shape == x_test.shape
    assert np.isfinite(train_scaled).all()
    assert np.isfinite(validation_scaled).all()
    assert np.isfinite(test_scaled).all()


def test_empty_training_data_is_rejected():
    empty_features = pd.DataFrame(columns=["feature_a"])

    with pytest.raises(ValueError, match="must not be empty"):
        fit_preprocessor(empty_features)


def test_mismatched_feature_and_target_rows_are_rejected():
    features = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0, 4.0]})
    target = pd.Series([0, 1])

    with pytest.raises(ValueError, match="row counts do not match"):
        split_dataset(features, target)


def test_train_validation_and_test_indices_are_disjoint():
    features, target = load_dataset()

    (
        x_train,
        x_validation,
        x_test,
        _,
        _,
        _,
    ) = split_dataset(features, target)

    train_indices = set(x_train.index)
    validation_indices = set(x_validation.index)
    test_indices = set(x_test.index)

    assert train_indices.isdisjoint(validation_indices)
    assert train_indices.isdisjoint(test_indices)
    assert validation_indices.isdisjoint(test_indices)


def test_every_split_contains_both_target_classes():
    features, target = load_dataset()

    (
        _,
        _,
        _,
        y_train,
        y_validation,
        y_test,
    ) = split_dataset(features, target)

    for split_target in (y_train, y_validation, y_test):
        assert set(split_target.unique()) == set(target.unique())
