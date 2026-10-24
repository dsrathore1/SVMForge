import numpy as np
from sklearn.pipeline import Pipeline

from svmforge.data import load_dataset
from svmforge.model import create_linear_svm
from svmforge.preprocessing import split_dataset


def test_linear_svm_pipeline_trains_and_predicts():
    features, target = load_dataset()

    (
        x_train,
        x_validation,
        _x_test,
        y_train,
        _y_validation,
        _y_test,
    ) = split_dataset(features, target)

    model = create_linear_svm()
    model.fit(x_train, y_train)

    predictions = model.predict(x_validation)

    assert isinstance(model, Pipeline)
    assert len(predictions) == len(x_validation)
    assert set(np.unique(predictions)).issubset(set(target.unique()))
    assert np.isfinite(predictions).all()


def test_linear_svm_uses_expected_configuration():
    model = create_linear_svm()

    assert model.named_steps["scaler"].__class__.__name__ == "StandardScaler"
    assert model.named_steps["classifier"].kernel == "linear"
    assert model.named_steps["classifier"].C == 1.0
