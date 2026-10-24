from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def create_linear_svm() -> Pipeline:
    """Create a leakage-safe Linear SVM classification pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="linear", C=1.0)),
        ]
    )
