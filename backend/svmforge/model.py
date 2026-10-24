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

def create_rbf_svm() -> Pipeline:
    """Create an RBF-kernel SVM classification pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(kernel="rbf", C=1.0, gamma="scale"),
            ),
        ]
    )