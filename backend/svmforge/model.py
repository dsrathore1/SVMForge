from sklearn.model_selection import GridSearchCV, StratifiedKFold
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


def create_svm_search() -> GridSearchCV:
    """Create a leakage-safe, stratified five-fold SVM search."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", SVC()),
        ]
    )

    parameter_grid = [
        {
            "classifier__kernel": ["linear"],
            "classifier__C": [0.1, 1.0, 10.0],
        },
        {
            "classifier__kernel": ["rbf"],
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__gamma": ["scale", "auto"],
        },
    ]

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
    )
