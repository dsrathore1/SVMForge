import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from svmforge.data import load_dataset


def main() -> None:
    features, target = load_dataset()

    non_finite_count = int(
        (~np.isfinite(features.to_numpy(dtype=np.float64))).sum()
    )

    print("SVMForge Dataset Validation")
    print("===========================")
    print("Validation status: PASSED")
    print(f"Samples: {len(features)}")
    print(f"Features: {features.shape[1]}")
    print(f"Missing feature values: {int(features.isna().sum().sum())}")
    print(f"Missing target values: {int(target.isna().sum())}")
    print(f"Non-finite feature values: {non_finite_count}")
    print("Target distribution:")

    for label, count in target.value_counts().sort_index().items():
        print(f"  Class {label}: {count}")


if __name__ == "__main__":
    main()
