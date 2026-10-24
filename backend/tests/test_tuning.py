from sklearn.model_selection import GridSearchCV, StratifiedKFold

from svmforge.model import create_svm_search


def test_svm_search_uses_stratified_five_fold_cross_validation():
    search = create_svm_search()

    assert isinstance(search, GridSearchCV)
    assert isinstance(search.cv, StratifiedKFold)
    assert search.cv.n_splits == 5
    assert search.cv.shuffle is True
    assert search.scoring == "f1"
    assert search.refit is True


def test_svm_search_covers_linear_and_rbf_kernels():
    search = create_svm_search()

    kernels = {
        grid["classifier__kernel"][0]
        for grid in search.param_grid
    }

    assert kernels == {"linear", "rbf"}
