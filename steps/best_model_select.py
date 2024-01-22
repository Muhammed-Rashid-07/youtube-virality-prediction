
import pandas as pd
from typing import Tuple
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated

from zenml import step


@step
def best_model_selector(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model1: ClassifierMixin,
    model2: ClassifierMixin,
    model3: ClassifierMixin,
) -> Tuple[
    Annotated[ClassifierMixin, "best_model"],
    Annotated[float, "best_model_test_acc"],
]:
    """
    Selects the best model based on test accuracy.

    Args:
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        model1 (ClassifierMixin): First model for evaluation.
        model2 (ClassifierMixin): Second model for evaluation.
        model3 (ClassifierMixin): Third model for evaluation.

    Returns:
        Tuple[ClassifierMixin, float]: Best model and its test accuracy.
    """
    # Calculate test accuracy for each model
    test_acc1 = model1.score(X_test.to_numpy(), y_test.to_numpy())
    test_acc2 = model2.score(X_test.to_numpy(), y_test.to_numpy())
    test_acc3 = model3.score(X_test.to_numpy(), y_test.to_numpy())

    # Print test accuracy for each model
    print(f"Test accuracy ({model1.__class__.__name__}): {test_acc1}")
    print(f"Test accuracy ({model2.__class__.__name__}): {test_acc2}")
    print(f"Test accuracy ({model3.__class__.__name__}): {test_acc3}")

    # Determine the best model based on test accuracy
    if test_acc1 > test_acc2 > test_acc3:
        best_model = model1
        best_model_test_acc = test_acc1
    elif test_acc2 > test_acc3:
        best_model = model2
        best_model_test_acc = test_acc2
    else:
        best_model = model3
        best_model_test_acc = test_acc3

    return best_model, best_model_test_acc
