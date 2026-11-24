
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained regression model on test data.

    Args:
        model: A fitted regression model.
        X_test: Testing input features.
        y_test: Actual testing target values.

    Returns:
        dict: MAE, MSE, RMSE, and R2 evaluation metrics.
    """

    y_pred = model.predict(X_test)

    results = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }

    return results


def compare_models(model_results):
    """
    Create a sorted comparison table from model evaluation results.

    Args:
        model_results: Dictionary containing model names and
                       their evaluation metrics.

    Returns:
        pandas.DataFrame: Models sorted by ascending RMSE.
    """

    results_df = pd.DataFrame.from_dict(
        model_results,
        orient="index"
    )

    results_df.index.name = "Model"

    return results_df.sort_values(by="RMSE")
