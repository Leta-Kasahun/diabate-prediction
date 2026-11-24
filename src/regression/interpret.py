
import pandas as pd

from sklearn.inspection import permutation_importance


def get_feature_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
):
    """
    Calculate and rank feature importance using permutation importance.

    A feature is considered important when randomly shuffling its
    values causes the model's prediction performance to decrease.

    Args:
        model: A fitted regression model.
        X_test: Testing input features as a DataFrame.
        y_test: Actual testing target values.
        n_repeats: Number of times each feature is shuffled.
        random_state: Random seed for reproducibility.

    Returns:
        pandas.DataFrame: Features ranked by descending importance.
    """

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=random_state
    )

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": importance.importances_mean,
        "Std": importance.importances_std
    })

    return importance_df.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)
