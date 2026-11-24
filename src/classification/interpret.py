import pandas as pd
from sklearn.inspection import permutation_importance


def interpret_model(
    model,
    preprocessor,
    X_test,
    y_test
):
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    feature_names = preprocessor.get_feature_names_out()

    results = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance.importances_mean,
        "Std": importance.importances_std
    })

    return results.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)