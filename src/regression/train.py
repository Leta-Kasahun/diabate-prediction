
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


def get_regression_models():
    """
    Create and return all candidate regression models.

    Returns:
        dict: Model names mapped to untrained regression estimators.
    """

    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "ElasticNet": ElasticNet(),
        "KNN": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ),
        "CatBoost": CatBoostRegressor(
            random_seed=42,
            verbose=0
        ),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "SVR": SVR()
    }

    return models


def train_model(model, X_train, y_train):
    """
    Train a regression model using the training dataset.

    Args:
        model: An untrained scikit-learn-compatible regressor.
        X_train: Training input features.
        y_train: Training target values.

    Returns:
        The fitted regression model.
    """

    model.fit(X_train, y_train)

    return model
