
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(X):
    """
    Build a reusable preprocessing pipeline for machine learning.

    - Numerical features: fill missing values using the median
      and standardize values using StandardScaler.
    - Categorical features: fill missing values using the
      most frequent category and encode them using OneHotEncoder.
    - ColumnTransformer applies the appropriate pipeline to
      each feature type.

    The preprocessing parameters are learned from training data
    when fit() is called and reused when transform() is applied
    to testing or new data.
    """

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return preprocessor
