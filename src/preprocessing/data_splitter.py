from sklearn.model_selection import train_test_split
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and target into training and testing sets.

    Args:
        X: Input features.
        y: Target variable.
        test_size: Proportion of data reserved for testing.
        random_state: Controls reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test