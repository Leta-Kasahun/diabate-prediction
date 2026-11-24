
import pandas as pd


def clean_data(df):
    """
    Perform basic cleaning on the diabetes dataset.

    Cleaning operations:
    1. Create a copy of the original DataFrame to avoid modifying
       the raw dataset.
    2. Remove 'bp.2s' and 'bp.2d' because both columns have
       approximately 65% missing values. Keeping them at this
       stage could reduce the reliability of these features.
    3. Remove leading and trailing whitespace from categorical
       columns ('location', 'gender', and 'frame') to ensure
       consistent category values during encoding.

    Notes:
    - The original raw dataset remains unchanged.
    - Other missing values are not handled here; they will be
      handled later in the preprocessing pipeline.
    - The 'id' column is retained for record identification,
      but it should not be used as a model feature.

    Args:
        df (pd.DataFrame): Original diabetes dataset.

    Returns:
        pd.DataFrame: Basic cleaned copy of the diabetes dataset.
    """

    data = df.copy()

    # Remove columns with excessive missing values (approximately 65%)
    data = data.drop(columns=["bp.2s", "bp.2d"])

    # Remove unnecessary whitespace from categorical values
    categorical_columns = ["location", "gender", "frame"]
    for col in categorical_columns:
        data[col] = data[col].str.strip()

    return data
