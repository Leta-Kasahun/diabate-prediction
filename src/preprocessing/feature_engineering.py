
import pandas as pd
def create_features(df):
    """
    Create additional features for the diabetes prediction dataset.

    Features created:
    1. BMI (Body Mass Index):
       Calculated using weight in pounds and height in inches.
       Formula: (weight * 703) / (height ** 2).

    2. WHR (Waist-to-Hip Ratio):
       Calculated by dividing waist circumference by hip circumference.
       Both measurements are assumed to be in the same unit (inches).

    Notes:
    - Creates a copy to preserve the original DataFrame.
    - Missing input values naturally produce missing derived values.
    - Original measurement columns are retained.
    - Measurement units should be verified against the dataset documentation.

    Args:
        df (pd.DataFrame): Cleaned diabetes dataset.

    Returns:
        pd.DataFrame: Dataset with additional engineered features.
    """
    data = df.copy()

    # Calculate Body Mass Index (BMI)
    data["bmi"] = (
        data["weight"] * 703
    ) / (data["height"] ** 2)

    # Calculate Waist-to-Hip Ratio (WHR)
    data["waist_hip_ratio"] = (
        data["waist"] / data["hip"]
    )

    return data
