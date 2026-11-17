from pathlib import Path
import pandas as pd

PROJECT_ROOT=Path(__file__).resolve().parents[2]
DATA_PATH=PROJECT_ROOT/"data"/"raw"/"diabetes.csv"

def load_data():
    """
    Load the diabetes dataset from the specified path.

    Returns:
        pd.DataFrame: A DataFrame containing the diabetes dataset.
    """
    try:
        df= pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        print(f"File not found at {DATA_PATH}. Please ensure the file exists.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None