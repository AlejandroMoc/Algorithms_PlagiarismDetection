import pandas as pd
from sklearn.impute import SimpleImputer

def fill_numbers_with_average(tbl: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Fill missing numeric values in specified columns with the average value of each column.

    Args:
    - tbl (pd.DataFrame): The input DataFrame.
    - cols (list): List of column names to fill.

    Returns:
    - pd.DataFrame: A new DataFrame with missing numeric values filled with the average.

    Raises:
    - ValueError: If any of the columns do not exist in the DataFrame.
    """
    missing_columns = [col for col in cols if col not in tbl.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} do not exist in the DataFrame.")
    
    filler = SimpleImputer(strategy='mean')
    tbl[cols] = filler.fit_transform(tbl[cols])
    return tbl

def fill_categorical_with_mode(tbl: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Fill missing categorical values in specified columns with the most frequent value (mode).

    Args:
    - tbl (pd.DataFrame): The input DataFrame.
    - cols (list): List of column names to fill.

    Returns:
    - pd.DataFrame: A new DataFrame with missing categorical values filled with the mode.

    Raises:
    - ValueError: If any of the columns do not exist in the DataFrame.
    """
    missing_columns = [col for col in cols if col not in tbl.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} do not exist in the DataFrame.")
    
    filler = SimpleImputer(strategy='most_frequent')
    tbl[cols] = filler.fit_transform(tbl[cols])
    return tbl

def remove_missing_values(tbl: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Remove rows with missing values if the number of missing values exceeds a given threshold.

    Args:
    - tbl (pd.DataFrame): The input DataFrame.
    - threshold (float): Proportion of non-null values required in a row to keep it. Default is 0.5.

    Returns:
    - pd.DataFrame: A new DataFrame with rows removed based on the missing value threshold.
    """
    # Drop rows where the number of non-null values is less than the threshold
    return tbl.dropna(thresh=int(tbl.shape[1] * threshold))