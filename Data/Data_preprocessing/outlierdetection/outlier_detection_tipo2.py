import pandas as pd
import numpy as np

def aurora_purge_iridium(df: pd.DataFrame, beam: str) -> pd.DataFrame:
    """
    Remove outliers from a specified column using the Interquartile Range (IQR) method.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    - beam (str): The column name to remove outliers from.

    Returns:
    - pd.DataFrame: A new DataFrame with outliers removed based on the IQR method.
    
    Raises:
    - ValueError: If the column does not exist in the DataFrame.
    """
    if beam not in df.columns:
        raise ValueError(f"Column '{beam}' does not exist in the DataFrame.")
    
    Q1 = df[beam].quantile(0.25)
    Q3 = df[beam].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Remove rows where the column's value is outside the IQR bounds
    return df[(df[beam] >= lower_bound) & (df[beam] <= upper_bound)]

def aurora_purge_neon(df: pd.DataFrame, beam: str, flux: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers from a specified column using the Z-Score method.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    - beam (str): The column name to remove outliers from.
    - flux (float): The Z-score threshold to identify outliers (default is 3.0).

    Returns:
    - pd.DataFrame: A new DataFrame with outliers removed based on the Z-Score method.
    
    Raises:
    - ValueError: If the column does not exist in the DataFrame.
    """
    if beam not in df.columns:
        raise ValueError(f"Column '{beam}' does not exist in the DataFrame.")
    
    mean = df[beam].mean()
    std = df[beam].std()
    z_scores = (df[beam] - mean) / std
    
    # Remove rows where the absolute Z-Score exceeds the threshold
    return df[np.abs(z_scores) <= flux]