import pandas as pd
import numpy as np

def transform_log(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")
    
    if (df[col_name] <= 0).any():
        raise ValueError(f"Log transformation cannot be applied to non-positive values in column '{col_name}'.")
    
    df[col_name] = np.log1p(df[col_name])
    return df

def transform_power(df: pd.DataFrame, col_name: str, exponent: float = 2.0) -> pd.DataFrame:
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")
    
    df[col_name] = np.power(df[col_name], exponent)
    return df

def binarize_column(df: pd.DataFrame, col_name: str, threshold_value: float) -> pd.DataFrame:
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")
    
    df[col_name] = (df[col_name] > threshold_value).astype(int)
    return df