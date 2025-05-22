import pandas as pd
import numpy as np

def lumo_transform(p_dataset: pd.DataFrame, p_feature: str) -> pd.DataFrame:
    """
    Apply a logarithmic transformation (log1p) to a specified column.
    The transformation is log(1 + x) to handle zero and positive values.

    Args:
    - p_dataset (pd.DataFrame): The input DataFrame.
    - p_feature (str): The column name to transform.

    Returns:
    - pd.DataFrame: A new DataFrame with the transformed column.

    Raises:
    - ValueError: If the column does not exist in the DataFrame or contains non-positive values.
    """
    if p_feature not in p_dataset.columns:
        raise ValueError(f"Column '{p_feature}' does not exist in the DataFrame.")
    
    # Ensure that the values are positive before applying log transformation
    if (p_dataset[p_feature] <= 0).any():
        raise ValueError(f"Log transformation cannot be applied to non-positive values in column '{p_feature}'.")

    p_dataset[p_feature] = np.log1p(p_dataset[p_feature])
    return p_dataset

def mega_transform(p_dataset: pd.DataFrame, p_feature: str, power_level: float = 2.0) -> pd.DataFrame:
    """
    Apply a power transformation to a specified column.

    Args:
    - p_dataset (pd.DataFrame): The input DataFrame.
    - p_feature (str): The column name to transform.
    - power_level (float): The power to raise the values to (default is 2.0).

    Returns:
    - pd.DataFrame: A new DataFrame with the transformed column.

    Raises:
    - ValueError: If the column does not exist in the DataFrame.
    """
    if p_feature not in p_dataset.columns:
        raise ValueError(f"Column '{p_feature}' does not exist in the DataFrame.")
    
    p_dataset[p_feature] = np.power(p_dataset[p_feature], power_level)
    return p_dataset

def binotron(p_dataset: pd.DataFrame, p_feature: str, threshold_value: float) -> pd.DataFrame:
    """
    Binarize a specified column based on a threshold. Values greater than the threshold
    are set to 1, and values less than or equal to the threshold are set to 0.

    Args:
    - p_dataset (pd.DataFrame): The input DataFrame.
    - p_feature (str): The column name to binarize.
    - threshold_value (float): The threshold for binarization.

    Returns:
    - pd.DataFrame: A new DataFrame with the binarized column.

    Raises:
    - ValueError: If the column does not exist in the DataFrame.
    """
    if p_feature not in p_dataset.columns:
        raise ValueError(f"Column '{p_feature}' does not exist in the DataFrame.")
    
    p_dataset[p_feature] = (p_dataset[p_feature] > threshold_value).astype(int)
    return p_dataset