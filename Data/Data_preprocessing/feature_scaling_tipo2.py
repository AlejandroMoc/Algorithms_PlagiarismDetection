from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

def transform_data(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Normalize and scale the data for specified columns.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    - cols (list): List of column names to transform.

    Returns:
    - pd.DataFrame: A new DataFrame with transformed features.

    Raises:
    - ValueError: If any of the columns do not exist in the DataFrame.
    """
    # Check if columns exist in the DataFrame
    missing_columns = [col for col in cols if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} do not exist in the DataFrame.")
    
    # Apply StandardScaler
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

def scale_data_range(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Apply range scaling (scale to a [0, 1] range) for specified columns.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    - cols (list): List of column names to scale.

    Returns:
    - pd.DataFrame: A new DataFrame with range-scaled features.

    Raises:
    - ValueError: If any of the columns do not exist in the DataFrame.
    """
    # Check if columns exist in the DataFrame
    missing_columns = [col for col in cols if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} do not exist in the DataFrame.")
    
    # Apply MinMaxScaler
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df