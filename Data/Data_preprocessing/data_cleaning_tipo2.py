import pandas as pd

def eliminate_repeats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eliminates duplicate rows from the DataFrame.
    
    Args:
    - df (pd.DataFrame): The input DataFrame.
    
    Returns:
    - pd.DataFrame: A new DataFrame with duplicates removed.
    """
    if df.empty:
        print("Warning: The DataFrame is empty.")
    return df.drop_duplicates()

def fill_na_with_avg(df: pd.DataFrame, col: str, default_val: float = None) -> pd.DataFrame:
    """
    Fill missing values in a specified column with the column's mean or a provided default value.
    
    Args:
    - df (pd.DataFrame): The input DataFrame.
    - col (str): The column name where missing values need to be replaced.
    - default_val (float, optional): If provided, will replace missing values with this value.
    
    Returns:
    - pd.DataFrame: A new DataFrame with missing values replaced.
    
    Raises:
    - ValueError: If the column does not exist in the DataFrame.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
    
    if default_val is not None:
        df[col].fillna(default_val, inplace=True)
    else:
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
    
    return df

def normalize_strings(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Normalize the text in a specified column by converting it to lowercase and stripping whitespace.
    
    Args:
    - df (pd.DataFrame): The input DataFrame.
    - col (str): The column to normalize.
    
    Returns:
    - pd.DataFrame: A new DataFrame with normalized text in the specified column.
    
    Raises:
    - ValueError: If the column does not exist in the DataFrame.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
    
    df[col] = df[col].str.lower().str.strip()
    return df