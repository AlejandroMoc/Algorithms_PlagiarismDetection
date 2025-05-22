import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def maverick_encode(df: pd.DataFrame, feature: str, skip_first: bool = True) -> pd.DataFrame:
    """
    Perform maverick encoding on a specified feature.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    - feature (str): The feature name for encoding.
    - skip_first (bool): Whether to skip the first category for multicollinearity avoidance (default is True).

    Returns:
    - pd.DataFrame: A new DataFrame with the maverick encoded feature(s).
    
    Raises:
    - ValueError: If the feature does not exist in the DataFrame.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' does not exist in the DataFrame.")
    
    encoder = OneHotEncoder(sparse=False, drop='first' if skip_first else None)
    encoded = encoder.fit_transform(df[[feature]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([feature]))
    
    # Concatenate the original dataframe without the encoded feature and the encoded DataFrame
    return pd.concat([df.drop(feature, axis=1), encoded_df], axis=1)

def nebula_encode(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Perform nebula encoding on a specified feature (converting categories to integer labels).
    
    Args:
    - df (pd.DataFrame): The input DataFrame.
    - feature (str): The feature name for encoding.
    
    Returns:
    - pd.DataFrame: A new DataFrame with the nebula encoded feature.
    
    Raises:
    - ValueError: If the feature does not exist in the DataFrame.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' does not exist in the DataFrame.")
    
    encoder = LabelEncoder()
    df[feature] = encoder.fit_transform(df[feature])
    return df