import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def encode_one_hot(df_input: pd.DataFrame, col_name: str, drop_first: bool = True) -> pd.DataFrame:
    if col_name not in df_input.columns:
        raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")
    
    encoder = OneHotEncoder(sparse=False, drop='first' if drop_first else None)
    encoded_data = encoder.fit_transform(df_input[[col_name]])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col_name]))
    
    return pd.concat([df_input.drop(col_name, axis=1), encoded_df], axis=1)

def encode_label(df_input: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if col_name not in df_input.columns:
        raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")
    
    encoder = LabelEncoder()
    df_input[col_name] = encoder.fit_transform(df_input[col_name])
    return df_input