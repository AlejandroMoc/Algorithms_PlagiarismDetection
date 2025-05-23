import pandas as pd
from sklearn.impute import SimpleImputer

def completar_numericos_con_media(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    columnas_faltantes = [col for col in columnas if col not in df.columns]
    if columnas_faltantes:
        raise ValueError(f"Las columnas {columnas_faltantes} no existen en el DataFrame.")
    
    imputador = SimpleImputer(strategy='mean')
    df[columnas] = imputador.fit_transform(df[columnas])
    return df

def completar_categoricos_con_moda(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    columnas_faltantes = [col for col in columnas if col not in df.columns]
    if columnas_faltantes:
        raise ValueError(f"Las columnas {columnas_faltantes} no existen en el DataFrame.")
    
    imputador = SimpleImputer(strategy='most_frequent')
    df[columnas] = imputador.fit_transform(df[columnas])
    return df

def eliminar_faltantes(df: pd.DataFrame, umbral: float = 0.5) -> pd.DataFrame:
    return df.dropna(thresh=int(df.shape[1] * umbral))