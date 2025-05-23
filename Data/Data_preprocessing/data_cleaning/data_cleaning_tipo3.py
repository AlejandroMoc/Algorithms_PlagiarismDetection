python
import pandas as pd

def eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        print("Advertencia: El DataFrame está vacío.")
    return df.drop_duplicates()

def reemplazar_faltantes_con_media(df: pd.DataFrame, columna: str, valor_predeterminado: float = None) -> pd.DataFrame:
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")
    
    if valor_predeterminado is not None:
        df[columna].fillna(valor_predeterminado, inplace=True)
    else:
        valor_medio = df[columna].mean()
        df[columna].fillna(valor_medio, inplace=True)
    
    return df

def estandarizar_texto(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")
    
    df[columna] = df[columna].str.lower().str.strip()
    return df