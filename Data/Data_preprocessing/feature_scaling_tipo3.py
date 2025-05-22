from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

def estandarizar_caracteristicas(datos: pd.DataFrame, columnas: list) -> pd.DataFrame:
    faltantes = [col for col in columnas if col not in datos.columns]
    if faltantes:
        raise ValueError(f"Las columnas {faltantes} no existen en el DataFrame.")
    
    escalador = StandardScaler()
    datos[columnas] = escalador.fit_transform(datos[columnas])
    return datos

def escalar_min_max_caracteristicas(datos: pd.DataFrame, columnas: list) -> pd.DataFrame:
    faltantes = [col for col in columnas if col not in datos.columns]
    if faltantes:
        raise ValueError(f"Las columnas {faltantes} no existen en el DataFrame.")
    
    escalador = MinMaxScaler()
    datos[columnas] = escalador.fit_transform(datos[columnas])
    return datos