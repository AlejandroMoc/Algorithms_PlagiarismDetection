import pandas as pd
import numpy as np

def eliminar_atipicos_iqr(datos: pd.DataFrame, columna: str) -> pd.DataFrame:
    if columna not in datos.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")
    
    Q1 = datos[columna].quantile(0.25)
    Q3 = datos[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    return datos[(datos[columna] >= limite_inferior) & (datos[columna] <= limite_superior)]

def eliminar_atipicos_zscore(datos: pd.DataFrame, columna: str, umbral: float = 3.0) -> pd.DataFrame:
    if columna not in datos.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")
    
    media = datos[columna].mean()
    desviacion_estandar = datos[columna].std()
    z_scores = (datos[columna] - media) / desviacion_estandar
    
    return datos[np.abs(z_scores) <= umbral]