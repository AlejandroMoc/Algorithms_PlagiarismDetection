import numpy as np

from utils import time_my_func

def eliminar_sin_nombre(df):
    """
    """
    cols_sin_nombre = [x for x in df.columns if x.lower().startswith('unnamed')]

    if len(cols_sin_nombre) >= 1:
        df.drop(cols_sin_nombre, axis=1, inplace=True)
    else:
        pass
    return df

def eliminar_zv(df_):
    """
    Eliminar columnas con varianza cero
    Para Categóricos, si nunique == 1
    Para Numéricos, si std == 0
    """
    cols_catg_zv = \
    (df_
     .select_dtypes(include='object')
     .nunique()
     .where(lambda i: i == 1)
     .dropna()
     .index
     .tolist()
    )

    cols_numeric_zv = \
    (df_
     .select_dtypes(include=np.number)
     .std()
     .where(lambda i: i == 0)
     .dropna()
     .index
     .tolist()
    )

    cols_zv = cols_catg_zv + cols_numeric_zv

    if len(cols_zv) >= 1:
        print("Las siguientes columnas tienen varianza cero y serán eliminadas \n{}".format(cols_zv))
        df_.drop(cols_zv, axis=1, inplace=True)
    else:
        print("No hay columnas con varianza cero.")
    return df_

def eliminar_nzv(df_, nzv_limite=0.95):
    """
    Eliminar columnas categóricas con varianza cercana a cero
    Estas variables tienen muy poco poder predictivo
    es decir, si la frecuencia del modo > límite
    """
    cols_catg_nzv = \
    (df_
     .select_dtypes(include='object')
     .apply(lambda c: c.value_counts(normalize=True).agg(['max', 'idxmax']))
     .T
     .query("max > {}".format(nzv_limite))
     .index
     .tolist()
    )

    if len(cols_catg_nzv) >= 1:
        print("El modo de estas columnas tiene una frecuencia mayor que {}. Eliminando estas. \n{}"
              .format(nzv_limite, cols_catg_nzv))
        df_.drop(cols_catg_nzv, axis=1, inplace=True)
    else:
        print("No se encontraron columnas categóricas con varianza cercana a cero.")
    return df_

def eliminar_faltantes(df_, umbral_faltantes=0.8):
    """
    Eliminar columnas con más faltantes que el umbral
    """
    cols_faltantes = \
    (df_
     .isnull()
     .mean()
     .where(lambda i: i > umbral_faltantes)
     .dropna()
     .index
     .tolist()
    )

    if len(cols_faltantes) >= 1:
        print("Las siguientes columnas tienen más del {:.2f}% de faltantes y serán eliminadas...\n{}"
              .format(umbral_faltantes * 100, cols_faltantes))
        df_.drop(cols_faltantes, inplace=True, axis=1)
    else:
        print("No hay columnas con más del {:.2f}% de faltantes.".format(umbral_faltantes))
    return df_

@time_my_func
def quitar_zv_faltantes(df, umbral_faltantes=0.85, nzv_limite=0.95):
    """
    Limpiar conjunto de datos pasado eliminando columnas con
    * más del umbral_faltantes porcentaje de valores faltantes
    * más del nzv_limite frecuencia del modo
    * varianza cero

    Parámetros
    ---------
    df: DataFrame
        El conjunto de datos de entrada

    umbral_faltantes: float
        Límite aceptable para faltantes

    nzv_limite: float
        Límite aceptable para frecuencia del modo

    Devuelve
    -------
        DataFrame limpio
    """
    df_ = \
    (df
     .copy()
     .pipe(eliminar_sin_nombre)
     .pipe(eliminar_faltantes, umbral_faltantes=umbral_faltantes)
     .pipe(eliminar_zv)
     .pipe(eliminar_nzv, nzv_limite=nzv_limite)
    )
    return df_