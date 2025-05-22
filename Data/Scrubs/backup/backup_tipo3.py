python
import pandas as pd

from compress import compress_numeric
from utils import time_my_func

@time_my_func
def respaldar_datos(dataframe, ruta_limpia):
    """
    Writes a dataframe to disk after
     cleaning dates
     imputing categoricals
     compressing numerics
    """
    print("Fixing dates...")
    columnas_fecha = dataframe.columns.map(lambda i: i if 'date' in i else None).dropna().tolist()
    if len(columnas_fecha) >= 1:
        for COL in columnas_fecha:
            dataframe.loc[:, COL] = pd.to_datetime(dataframe[COL])

    print("Fixing categoricals...")
    # categoricals
    columnas_cat = dataframe.select_dtypes(include=object).columns.tolist()
    if len(columnas_cat) >= 1:
        for COL in columnas_cat:
            dataframe.loc[:, COL] = dataframe[COL].fillna('_missing_')

    print("Fixing numerics...")
    dataframe = dataframe.apply(compress_numeric)

    print("Saving cleaned file to {}".format(ruta_limpia))
    dataframe.to_csv(ruta_limpia)
    return None