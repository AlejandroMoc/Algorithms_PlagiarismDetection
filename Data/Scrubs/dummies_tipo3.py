import pandas as pd
from utils import time_my_func

def generar_variables_ficticias(serie, ELIMINAR_UNA=True):
    if serie.nunique() > 10:
        print("La variable categórica tiene demasiados niveles, considere cortarla")
        df_fic = None
    else:
        PREFIJO = 'flag_' + serie.name + '_'
        df_fic = pd.get_dummies(serie, prefix=PREFIJO)
        if ELIMINAR_UNA:
            otra_columna = [c for c in df_fic if 'Otro' in c]
            a_eliminar = otra_columna if otra_columna else df_fic.mean().idxmin()
            print("Eliminando {}".format(a_eliminar))
            df_fic.drop(a_eliminar, axis=1, inplace=True)
    return df_fic

@time_my_func
def crear_dataframe_ficticio(df, eliminar_una=True):
    df_ = df.copy()

    columnas_ficticias = \
    (df_
    .select_dtypes(include=object)
    .columns
    .tolist())
    print("Creando variables ficticias para \n{}".format(columnas_ficticias))

    lista_df_ficticios = \
    [generar_variables_ficticias(df_[COL], ELIMINAR_UNA=eliminar_una) for COL in columnas_ficticias]

    df_2 = \
    pd.concat([
        df_.drop(columnas_ficticias, axis=1),
        pd.concat(lista_df_ficticios, axis=1)
    ], axis=1)

    return df_2