import numpy as np

from utilities import time_execution

def eliminate_unnamed(data_frame):
    """
    """
    unnamed_columns = [x for x in data_frame.columns if x.lower().startswith('unnamed')]

    if len(unnamed_columns) >= 1:
        data_frame.drop(unnamed_columns, axis=1, inplace=True)
    else:
        pass
    return data_frame

def eliminate_zv(df_):
    """
    Drop columns that have zero-variance
    For Categoricals, if nunique == 1
    For Numeric, if std == 0
    """
    catg_zv_columns = \
    (df_
     .select_dtypes(include='object')
     .nunique()
     .where(lambda i: i == 1)
     .dropna()
     .index
     .tolist()
    )

    numeric_zv_columns = \
    (df_
     .select_dtypes(include=np.number)
     .std()
     .where(lambda i: i == 0)
     .dropna()
     .index
     .tolist()
    )

    zv_columns = catg_zv_columns + numeric_zv_columns

    if len(zv_columns) >= 1:
        print("The following columns have zero-variance and will be dropped \n{}".format(zv_columns))
        df_.drop(zv_columns, axis=1, inplace=True)
    else:
        print("No columns with zero-variance.")
    return df_

def eliminate_nzv(df_, nzv_threshold=0.95):
    """
    Drop categorical columns that have near-zero variance
    Such variables have very little predictive power
    i.e., if frequency of mode > threshold
    """
    catg_nzv_columns = \
    (df_
     .select_dtypes(include='object')
     .apply(lambda c: c.value_counts(normalize=True).agg(['max', 'idxmax']))
     .T
     .query("max > {}".format(nzv_threshold))
     .index
     .tolist()
    )

    if len(catg_nzv_columns) >= 1:
        print("The mode of these columns has a frequency higher than {}. Dropping these. \n{}"
              .format(nzv_threshold, catg_nzv_columns))
        df_.drop(catg_nzv_columns, axis=1, inplace=True)
    else:
        print("No categorical columns with near-zero variance found.")
    return df_

def eliminate_missing_values(df_, NA_threshold=0.8):
    """
    Drop columns that have more missings than threshold
    """
    missing_columns = \
    (df_
     .isnull()
     .mean()
     .where(lambda i: i > NA_threshold)
     .dropna()
     .index
     .tolist()
    )

    if len(missing_columns) >= 1:
        print("The following columns have more than {:.2f}% missings and will be dropped...\n{}"
              .format(NA_threshold * 100, missing_columns))
        df_.drop(missing_columns, inplace=True, axis=1)
    else:
        print("No columns have more than {:.2f}% missings.".format(NA_threshold))
    return df_

@time_execution
def clean_data(df, NA_threshold=0.85, nzv_threshold=0.95):
    """
    Clean passed dataset by removing columns with
    * gt NA_threshold percentage of missing values
    * gt nzv_threshold frequency of the mode
    * zero variance

    Parameters
    ---------
    df: DataFrame
        The input dataset

    NA_threshold: float
        Acceptable limit for missings

    nzv_threshold: float
        Acceptable limit for frequency of mode

    Returns
    -------
        Cleaned DataFrame
    """
    df_ = \
    (df
     .copy()
     .pipe(eliminate_unnamed)
     .pipe(eliminate_missing_values, NA_threshold=NA_threshold)
     .pipe(eliminate_zv)
     .pipe(eliminate_nzv, nzv_threshold=nzv_threshold)
    )
    return df_