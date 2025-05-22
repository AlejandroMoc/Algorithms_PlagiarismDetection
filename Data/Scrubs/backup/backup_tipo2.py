import pandas as pd

from squeeze import squeeze_numeric
from tools import measure_execution_time

@measure_execution_time
def store_data_frame(data_frame, storage_path):
    """
    Writes a dataframe to disk after
     cleaning dates
     imputing categoricals
     compressing numerics
    """
    print("Fixing dates...")
    date_columns = data_frame.columns.map(lambda i: i if 'date' in i else None).dropna().tolist()
    if len(date_columns) >= 1:
        for COL in date_columns:
            data_frame.loc[:, COL] = pd.to_datetime(data_frame[COL])

    print("Fixing categoricals...")
    # categoricals
    categorical_columns = data_frame.select_dtypes(include=object).columns.tolist()
    if len(categorical_columns) >= 1:
        for COL in categorical_columns:
            data_frame.loc[:, COL] = data_frame[COL].fillna('_missing_')

    print("Fixing numerics...")
    data_frame = data_frame.apply(squeeze_numeric)

    print("Saving cleaned file to {}".format(storage_path))
    data_frame.to_csv(storage_path)
    return None