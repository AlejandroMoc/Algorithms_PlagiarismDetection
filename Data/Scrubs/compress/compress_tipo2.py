import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from .clip import clip_categorical

def shrink_numeric(COLUMN):
    """
    If the passed COLUMN is numeric,
    downcast it to the smallest size.
    Else,
    Return as-is.
    Parameters
    -----------
    COLUMN: pandas.Series
        The Series to shrink
    Returns
    -------
    if numeric, a shrunken series
    """
    if 'float' in str(COLUMN.dtype):
        print("Reducing {} to float".format(COLUMN.name))
        result = pd.to_numeric(COLUMN, downcast='float', errors='ignore')
    elif 'int' in str(COLUMN.dtype):
        print("Reducing {} to int".format(COLUMN.name))
        result = pd.to_numeric(COLUMN, downcast='integer', errors='ignore')
    else:
        print("{} is not numeric. Returning as-is".format(COLUMN.name))
        result = COLUMN
    return result

def encode_categorical(COLUMN):
    """
    Encode categorical variables with >2 classes
    Persist the encoder as JSON
    """
    if COLUMN.nunique() > 8:
        print("{} has too many levels, clipping it.")
        COLUMN_clipped = clip_categorical(COLUMN, MIN_LEVELS=8)
    else:
        COLUMN_clipped = COLUMN.copy()

    le = LabelEncoder()
    lookup = pd.Series(le.classes_).to_dict()
    COLUMN_encoded = pd.Series(le.transform(COLUMN_clipped), index=COLUMN.index, name=COLUMN.name)

    path_persist = "data/interim/{}_lookup.json".format(COLUMN.name)
    print("Persisting encoder at {}".format(path_persist))
    with open(path_persist, 'w') as fp:
        json.dump(lookup, fp)
    return COLUMN_encoded