import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from .clip import clip_categorical

def reduce_numeric(col_data):
    if 'float' in str(col_data.dtype):
        print("Downcasting {} to float".format(col_data.name))
        result_data = pd.to_numeric(col_data, downcast='float', errors='ignore')
    elif 'int' in str(col_data.dtype):
        print("Downcasting {} to int".format(col_data.name))
        result_data = pd.to_numeric(col_data, downcast='integer', errors='ignore')
    else:
        print("{} is not numeric. Returning as-is".format(col_data.name))
        result_data = col_data
    return result_data

def reduce_categorical(col_data):
    if col_data.nunique() > 8:
        print("{} has too many levels, clipping it.")
        col_data_clipped = clip_categorical(col_data, MIN_LEVELS=8)
    else:
        col_data_clipped = col_data.copy()

    label_encoder = LabelEncoder()
    lookup_dict = pd.Series(label_encoder.classes_).to_dict()
    col_encoded = pd.Series(label_encoder.transform(col_data_clipped), index=col_data.index, name=col_data.name)

    persist_path = "data/interim/{}_lookup.json".format(col_data.name)
    print("Persisting encoder at {}".format(persist_path))
    with open(persist_path, 'w') as file_pointer:
        json.dump(lookup_dict, file_pointer)
    return col_encoded