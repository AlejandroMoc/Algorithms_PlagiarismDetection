import pandas as pd
from utils import time_my_func

def generate_encodings(data, OMIT_ONE=True):
    """
    Produce encodings for various tiers of a trimmed grouping
    Omit one to sidestep the pitfall
    Parameters
    ----------
    data: input grouping data
        pandas.Series
    Returns
    -------
    encoded_df: encoded variables with one tier omitted
        pandas.DataFrame
    """
    if data.nunique() > 10:
        print("Grouping has too many tiers, consider trimming")
        encoded_df = None
    else:
        PREFIX = 'signal_' + data.name + '_'
        encoded_df = pd.get_dummies(data, prefix=PREFIX)
        if OMIT_ONE:
            other_col = [c for c in encoded_df if 'Other' in c]
            to_omit_ = other_col if other_col else encoded_df.mean().idxmin()
            print("Omitting {}".format(to_omit_))
            encoded_df.drop(to_omit_, axis=1, inplace=True)
    return encoded_df

@time_my_func
def form_encoded_df(frame, omit_one=True):
    """
    For every (trimmed) grouping column
    * Form encoding DataFrame
    * Concatenate to input frame
    * Omit the grouping

    Returns
    -------
        Passed frame with signal_* columns substituting groupings
    """
    frame_ = frame.copy()

    cols_encoded = \
    (frame_
    .select_dtypes(include=object)
    .columns
    .tolist())
    print("Creating encodings for \n{}".format(cols_encoded))

    list_encoded_df = \
    [generate_encodings(frame_[COL], OMIT_ONE=omit_one) for COL in cols_encoded]

    frame_2 = \
    pd.concat([
        frame_.drop(cols_encoded, axis=1),
        pd.concat(list_encoded_df, axis=1)
    ], axis=1)

    return frame_2