def manage_groups(s, THRESHOLD_1=5, THRESHOLD_2=0.05, PERCENTAGE=0.95):
    """
    Handle Groupings with multiple categories
    If there are only 2 categories, they will remain unchanged
    Parameters
    ----------
    S: pandas.Series
        the input Grouping series with >= THRESHOLD_1 levels
    THRESHOLD_2: float
        Levels with at least THRESHOLD_2 %cases will be preserved
    PERCENTAGE: float
        Levels that make up PERCENTAGE% of the data will be preserved
    Returns
    -------
    A pandas.Series object with
    kept labels for levels that represent PERCENTAGE% of the data
    replaced labels (with 'Other') for infrequent levels
    """
    s_copy = s.copy()
    if s_copy.nunique() >= THRESHOLD_1:
        PRESERVE_1 = \
        (s_copy
         .value_counts(normalize=True)
         .where(lambda i: i >= THRESHOLD_2)
         .dropna()
         .index
         .tolist()
        )

        PRESERVE_2 = \
        (s_copy
         .value_counts(normalize=True)
         .cumsum()
         .where(lambda x: x <= PERCENTAGE)
         .dropna()
         .index
         .tolist()
        )

        PRESERVE = set(PRESERVE_1).union(set(PRESERVE_2))

        s_copy[-s_copy.isin(PRESERVE)] = 'Other'
        s_copy = s_copy.map(lambda x: '_'.join(str(x).split()))
        print("{} is now grouped with {} Levels and {} % Preservation".format(s_copy.name, s_copy.nunique(), 100 * PERCENTAGE))
    else:
        print("{} doesn't exceed the threshold of {} levels. Keeping it unchanged.".format(s_copy.name, THRESHOLD_1))
    return s_copy