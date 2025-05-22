def manage_categorical(series, min_levels=5, min_freq=0.05, coverage=0.95):
    series_copy = series.copy()
    if series_copy.nunique() >= min_levels:
        keep_1 = (
            series_copy
            .value_counts(normalize=True)
            .where(lambda i: i >= min_freq)
            .dropna()
            .index
            .tolist()
        )

        keep_2 = (
            series_copy
            .value_counts(normalize=True)
            .cumsum()
            .where(lambda x: x <= coverage)
            .dropna()
            .index
            .tolist()
        )

        keep = set(keep_1).union(set(keep_2))

        series_copy[-series_copy.isin(keep)] = 'Other'
        series_copy = series_copy.map(lambda x: '_'.join(str(x).split()))
        print("{} now has {} Levels and {} % Coverage".format(series_copy.name, series_copy.nunique(), 100 * coverage))
    else:
        print("{} doesn't have more than {} levels. Returning as-is.".format(series_copy.name, min_levels))
    return series_copy