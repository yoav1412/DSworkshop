from scipy.stats import iqr, kurtosis, skew
import numpy as np
import pandas as pd
from constants import *


def agg_outliers(series):
    IQR = iqr(series)
    first_quartile = np.percentile(series, 25)
    third_quartile = np.percentile(series, 75)
    is_outliers = []
    for v in series.tolist():
        is_outlier =  v <= first_quartile - 1.5*IQR \
        or v >= third_quartile + 1.5*IQR
        is_outliers.append(is_outlier)
    return np.mean(is_outliers)

def agg_iqr(series):
    first_quartile = np.percentile(series, 25)
    second_quartile = np.percentile(series, 50)
    third_quartile = np.percentile(series, 75)
    return (second_quartile - first_quartile)/(third_quartile - first_quartile)

def agg_histogram(series, which_bin):
    hist = np.histogram(series, range=(0,500), bins=4, normed=True)
    return hist[0][which_bin]

def agg_histogram_bin0(series):
    return agg_histogram(series, 0)

def agg_histogram_bin1(series):
    return agg_histogram(series, 1)

def agg_histogram_bin2(series):
    return agg_histogram(series, 2)

def agg_histogram_bin3(series):
    return agg_histogram(series, 3)

MIN_PRESSES_PER_BUCKET_THRESHOLD = 0 # Filtering on count didn't prove usefull.

# Create the NQI features for both the MIT and the Kaggle Datasets:
for taps_input_file_path, nqi_features_output_filepath in \
        zip([MIT_TAPS_INPUT, KAGGLE_TAPS_INPUT],[MIT_NQI_FEATURES, KAGGLE_NQI_FEATURES]):

    taps = pd.read_csv(taps_input_file_path)
    grouped_taps = taps.groupby(["ID", "binIndex"])["HoldTime"].agg([agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
                                                        agg_histogram_bin2, agg_histogram_bin3, np.count_nonzero])
    t = taps.groupby(["ID", "binIndex"])["FlightTime"].agg([np.mean, np.std]).reset_index()
    nqi_calculator_input = grouped_taps.reset_index()
    nqi_calculator_input = nqi_calculator_input.merge(t, on=["ID","binIndex"])
    nqi_calculator_input = nqi_calculator_input.rename(columns={"FlightTime": "mean_flight"})

    nqi_calculator_input = nqi_calculator_input[nqi_calculator_input.count_nonzero > MIN_PRESSES_PER_BUCKET_THRESHOLD]

    nqi_calculator_input.to_csv(nqi_features_output_filepath, index=False)
