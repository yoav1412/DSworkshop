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
    try:
        return (second_quartile - first_quartile)/(third_quartile - first_quartile)
    except RuntimeWarning:
        print("{} / {}".format((second_quartile - first_quartile) ,(third_quartile - first_quartile)))
        print(series)
        raise

def agg_histogram(series, which_bin):
    """
    :param series: a pandas series containing numeric values for histogram
    :param which_bin: which of the 4 bins of the histogram to return
    :return: the value of the requested bin in the 4 bin normalized histogram of the given values.
    """
    if len([v for v in series.values if (v>=0 and v<500)]) > 0:
        hist = np.histogram(series, bins=4, range=(0,500), normed=True)
    else: # For edge cases in aggle data, where there are no values in the (0,500) interval.
        hist = np.histogram(series, bins=4, normed=True)
    return hist[0][which_bin]

# Following functions are simple wrappers so that they cn be sent as aggregators to pandas DataFrame.agg :

def agg_histogram_bin0(series):
    return agg_histogram(series, 0)

def agg_histogram_bin1(series):
    return agg_histogram(series, 1)

def agg_histogram_bin2(series):
    return agg_histogram(series, 2)

def agg_histogram_bin3(series):
    return agg_histogram(series, 3)