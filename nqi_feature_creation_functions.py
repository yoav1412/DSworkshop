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