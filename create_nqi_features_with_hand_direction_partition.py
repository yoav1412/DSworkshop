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



taps = pd.read_csv(MIT_TAPS_INPUT)
l_grouped_taps = taps[taps.Hand == "L"].groupby(["ID", "binIndex"])["HoldTime"].agg([agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
                                                    agg_histogram_bin2, agg_histogram_bin3, np.std])

r_grouped_taps = taps[taps.Hand == "L"].groupby(["ID", "binIndex"])["HoldTime"].agg([agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
                                                    agg_histogram_bin2, agg_histogram_bin3, np.std])

lr_grouped_taps = taps[taps.Direction == "LR"].groupby(["ID", "binIndex"])["LatencyTime"].agg([agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
                                                    agg_histogram_bin2, agg_histogram_bin3, np.std])
ll_grouped_taps = taps[taps.Direction== "LL"].groupby(["ID", "binIndex"])["LatencyTime"].agg([agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
                                                    agg_histogram_bin2, agg_histogram_bin3, np.std])
rl_grouped_taps = taps[taps.Direction == "RL"].groupby(["ID", "binIndex"])["LatencyTime"].agg([agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
                                                    agg_histogram_bin2, agg_histogram_bin3, np.std])
rr_grouped_taps = taps[taps.Direction == "RR"].groupby(["ID", "binIndex"])["LatencyTime"].agg([agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
                                                    agg_histogram_bin2, agg_histogram_bin3, np.std])


grouped_taps = l_grouped_taps.merge(r_grouped_taps, on=["ID", "binIndex"]).\
    merge(lr_grouped_taps, on=["ID", "binIndex"]).\
    merge(ll_grouped_taps, on=["ID", "binIndex"]).\
merge(rl_grouped_taps, on=["ID", "binIndex"]).\
merge(rr_grouped_taps, on=["ID", "binIndex"])


t = taps.groupby(["ID", "binIndex"])["FlightTime"].agg([np.mean,agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
                                                    agg_histogram_bin2, agg_histogram_bin3, np.std]).reset_index()
nqi_calculator_input = grouped_taps.reset_index()
nqi_calculator_input = nqi_calculator_input.merge(t, on=["ID","binIndex"])
nqi_calculator_input = nqi_calculator_input.rename(columns={"FlightTime":"mean_flight"})

nqi_calculator_input.to_csv(MIT_NQI_FEATURES, index=False)
