import pandas as pd
import numpy as np
from scipy import stats
from constants import *

raw_tappy_data = pd.read_csv(TAPS_INPUT)
users_data = pd.read_csv(USERS_INPUT)


def create_summary_statistics(full_data, columns_to_aggregate, aggregation_functions):
    """
    :param full_data: pandas dataframe containing all the raw data, one line per keystroke.
    :param columns_to_aggregate: list of columns that we wish to calculate summary statistics from
    :param aggregation_functions: list of function to be applied on each column (example: np.mean or np.std)
    :return: a dataframe with 1 row per user, with requested summary statistics as columns, calculated individually for
             left/right/LL/LR/RR/RL keystrokes
    """
    assert all([col in full_data.columns for col in ["ID", "Hand", "Direction"]])

    # Calculate statistics for rows that match 'Left' / 'Right' keystrokes:
    left_right_stats = full_data.groupby(["ID", "Hand"])[columns_to_aggregate]. \
        agg(aggregation_functions).reset_index()
    leftHandStats = left_right_stats[left_right_stats.Hand == 'L']
    rightHandStats = left_right_stats[left_right_stats.Hand == 'R']
    leftHandStats.columns = ["L_" + col + "_" + stat if col not in ("ID", "Hand") else col for col, stat in
                             leftHandStats.columns]
    rightHandStats.columns = ["R_" + col + "_" + stat if col not in ("ID", "Hand") else col for col, stat in
                              rightHandStats.columns]

    # Calculate statistics for rows that match 'LR'/'RR'/'RL'/'LL' keystrokes:
    lr_rl_ll_rr_stats = full_data.groupby(["ID", "Direction"])[columns_to_aggregate]. \
        agg(aggregation_functions).reset_index()
    ll_stats = lr_rl_ll_rr_stats[lr_rl_ll_rr_stats.Direction == "LL"]
    ll_stats.columns = ["LL_" + col + "_" + stat if col not in ("ID", "Direction") else col for col, stat in
                        ll_stats.columns]
    lr_stats = lr_rl_ll_rr_stats[lr_rl_ll_rr_stats.Direction == "LR"]
    lr_stats.columns = ["LR_" + col + "_" + stat if col not in ("ID", "Direction") else col for col, stat in
                        lr_stats.columns]
    rl_stats = lr_rl_ll_rr_stats[lr_rl_ll_rr_stats.Direction == "RL"]
    rl_stats.columns = ["RL_" + col + "_" + stat if col not in ("ID", "Direction") else col for col, stat in
                        rl_stats.columns]
    rr_stats = lr_rl_ll_rr_stats[lr_rl_ll_rr_stats.Direction == "RR"]
    rr_stats.columns = ["RR_" + col + "_" + stat if col not in ("ID", "Direction") else col for col, stat in
                        rr_stats.columns]

    total_num_rows_count = full_data.groupby(["ID"])["Hand"].count().reset_index()
    total_num_rows_count.columns = ["ID", "total_count"]
    # Join all dfs together:
    res = leftHandStats.merge(rightHandStats, on="ID", how="outer"). \
        merge(ll_stats, on="ID", how="outer").merge(lr_stats, on="ID", how="outer"). \
        merge(rl_stats, on="ID", how="outer").merge(rr_stats, on="ID", how="outer"). \
        merge(total_num_rows_count, on="ID", how="outer")

    # Check if any users were lost during the aggregation process:
    if len(set(res.ID.values)) < len(set(full_data.ID.values)):
        s = set(full_data.ID.values)
        lost = [u for u in s if u not in res.ID.values]
        print("lost {} unique IDS in the aggregation process: ".format(len(lost)), lost)

    res.set_index("ID")
    cols = set(res.columns.values.copy())
    for col in cols:
        if col.startswith("Hand") or col.startswith("Direction"):
            res = res.drop(col, axis=1)
    return res


# We check the correlation between the main columns in the raw data:
RAW_DATA_COLUMNS = ["FlightTime", "HoldTime", "LatencyTime"]
corrdf = pd.DataFrame()
corrdf = corrdf.append([{"columns": "FlightTime-HoldTime",
                         "corr": stats.pearsonr(raw_tappy_data.FlightTime, raw_tappy_data.HoldTime)[0]}])
corrdf = corrdf.append([{"columns": "FlightTime-LatencyTime",
                         "corr": stats.pearsonr(raw_tappy_data.FlightTime, raw_tappy_data.LatencyTime)[0]}])
corrdf = corrdf.append([{"columns": "LatencyTime-HoldTime",
                         "corr": stats.pearsonr(raw_tappy_data.LatencyTime, raw_tappy_data.HoldTime)[0]}])
print(corrdf)


# TODO: insert an epxlanation that this explains why in the article they did not use "LatencyTime"

# Create a processed dataset with desired features calculated per user:

def percnt90(series):
    return np.percentile(series, 90)


def percnt80(series):
    return np.percentile(series, 80)


def percnt70(series):
    return np.percentile(series, 70)


def percnt60(series):
    return np.percentile(series, 60)


def percnt40(series):
    return np.percentile(series, 40)


def percnt20(series):
    return np.percentile(series, 20)


def percnt10(series):
    return np.percentile(series, 10)


data = create_summary_statistics(raw_tappy_data,
                                 columns_to_aggregate=["FlightTime", "HoldTime", "LatencyTime"],
                                 aggregation_functions=[np.mean, np.std, stats.kurtosis, stats.skew,
                                                        stats.entropy, percnt10, percnt20, percnt40, percnt60, percnt70,
                                                        percnt80, percnt90])
# Add a feature of the mean-diff between Left and Right HoldTimes, and Between LR and RL LatencyTimes:
data["mean_diff_L_R_HoldTime"] = data.R_HoldTime_mean - data.L_HoldTime_mean
data["mean_diff_LR_RL_LatencyTime"] = data.RL_LatencyTime_mean - data.LR_LatencyTime_mean
data["mean_diff_LL_RR_LatencyTime"] = data.LL_LatencyTime_mean - data.RR_LatencyTime_mean

# Join with the Users data:
data = data.merge(users_data, on="ID", how="left")

data.to_csv(FINAL_DATASET)
