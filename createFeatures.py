import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\yoav1\OneDrive\Desktop\total.csv")

def createSummaryStatistic(full_data,columns_to_aggregate, aggregation_functions):
    """
    :param full_data: pandas dataframe containing all the raw data, one line per keystroke.
    :param columns_to_aggregate: list of columns that we wish to calculate summary statistics from
    :param aggregation_functions: list of function to be applied on each column (example: np.mean or np.std)
    :return: a dataframe with 1 row per user, with requested summary statistics as columns, calculated individually for
             left/right/LL/LR/RR/RL keystrokes
    """
    assert all([col in full_data.columns for col in ["ID","Hand","Direction"]])

    # Calculate statistics for rows that match 'Left' / 'Right' keystrokes:
    left_right_stats = full_data.groupby(["ID", "Hand"])[columns_to_aggregate].\
        agg(aggregation_functions).reset_index()
    leftHandStats = left_right_stats[left_right_stats.Hand == 'L']
    rightHandStats = left_right_stats[left_right_stats.Hand == 'R']
    leftHandStats.columns = ["L_"+col+"_"+stat if col !="ID" else col for col,stat in leftHandStats.columns]
    rightHandStats.columns = ["R_"+col+"_"+stat if col !="ID" else col for col,stat in rightHandStats.columns]

    # Calculate statistics for rows that match 'LR'/'RR'/'RL'/'LL' keystrokes:
    lr_rl_ll_rr_stats = full_data.groupby(["ID", "Direction"])[columns_to_aggregate]. \
        agg(aggregation_functions).reset_index()
    LL_stats = lr_rl_ll_rr_stats[lr_rl_ll_rr_stats.Direction == "LL"]
    LL_stats.columns = ["LL_" + col + "_" + stat if col !="ID" else col for col, stat in LL_stats.columns]
    LR_stats = lr_rl_ll_rr_stats[lr_rl_ll_rr_stats.Direction == "LR"]
    LR_stats.columns = ["LR_" + col + "_" + stat if col !="ID" else col for col, stat in LR_stats.columns]
    RL_stats = lr_rl_ll_rr_stats[lr_rl_ll_rr_stats.Direction == "RL"]
    RL_stats.columns = ["RL_" + col + "_" + stat if col !="ID" else col for col, stat in RL_stats.columns]
    RR_stats = lr_rl_ll_rr_stats[lr_rl_ll_rr_stats.Direction == "RR"]
    RR_stats.columns = ["RR_" + col + "_" + stat if col !="ID" else col for col, stat in RR_stats.columns]

    # Join all dfs together:
    res = leftHandStats.merge(rightHandStats, on="ID").\
        merge(LL_stats, on="ID").merge(LR_stats, on="ID").merge(RL_stats, on="ID").merge(RR_stats, on="ID")

    return res