from nqi_feature_creation_functions import *
import pandas as pd

def create_nqi_features_from_raw_data_with_sides_partitions(raw_data_input_path, output_path):
    """
    :param raw_data_path: path to raw 'taps' data file
    :return: creates nqi features with sides partitions (measures of patient assymetry), saves to csv and returns path to result file
    """
    taps = pd.read_csv(raw_data_input_path)


    l_grouped_taps = taps[taps.Hand == "L"].groupby(["ID", "binIndex"]).filter(
        lambda x: np.percentile(x.HoldTime, 25) != np.percentile(x.HoldTime, 75))
    l_grouped_taps = l_grouped_taps[l_grouped_taps.Hand == "L"].groupby(["ID", "binIndex"])["HoldTime"].agg(
        [agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
         agg_histogram_bin2, agg_histogram_bin3, np.std])
    l_grouped_taps.rename(columns={c: 'L_' + c for c in l_grouped_taps.columns}, inplace=True)

    r_grouped_taps = taps[taps.Hand == "R"].groupby(["ID", "binIndex"]).filter(
        lambda x: np.percentile(x.HoldTime, 25) != np.percentile(x.HoldTime, 75))
    r_grouped_taps = r_grouped_taps[r_grouped_taps.Hand == "R"].groupby(["ID", "binIndex"])["HoldTime"].agg(
        [agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
         agg_histogram_bin2, agg_histogram_bin3, np.std])
    r_grouped_taps.rename(columns={c: 'R_' + c for c in r_grouped_taps.columns}, inplace=True)

    lr_grouped_taps = taps[taps.Direction == "LR"].groupby(["ID", "binIndex"]).filter(
        lambda x: np.percentile(x.LatencyTime, 25) != np.percentile(x.LatencyTime, 75))
    lr_grouped_taps = lr_grouped_taps[lr_grouped_taps.Direction == "LR"].groupby(["ID", "binIndex"])["LatencyTime"].agg(
        [agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
         agg_histogram_bin2, agg_histogram_bin3, np.std])
    lr_grouped_taps.rename(columns={c: 'LR_' + c for c in lr_grouped_taps.columns}, inplace=True)

    ll_grouped_taps = taps[taps.Direction == "LL"].groupby(["ID", "binIndex"]).filter(
        lambda x: np.percentile(x.LatencyTime, 25) != np.percentile(x.LatencyTime, 75))
    ll_grouped_taps = ll_grouped_taps[ll_grouped_taps.Direction == "LL"].groupby(["ID", "binIndex"])["LatencyTime"].agg(
        [agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
         agg_histogram_bin2, agg_histogram_bin3, np.std])
    ll_grouped_taps.rename(columns={c: 'LL_' + c for c in ll_grouped_taps.columns}, inplace=True)

    rl_grouped_taps = taps[taps.Direction == "RL"].groupby(["ID", "binIndex"]).filter(
        lambda x: np.percentile(x.LatencyTime, 25) != np.percentile(x.LatencyTime, 75))
    rl_grouped_taps = rl_grouped_taps[rl_grouped_taps.Direction == "RL"].groupby(["ID", "binIndex"])["LatencyTime"].agg(
        [agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
         agg_histogram_bin2, agg_histogram_bin3, np.std])
    rl_grouped_taps.rename(columns={c: 'RL_' + c for c in rl_grouped_taps.columns}, inplace=True)

    rr_grouped_taps = taps[taps.Direction == "RR"].groupby(["ID", "binIndex"]).filter(
        lambda x: np.percentile(x.LatencyTime, 25) != np.percentile(x.LatencyTime, 75))
    rr_grouped_taps = rr_grouped_taps[rr_grouped_taps.Direction == "RR"].groupby(["ID", "binIndex"])["LatencyTime"].agg(
        [agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
         agg_histogram_bin2, agg_histogram_bin3, np.std])
    rr_grouped_taps.rename(columns={c: 'RR_' + c for c in rr_grouped_taps.columns}, inplace=True)

    grouped_taps = l_grouped_taps.merge(r_grouped_taps, on=["ID", "binIndex"]). \
        merge(lr_grouped_taps, on=["ID", "binIndex"]). \
        merge(ll_grouped_taps, on=["ID", "binIndex"]). \
        merge(rl_grouped_taps, on=["ID", "binIndex"]). \
        merge(rr_grouped_taps, on=["ID", "binIndex"])

    t = taps.groupby(["ID", "binIndex"]).filter(
        lambda x: np.percentile(x.FlightTime, 25) != np.percentile(x.FlightTime, 75))
    t = t.groupby(["ID", "binIndex"])["FlightTime"].agg(
        [np.mean, agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
         agg_histogram_bin2, agg_histogram_bin3, np.std]).reset_index()
    nqi_calculator_input = grouped_taps.reset_index()
    nqi_calculator_input = nqi_calculator_input.merge(t, on=["ID", "binIndex"])
    nqi_calculator_input = nqi_calculator_input.rename(columns={"FlightTime": "mean_flight"})

    nqi_calculator_input.to_csv(output_path, index=False)
    return output_path
