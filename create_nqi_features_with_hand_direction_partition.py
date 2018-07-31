from nqi_feature_creation_functions import *
import pandas as pd


def create_nqi_features_from_raw_data_with_sides_partitions(raw_data_input_path, output_path):
    """
    :param raw_data_path: path to raw 'taps' data file
    :return: creates nqi features with sides partitions (measures of patient assymetry), saves to csv and returns path to result file
    """
    taps = pd.read_csv(raw_data_input_path)

    bin_identifier = ["ID", "binIndex"]
    l_grouped_taps = taps[taps.Hand == "L"].groupby(bin_identifier).filter(
        lambda x: np.percentile(x.HoldTime, 25) != np.percentile(x.HoldTime, 75))
    l_grouped_taps = l_grouped_taps[l_grouped_taps.Hand == "L"].groupby(bin_identifier)["HoldTime"].agg(
        [agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
         agg_histogram_bin2, agg_histogram_bin3])
    l_grouped_taps.rename(columns={c: 'L_hold_' + c for c in l_grouped_taps.columns if c not in bin_identifier}, inplace=True)

    r_grouped_taps = taps[taps.Hand == "R"].groupby(bin_identifier).filter(
        lambda x: np.percentile(x.HoldTime, 25) != np.percentile(x.HoldTime, 75))
    r_grouped_taps = r_grouped_taps[r_grouped_taps.Hand == "R"].groupby(bin_identifier)["HoldTime"].agg(
        [agg_outliers, agg_iqr, agg_histogram_bin0, agg_histogram_bin1,
         agg_histogram_bin2, agg_histogram_bin3])
    r_grouped_taps.rename(columns={c: 'R_hold_' + c for c in r_grouped_taps.columns if c not in bin_identifier}, inplace=True)

    lr_grouped_taps = taps[taps.Direction == "LR"].groupby(bin_identifier).filter(
        lambda x: np.percentile(x.LatencyTime, 25) != np.percentile(x.LatencyTime, 75))
    lr_grouped_taps = lr_grouped_taps[lr_grouped_taps.Direction == "LR"].groupby(bin_identifier)["FlightTime"].agg(
        [np.mean, np.std])
    lr_grouped_taps.rename(columns={c: 'LR_flight_' + c for c in lr_grouped_taps.columns if c not in bin_identifier}, inplace=True)

    ll_grouped_taps = taps[taps.Direction == "LL"].groupby(bin_identifier).filter(
        lambda x: np.percentile(x.LatencyTime, 25) != np.percentile(x.LatencyTime, 75))
    ll_grouped_taps = ll_grouped_taps[ll_grouped_taps.Direction == "LL"].groupby(bin_identifier)["FlightTime"].agg(
        [np.mean, np.std])
    ll_grouped_taps.rename(columns={c: 'LL_flight_' + c for c in ll_grouped_taps.columns if c not in bin_identifier}, inplace=True)

    rl_grouped_taps = taps[taps.Direction == "RL"].groupby(bin_identifier).filter(
        lambda x: np.percentile(x.LatencyTime, 25) != np.percentile(x.LatencyTime, 75))
    rl_grouped_taps = rl_grouped_taps[rl_grouped_taps.Direction == "RL"].groupby(bin_identifier)["FlightTime"].agg(
        [np.mean, np.std])
    rl_grouped_taps.rename(columns={c: 'RL_flight_' + c for c in rl_grouped_taps.columns if c not in bin_identifier}, inplace=True)

    rr_grouped_taps = taps[taps.Direction == "RR"].groupby(bin_identifier).filter(
        lambda x: np.percentile(x.LatencyTime, 25) != np.percentile(x.LatencyTime, 75))
    rr_grouped_taps = rr_grouped_taps[rr_grouped_taps.Direction == "RR"].groupby(bin_identifier)["FlightTime"].agg(
        [np.mean, np.std])
    rr_grouped_taps.rename(columns={c: 'RR_flight_' + c for c in rr_grouped_taps.columns if c not in bin_identifier}, inplace=True)

    grouped_taps = l_grouped_taps.merge(r_grouped_taps, on=bin_identifier). \
        merge(lr_grouped_taps, on=bin_identifier). \
        merge(ll_grouped_taps, on=bin_identifier). \
        merge(rl_grouped_taps, on=bin_identifier). \
        merge(rr_grouped_taps, on=bin_identifier)

    nqi_calculator_input = grouped_taps.reset_index()
    nqi_calculator_input.to_csv(output_path, index=False)
    return output_path
