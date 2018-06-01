from constants import *
import os
import math
import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np
from matplotlib import pyplot as plt
from localConstants import *

SHOW_PLOTS = False # if true, plots will be printed in the quantiles filter section

USERS = MIT_DATA_FOLDER + r"\\users.csv"
TAPS_ROOT_FOLDER = MIT_DATA_FOLDER + r"\\taps\\"
TAPS_FILE_NAMES = os.listdir(TAPS_ROOT_FOLDER)
TAPS_LOAD_COLUMNS = ["Key", "HoldTime", "releaseTime", "pressTime"]
FLOAT_COLUMNS = ["HoldTime", "LatencyTime", "FlightTime", "pressTime"]
TAPS_COLUMNS = ["ID", "Hand", "Direction"] + FLOAT_COLUMNS

DIRECTIONS = ['LL', 'LR', 'RL', 'RR', 'LS', 'SL', 'RS', 'SR', 'SS']
HANDS = ['L', 'R', 'S']

RIGHT_KEYS = ['y', 'u', 'i', 'o', 'p', 'h', 'j', 'k', 'l', ';', 'n', 'm', ',', '.', '?', '/', 'comma', 'period',
              'colon']
LEFT_KEYS = ['q', 'w', 'e', 'r', 't', 'a', 's', 'd', 'f', 'g', 'z', 'x', 'c', 'v', 'b']
RIGHT_IGNORE_KEYS = ['underscore', 'semicolon', 'question', 'plus', 'apostrophe', 'right', 'num_lock', 'left', 'insert',
                     'end', 'down', 'delete', 'control_r', 'shift_r', 'return', 'minus', 'backspace', '7', '8', '9',
                     '0']
LEFT_IGNORE_KEYS = ['tab', 'escape', 'control_l', 'alt_l', 'shift_l', 'caps_lock']
MIDDLE_KEYS = ['space']
MIDDLE_IGNORE_KEYS = ['6']

PRESS_TIME_BIN_LENGTH = 90 # seconds

BAD_VALUE = "BAD_VALUE"
NEG_VALUE = "NEGATIVE_VALUE"
NAN_VALUE = "NAN_VALUE"
ERROR_VALUES = [NEG_VALUE, BAD_VALUE, NAN_VALUE]

users = pd.read_csv(USERS, delimiter=',', header=0, error_bad_lines=False,
                    low_memory=False, usecols=["pID", "gt", "updrs108", "file_1", "file_2"])


def file_to_id(filename):
    index_by_file_1 = users.index[users.file_1 == filename].tolist()
    index_by_file_2 = users.index[users.file_2 == filename].tolist()
    index_as_list = max(index_by_file_1, index_by_file_2)
    if len(index_as_list) > 0:
        return users.loc[index_as_list[0]]["pID"]
    return -1


def set_hand(row):
    key = str(row['Key']).lower()
    if key in RIGHT_KEYS:
        return 'R'
    if key in RIGHT_IGNORE_KEYS:
        return 'R_Ignore'
    if key in LEFT_KEYS:
        return 'L'
    if key in LEFT_IGNORE_KEYS:
        return 'L_Ignore'
    if key in MIDDLE_KEYS:
        return 'S'
    if key in MIDDLE_IGNORE_KEYS:
        return 'S_Ignore'
    return ' '


def set_direction(current, previous):
    return previous + current


def read_taps_file(filename, dir_path):
    try:
        user_file_df = pd.read_csv(dir_path + filename, delimiter=',', header=None, error_bad_lines=False,
                                   names=TAPS_LOAD_COLUMNS, low_memory=False)
        user_file_df["ID"] = file_to_id(file_name)

        # If no user found for this file
        if user_file_df["ID"][0] == -1:
            raise EmptyDataError

        # Calculate latency and flight time
        user_file_df["LatencyTime"] = user_file_df["pressTime"] - user_file_df["pressTime"].shift(+1)
        user_file_df["FlightTime"] = user_file_df["pressTime"] - user_file_df["releaseTime"].shift(+1)

        # Parse hand and direction by pressed key
        user_file_df["Hand"] = user_file_df.apply(lambda row: set_hand(row), axis=1)
        user_file_df["Direction"] = set_direction(user_file_df["Hand"], user_file_df["Hand"].shift(+1))
        user_file_df.drop(user_file_df.index[0], inplace=True)  # Because first row cannot have a valid direction

        # Keep only necessary columns
        user_file_df = user_file_df[TAPS_COLUMNS]

    except EmptyDataError:
        user_file_df = pd.DataFrame(columns=TAPS_LOAD_COLUMNS)
    return user_file_df


def filter_taps_by_col(column):
    global taps
    len_before = len(taps)
    for err_val in ERROR_VALUES:
        taps = taps[taps[column] != err_val]
    len_after = len(taps)
    print("Filtered out {} rows with bad values in column '{}'".format((len_before - len_after), column))


def str_to_float(s):
    try:
        res = float(s)
        if res < 0:
            return NEG_VALUE
        if math.isnan(res):
            return NAN_VALUE
        return res
    except ValueError:
        return BAD_VALUE


def invalidate_hand(hand):
    if hand in HANDS:
        return hand
    return BAD_VALUE


def invalidate_direction(direction):
    if direction in DIRECTIONS:
        return direction
    return BAD_VALUE


# ############### Filter users that have severe parkinson ###############

# users = users[users["gt"] == False | ((users["gt"] == True) & (users["updrs108"] < 20))]  # <20 is Mild parkinson

# ############### Read taps, build columns and remove bad values ###############

taps_list = []
for file_name in TAPS_FILE_NAMES:
    df = read_taps_file(file_name, TAPS_ROOT_FOLDER)
    taps_list.append(df)
taps = pd.concat(taps_list)

# try to convert float fields to float, set error label else, and invalidate
for col in FLOAT_COLUMNS:
    taps[col] = taps[col].apply(str_to_float)
for col in FLOAT_COLUMNS:
    filter_taps_by_col(col)

taps['Hand'] = taps['Hand'].apply(invalidate_hand)
taps['Direction'] = taps['Direction'].apply(invalidate_direction)
filter_taps_by_col('Hand')
filter_taps_by_col('Direction')

# Group to bin indexes by pressTime
max_press = (int(max(taps["pressTime"]) / 90) + 1) * 90 + 1
user_bins = [i for i in range(0, max_press, 90)]
taps["binIndex"] = pd.cut(taps["pressTime"], user_bins)


# ############### Filter outliers ###############

def plot_percentile(column):
    X = np.linspace(98, 99.9999, 40)
    Y = [np.percentile(taps[column], x) for x in X]
    plt.plot(X, Y)
    plt.title(column + " Percentiles")
    plt.xlabel("Percent")
    plt.ylabel("Percentile Value")
    plt.show()


def filter_column_by_quantile(column, threshold):
    global taps
    len_before = len(taps)
    taps = taps[taps[column] < np.percentile(taps[column], threshold)]
    len_after = len(taps)
    print("Filtered out {} rows with outliers in column '{}'".format((len_before - len_after), column))


if SHOW_PLOTS:
    for col in FLOAT_COLUMNS:
        plot_percentile(col)

filter_column_by_quantile("HoldTime", 99.99)
filter_column_by_quantile("LatencyTime", 99.4)
filter_column_by_quantile("FlightTime", 99.95)

# ############### Save to file ###############

# Taps file
taps.to_csv(MIT_TAPS_INPUT, index=False)

# Users file
users.rename(columns={'pID': 'ID', 'gt': 'Parkinsons', 'updrs108': 'UDPRS'}, inplace=True)
users = users[['ID', 'Parkinsons', 'UDPRS']]
users.to_csv(MIT_USERS_INPUT, index=False)