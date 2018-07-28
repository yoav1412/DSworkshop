import os
import math
import pandas as pd
from pandas.errors import EmptyDataError
from localConstants import *

# ===============
# Configurations:
# ===============

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

PRESS_TIME_BIN_LENGTH = 90  # seconds

BAD_VALUE = "BAD_VALUE"
NEG_VALUE = "NEGATIVE_VALUE"
NAN_VALUE = "NAN_VALUE"
ERROR_VALUES = [NEG_VALUE, BAD_VALUE, NAN_VALUE]


def file_to_id(filename):
    return int(filename[filename.index(".") + 1: filename.index("_")])


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
        user_file_df["ID"] = file_to_id(filename)

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


def create_merged_taps_dataframe():
    taps_list = []
    for file_name in TAPS_FILE_NAMES:
        df = read_taps_file(file_name, TAPS_ROOT_FOLDER)
        taps_list.append(df)
    return pd.concat(taps_list)


def filter_taps_by_col(df, column):
    len_before = len(df)
    for err_val in ERROR_VALUES:
        try:
            df = df[df[column] != err_val]
        except TypeError:
            continue  # Patch due to a bug, throws an error when all lines are fine (not containing error)
    len_after = len(df)
    print("Filtered out {} rows with bad values in column '{}'".format((len_before - len_after), column))
    return df


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


def clean_errors_and_bad_values(df):
    # try to convert float fields to float, set error label else, and invalidate
    for col in FLOAT_COLUMNS:
        df[col] = df[col].apply(str_to_float)
    for col in FLOAT_COLUMNS:
        df = filter_taps_by_col(df, col)

    df['Hand'] = df['Hand'].apply(invalidate_hand)
    df['Direction'] = df['Direction'].apply(invalidate_direction)
    df = filter_taps_by_col(df, 'Hand')
    df = filter_taps_by_col(df, 'Direction')
    return df

