import constants
import os
import math
import time
import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

USERS_ROOT_FOLDER = constants.DATA_FOLDER + r"\\Archived users\\"
USERS_FILE_NAMES = os.listdir(USERS_ROOT_FOLDER)
TAPS_ROOT_FOLDER = constants.DATA_FOLDER + r"\\Tappy Data\\"
TAPS_FILE_NAMES = os.listdir(TAPS_ROOT_FOLDER)

TAPS_COLUMNS = ['ID', 'Date', 'TimeStamp', 'Hand', 'HoldTime', 'Direction', 'LatencyTime', 'FlightTime']
TAPS_FINAL_COLUMNS = list(set(TAPS_COLUMNS) - {'Date', 'TimeStamp'})
FLOAT_COLUMNS = ['HoldTime', 'LatencyTime', 'FlightTime']
DIRECTIONS = ['LL', 'LR', 'RL', 'RR', 'LS', 'SL', 'RS', 'SR', 'SS']
HANDS = ['L', 'R', 'S']

BAD_VALUE = "BAD_VALUE"
NEG_VALUE = "NEGATIVE_VALUE"
NAN_VALUE = "NAN_VALUE"
ERROR_VALUES = [NEG_VALUE, BAD_VALUE, NAN_VALUE]


# ############### Create users and taps dataframes ###############

def read_user_file(filename, dir_path):
    result = {}
    with open(dir_path + filename) as f:
        for line in f.readlines():
            result['ID'] = filename[5:-4]
            key, value = line.split(": ")
            result[key] = value.strip()
    return result


def read_taps_file(filename, dir_path):
    global file_num
    file_num += 1
    if file_num % 100 == 0:
        print("Loading taps files: %d/%d" % (file_num, len(TAPS_FILE_NAMES)))
    try:
        df = pd.read_csv(dir_path + filename, delimiter='\t', header=None, error_bad_lines=False,
                         usecols=range(8), low_memory=False)
        df.columns = TAPS_COLUMNS
    except EmptyDataError:
        df = pd.DataFrame(columns=TAPS_COLUMNS)

    return df


def create_merged_users_details_file():
    users_list = []
    for file_name in USERS_FILE_NAMES:
        users_list.append(read_user_file(file_name, USERS_ROOT_FOLDER))
    users = pd.DataFrame(users_list)
    users.to_csv(constants.KAGGLE_USERS_INPUT, index=False)
    return users


def create_merged_taps_dataframe():
    global file_num
    taps_list = []
    file_num = 0
    for file_name in TAPS_FILE_NAMES:
        taps_list.append(read_taps_file(file_name, TAPS_ROOT_FOLDER))
    return pd.concat(taps_list)


# ############### Update datatypes of taps dataframe and filter out bad values ###############

def filter_error_values_from_column(df, col):
    len_before = len(df)
    for err_val in ERROR_VALUES:
        df = df[df[col].astype(str) != err_val]
    len_after = len(df)
    print("Filtered out {} rows with bad values in column '{}'".format((len_before - len_after), col))
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


def parsed_time_to_unix(x):
    strptime = datetime.strptime(x, "%y%m%d %H:%M:%S.%f")
    return time.mktime(strptime.timetuple()) * 1e3 + strptime.microsecond / 1e3


def diff_from_initial(row):
    x = BAD_VALUE
    global initial_timestamp_per_id
    try:
        x = row['ParsedDateTime'] - initial_timestamp_per_id[row['ID']]
        if x == 0:
            x = 0.001
    finally:
        return x


def clean_bad_values(df):
    # try to convert float fields to float, set error label else, and invalidate
    for col in FLOAT_COLUMNS:
        df[col] = df[col].apply(str_to_float)
        df = filter_error_values_from_column(df, col)
    # invalidate Direction column
    df['Direction'] = df['Direction'].apply(lambda x: x if x in DIRECTIONS else BAD_VALUE)
    df = filter_error_values_from_column(df, 'Direction')
    # invalidate Hand column
    df['Hand'] = df['Hand'].apply(lambda x: x if x in HANDS else BAD_VALUE)
    df = filter_error_values_from_column(df, 'Hand')
    return df


def clean_incompatible_user_ids(df, users):
    # invalidate all rows with invalid user ID
    df['ID'] = df['ID'].apply(lambda x: x if len(str(x)) == 10 else BAD_VALUE)
    missing_data_users_ids = set(u for u in df["ID"].values) - set(u for u in users["ID"].values)
    print("there are {} unique user IDs in the Tappy data with no entry in the Users file".format(
        len(missing_data_users_ids)))
    df['ID'] = df['ID'].apply(lambda x: x if x not in missing_data_users_ids else BAD_VALUE)
    df = filter_error_values_from_column(df, 'ID')
    return df


def add_cumulative_timestamps_column(df):
    global time, initial_timestamp_per_id
    time0 = time.time()

    print("Starting parsing of timestamps into cumulative time...")
    df['ParsedDateTime'] = df['Date'].astype(str) + " " + df['TimeStamp']
    df = df.drop(['Date', 'TimeStamp'], axis=1)  # save some memory because the df is huge
    df['ParsedDateTime'] = df['ParsedDateTime'].apply(parsed_time_to_unix)
    initial_timestamp_per_id = df.groupby(['ID'])['ParsedDateTime'].min()
    df['PressTimeCumulative'] = df.apply(diff_from_initial, axis=1)
    df = df.drop(['ParsedDateTime'], axis=1)  # save some memory because the df is huge
    df = filter_error_values_from_column(df, 'PressTimeCumulative')

    time = time.time() - time0
    print("Parsing ended, took {} seconds".format(round(time, 2)))
    return df
