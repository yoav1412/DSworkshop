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


users_list = []
file_num = 0
for file_name in USERS_FILE_NAMES:
    users_list.append(read_user_file(file_name, USERS_ROOT_FOLDER))
users = pd.DataFrame(users_list)
users.to_csv(constants.KAGGLE_USERS_INPUT)

taps_list = []
for file_name in TAPS_FILE_NAMES:
    taps_list.append(read_taps_file(file_name, TAPS_ROOT_FOLDER))
taps = pd.concat(taps_list)


# ############### Update datatypes of taps dataframe and filter out bad values ###############

def filter_taps_by_col(col):
    global taps
    len_before = len(taps)
    for err_val in ERROR_VALUES:
        taps = taps[taps[col] != err_val]
    len_after = len(taps)
    print("Filtered out {} rows with bad values in column '{}'".format((len_before - len_after), col))


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


# try to convert float fields to float, set error label else, and invalidate
for col in FLOAT_COLUMNS:
    taps[col] = taps[col].apply(str_to_float)
    filter_taps_by_col(col)

# invalidate Direction column
taps['Direction'] = taps['Direction'].apply(lambda x: x if x in DIRECTIONS else BAD_VALUE)
filter_taps_by_col('Direction')

# invalidate Hand column
taps['Hand'] = taps['Hand'].apply(lambda x: x if x in HANDS else BAD_VALUE)
filter_taps_by_col('Hand')

# invalidate all rows with invalid user ID
taps['ID'] = taps['ID'].apply(lambda x: x if len(str(x)) == 10 else BAD_VALUE)
missing_data_users_ids = set(u for u in taps["ID"].values) - set(u for u in users["ID"].values)
print("there are {} unique user IDs in the Tappy data with no entry in the Users file".format(
    len(missing_data_users_ids)))
taps['ID'] = taps['ID'].apply(lambda x: x if x not in missing_data_users_ids else BAD_VALUE)
filter_taps_by_col('ID')


# ############### Filter outliers ###############

def filter_column_by_quantile(column, threshold):
    global taps
    len_before = len(taps)
    taps = taps[taps[column] < np.percentile(taps[column], threshold)]
    len_after = len(taps)
    print("Filtered out {} rows with outliers in column '{}'".format((len_before - len_after), column))


# Filter out outliers of HoldTime:

X = np.linspace(99.96, 99.9999, 20)
Y = [np.percentile(taps['HoldTime'], x) for x in X]
plt.plot(X, Y)
plt.title("HoldTime Percentiles")
plt.xlabel("Percent")
plt.ylabel("Percentile Value")

# After the percentile 99.993 we see significantly higher values, which are definitely outliers.
filter_column_by_quantile('HoldTime', 99.993)


# Add parsed date and time column + calculate cumulative time
time0 = time.time()
print("Starting parsing of timestamps into cumulative time...")
taps['ParsedDateTime'] = taps['Date'].astype(str) + " " + taps['TimeStamp']
taps = taps.drop(['Date', 'TimeStamp'], axis=1)  # save some memory because the df is huge
taps['ParsedDateTime'] = taps['ParsedDateTime'].apply(parsed_time_to_unix)
initial_timestamp_per_id = taps.groupby(['ID'])['ParsedDateTime'].min()
taps['PressTimeCumulative'] = taps.apply(diff_from_initial, axis=1)
taps = taps.drop(['ParsedDateTime'], axis=1)  # save some memory because the df is huge
time1 = time.time()
print("Parsing ended after {} milliseconds".format(time1 - time0))
filter_taps_by_col('PressTimeCumulative')

# Group to bin indexes by PressTimeCumulative
time0 = time.time()
taps["PressTimeCumulative"] = taps["PressTimeCumulative"] / 1000
max_press = (int(max((taps["PressTimeCumulative"])) / 90) + 1) * 90 + 1
user_bins = [i for i in range(0, max_press, 90)]
taps["binIndex"] = pd.cut((taps["PressTimeCumulative"]), user_bins)
time1 = time.time()
print("Binning of timestamps took {} seconds".format(time1 - time0))

# ############### Save to file ###############

# Keep only necessary columns and save to file
taps = taps[TAPS_FINAL_COLUMNS + ['binIndex']]
taps.to_csv(constants.KAGGLE_TAPS_INPUT)
