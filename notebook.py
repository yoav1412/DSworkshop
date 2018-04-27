
# coding: utf-8

# ## Load all packages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from pandas.io.common import EmptyDataError
plt.rcParams['figure.figsize'] = (10, 6)
style.use('ggplot')

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, KFold, StratifiedKFold
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone, BaseEstimator, TransformerMixin

import os
from functools import partial
import re
from collections import Counter


# ## Load Data
user_root = r"C:\Users\yoav1\PycharmProjects\Parkinsons\Data\Archived users\\"
user_fn_list = os.listdir(user_root[:-1])

def read_one_file(fn, root):
    out = dict()
    with open(root + fn) as f:
        for line in f.readlines():
            k, v = line.split(": ")
            out[k] = v.strip()
            out['ID'] = re.findall(r'_(\w+)\.', fn)[0]
    return out


users_list = list(map(partial(read_one_file, root=user_root), user_fn_list))

users = pd.DataFrame(users_list)
users.replace('------', np.nan, inplace=True)
users.replace('', np.nan, inplace=True)
users['Levadopa'] = users['Levadopa'] == 'True'
users['MAOB'] = users['MAOB'] == 'True'
users['Parkinsons'] = users['Parkinsons'] == 'True'
users['Tremors'] = users['Tremors'] == 'True'
users['Other'] = users['Other'] == 'True'

#print(users.head())
print("Finished loading users data. moving to Tappy data..\n")

# We now move on to key logging data:
keys_root = r"C:\Users\yoav1\PycharmProjects\Parkinsons\Data\Tappy Data\\"
keys_fn_list = os.listdir(keys_root[:-1])


# The key logging data are in CSV format, as shown below. We may read all of them with Pandas and combine the dataframes.

# In[ ]:


sample = pd.read_csv(keys_root + keys_fn_list[0], delimiter='\t', header=None, usecols=range(8))
sample.columns = ['ID', 'Date', 'TS', 'Hand', 'HoldTime', 'Direction', 'LatencyTime', 'FlightTime']
sample.head()


# There are some broken lines in the CSVs here, therefore we have to define a custom read function]
#  that attempts to throw out ill-formed lines when necessary. We match the dataframe rows with the standard
# format for each column and remove the rows that fail to match in the final output.

# In[ ]:


def read_one_key_file(fn, root):
    try:
        df = pd.read_csv(root + fn, delimiter='\t', header=None, error_bad_lines=False,
                         usecols=range(8), low_memory=False,
                        dtype={0:'str', 1:'str', 2:'str', 3:'str', 4:'float', 5:'str', 6:'float', 7:'float'})
        df.columns = ['ID', 'Date', 'TS', 'Hand', 'HoldTime', 'Direction', 'LatencyTime', 'FlightTime']
    except ValueError:
        # should try to remove the bad lines and return
#         df = pd.DataFrame(columns = ['ID', 'Date', 'TS', 'Hand', 'HoldTime', 'Direction', 'LatencyTime', 'FlightTime'])
        try:
            df = pd.read_csv(root + fn, delimiter='\t', header=None, error_bad_lines=False,
                             usecols=range(8), low_memory=False)
            df.columns = ['ID', 'Date', 'TS', 'Hand', 'HoldTime', 'Direction', 'LatencyTime', 'FlightTime']
            df = df[df['ID'].apply(lambda x: len(str(x)) == 10)
                   & df['Date'].apply(lambda x: len(str(x)) == 6)
                   & df['TS'].apply(lambda x: len(str(x)) == 12)
                   & np.in1d(df['Hand'], ["L", "R", "S"])
                   & df['HoldTime'].apply(lambda x: re.search(r"[^\d.]", str(x)) is None)
                   & np.in1d(df['Direction'], ['LL', 'LR', 'RL', 'RR', 'LS', 'SL', 'RS', 'SR', 'RR'])
                   & df['LatencyTime'].apply(lambda x: re.search(r"[^\d.]", str(x)) is None)
                   & df['FlightTime'].apply(lambda x: re.search(r"[^\d.]", str(x)) is None)]
            df['HoldTime'] = df['HoldTime'].astype(np.float)
            df['LatencyTime'] = df['LatencyTime'].astype(np.float) #TODO: yoav: in this and the following line, changed rhs key to match lhs key
            df['FlightTime'] = df['FlightTime'].astype(np.float)
        except EmptyDataError:
            df =  pd.DataFrame(columns = ['ID', 'Date', 'TS', 'Hand', 'HoldTime',
                                          'Direction', 'LatencyTime', 'FlightTime'])
    except EmptyDataError:
        df =  pd.DataFrame(columns = ['ID', 'Date', 'TS', 'Hand', 'HoldTime',
                                      'Direction', 'LatencyTime', 'FlightTime'])
    return df


keys_list = list(map(partial(read_one_key_file, root=keys_root), keys_fn_list))

print("Finished loading Tappy data. will now process data set..\n")


keys = pd.concat(keys_list, ignore_index=True, axis=0)
keys.head()


# There are a total of 9276350 valid logged keystrokes.
keys.shape


# Let us see if the keylogs data and user data match:

key_id_set = set(keys['ID'].unique())
print("total user ids in key logs: {0}".format(len(key_id_set)))
user_id_set = set(users['ID'].unique())
print("total user ids in user info: {0}".format(len(user_id_set)))
overlap_id_set = key_id_set.intersection(user_id_set)
print("overlapping ids: {0}".format(len(overlap_id_set)))
diff_id_set = key_id_set.symmetric_difference(user_id_set)
print("non-matching ids: {0}".format(len(diff_id_set)))


# We now try to produce the subset of user and keystroke data according to the orignal research's selection criteria.
# 
# > • Those with at least 2000 keystrokes 
# 
# > • Of the ones with PD, just the ones with ‘Mild’ severity (since the study was into the detection
# > of PD at its early stage, not later stages)
# 
# > • Those not taking levodopa (Sinemet1 and the like), in order to prevent any effect of that
# > medication on their keystroke characteristics.


user_w_sufficient_data = set((keys.groupby('ID').size() >= 2000).index)
user_eligible = set(users[((users['Parkinsons']) & (users['Impact'] == 'Mild') 
                       | (~users['Parkinsons']))
                      & (~users['Levadopa'])]['ID'])
user_valid = user_w_sufficient_data.intersection(user_eligible)


# In the original research, there were 53 participants. However, using the same criteria, we found 87 valid participants. It appears that the data we have do not match that used in the original research.
print("num of valid users (by article criteria: ", len(user_valid))


# We have 55 non-PD and 32 PD participants here.
users.query('ID in @user_valid').groupby('Parkinsons').size()


# We remove keystrokes with negative hold/latency times (error) and with very long hold/latency times
#  (more likely to be deliberate). This is not mentioned in the original paper but seems to improve later performance.

valid_keys = keys[(keys['HoldTime'] > 0)
                   & (keys['LatencyTime'] > 0)
                   & (keys['HoldTime'] < 2000)
                   & (keys['LatencyTime'] < 2000)
                   & np.in1d(keys['ID'], list(user_valid))]

valid_keys.shape

print("will now calc mean, standard deviation, skewness and kurtosis for X fields..\n")
# We now calculate the mean, standard deviation, skewness and kurtosis for the following fields:
# 
# 1. L/R hand hold time
# 2. LL/LR/RL/RR transition latency
# 
# We also calculate the mean difference between L/R hold time, LR/RL latency and LL/RR latency.

hold_by_user =  valid_keys[valid_keys['Hand'] != 'S'].groupby(['ID', 'Hand'])['HoldTime'].agg([np.mean, np.std, skew, kurtosis])
hold_by_user.head(10)

latency_by_user = valid_keys[np.in1d(valid_keys['Direction'], ['LL', 'LR', 'RL', 'RR'])].groupby(['ID', 'Direction'])['LatencyTime'].agg([np.mean, np.std, skew, kurtosis])
latency_by_user.head(10)

hold_by_user_flat = hold_by_user.unstack()
hold_by_user_flat.columns = ['_'.join(col).strip() for col in hold_by_user_flat.columns.values]
hold_by_user_flat['mean_hold_diff'] = hold_by_user_flat['mean_L'] - hold_by_user_flat['mean_R']
hold_by_user_flat.head()

latency_by_user_flat = latency_by_user.unstack()
latency_by_user_flat.columns = ['_'.join(col).strip() for col in latency_by_user_flat.columns.values]
latency_by_user_flat['mean_LR_RL_diff'] = latency_by_user_flat['mean_LR'] - latency_by_user_flat['mean_RL']
latency_by_user_flat['mean_LL_RR_diff'] = latency_by_user_flat['mean_LL'] - latency_by_user_flat['mean_RR']
latency_by_user_flat.head()


# We now combine the hold time data and latency data together into the final dataset for machine learning.
combined = pd.concat([hold_by_user_flat, latency_by_user_flat], axis=1)

combined.shape
combined.head()


full_set = pd.merge(combined.reset_index(), users[['ID', 'Parkinsons']], on='ID')
full_set.set_index('ID', inplace=True)
full_set.dropna(inplace=True)  # should investigate why there are NAs despite choosing sequence length >= 2000
full_set.shape


full_set.head()

print("finished preparing dataset. will apply models..\n")

#full_set.to_csv(r"C:\Users\yoav1\OneDrive\Desktop\full_set_fixed.csv")
# Now that we have the full dataset, we may start replicating the machine learning pipeline in the original paper. We start with a model with the same data processing pipeline but without the ensemble of multiple different models.

# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(full_set.iloc[:, :-1], full_set.iloc[:, -1],
                                                    stratify=full_set.iloc[:, -1], random_state=7777)

#TODO: yoav: after above fix, train_X and test_X return from train_test_split with bad index, causes err when trying to train model. fixing:


class SubsetTransformer(TransformerMixin):
    def __init__(self, start=0, end=None):
        self.start = start
        self.end = end
        
    def fit(self, *_):
        return self
    
    def transform(self, X, *_):
        if self.end is None:
            return X.iloc[:, self.start:]
        else:
            return X.iloc[:, self.start: self.end]


select_left = SubsetTransformer(0, 9)
select_right = SubsetTransformer(9, -1)
scale = StandardScaler()
lda = LinearDiscriminantAnalysis()
rf = RandomForestClassifier(n_estimators=100, max_depth=5)

# ensemble here
pl_left = Pipeline([('select_left', select_left),
                    ('normalise', scale),
                    ('LDA', lda),
                    ('classify', rf)])
pl_right = Pipeline([('select_left', select_right),
                    ('normalise', clone(scale)),
                    ('LDA', clone(lda)),
                    ('classify', clone(rf))])

def gen_pipelines(clf):
    pl_left = Pipeline([('select_left', select_left),
                        ('normalise', scale),
                        ('LDA', lda),
                        ('classify', clf)])
    pl_right = Pipeline([('select_left', select_right),
                         ('normalise', clone(scale)),
                         ('LDA', clone(lda)),
                         ('classify', clone(clf))])
    return pl_left, pl_right


vote = VotingClassifier([('left', pl_left), ('right', pl_right)], weights=[1, 1.2], voting="soft")

scoring = ['accuracy', 'precision', 'recall', 'f1']
scores = cross_validate(vote, train_X, train_y,
                        cv=StratifiedKFold(n_splits=10, random_state=7777), scoring=scoring, return_train_score=True)

pd.DataFrame(scores).mean()


# As we see here, we are not able to get close to the cross validation performance recorded in the original research.
