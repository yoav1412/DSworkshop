import pandas as pd
import numpy as np
import scipy
from sklearn.svm import SVC, NuSVC
import constants
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from collections import namedtuple
from sklearn.decomposition import PCA
from os import cpu_count
from TwoGroupsWeightedModel import TwoGroupsWeightedModel
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from constants import *
from scipy import stats
from sklearn.dummy import DummyClassifier
import warnings
import time
import datetime
raw_tappy_data = pd.read_csv(TAPS_INPUT)
users_data = pd.read_csv(USERS_INPUT)


#Extract UnixTimeStamp from time columns and sort taps by their timestamp:
raw_tappy_data["UnixTime"] = raw_tappy_data.Date.astype(str) + raw_tappy_data.TimeStamp
raw_tappy_data["UnixTime"] = raw_tappy_data.UnixTime.apply(lambda s:
                                            time.mktime(datetime.datetime.strptime(s, "%y%m%d%H:%M:%S.%f").timetuple()))
raw_tappy_data = raw_tappy_data.sort_values(by="UnixTime")


def splitAgg(s, aggregator, num_splits=5):
    l = len(s)
    n = int(l / float(num_splits))
    means=[]
    ind=0
    for i in range(num_splits):
        mean = aggregator(s[ind:ind+n])
        ind=ind+n
        means.append(mean)
    return means

AGGREGATORS = [np.mean, np.std]#, stats.kurtosis]#, stats.skew, percnt10, percnt20,percnt80, percnt90]

num_agg_splits = 5
MIN_COUNT = 500

# get count per patient:
gbc = raw_tappy_data[["ID","HoldTime"]].groupby(["ID"])["HoldTime"]
counts = gbc.count().reset_index()
counts.columns = ["ID","count"]
counts = counts[counts["count"] > MIN_COUNT]

# filter users with low counts:
raw_tappy_data = raw_tappy_data[raw_tappy_data.ID.isin(counts.ID)]

# Split to right and left:
l_tappy_data = raw_tappy_data[raw_tappy_data["Hand"] == "L"]
r_tappy_data = raw_tappy_data[raw_tappy_data["Hand"] == "R"]
ll_tappy_data = raw_tappy_data[raw_tappy_data["Direction"] == "LL"]
lr_tappy_data = raw_tappy_data[raw_tappy_data["Direction"] == "LR"]
rl_tappy_data = raw_tappy_data[raw_tappy_data["Direction"] == "RL"]
rr_tappy_data = raw_tappy_data[raw_tappy_data["Direction"] == "RR"]

def gen_full(tappy_data, side, column):
    selected = tappy_data[["ID",column]]
    gb = selected.groupby("ID")[column]
    full=None
    for aggregator in AGGREGATORS:
        tmp_df = calculate_split_aggregation(gb, aggregator, num_agg_splits, side)
        if full is None:
            full = tmp_df.copy()
            continue
        full = full.merge(tmp_df, on="ID")

    return full


def calculate_split_aggregation(gb, aggregator,num_agg_splits, side):
    agg_name = aggregator.__name__
    aggregated = gb.apply(lambda x: splitAgg(x, aggregator, num_agg_splits)).reset_index()
    aggregated.columns = ["ID", agg_name]
    tmp = pd.DataFrame(aggregated[agg_name].values.tolist(),
                       columns=['{}_{}_{}'.format(side, agg_name, i) for i in range(1, num_agg_splits + 1)])
    tmp["ID"] = aggregated["ID"]
    aggregated = tmp

    return aggregated


full_hold_left = gen_full(l_tappy_data,'L',"HoldTime")
full_hold_right = gen_full(r_tappy_data,'R',"HoldTime")

full_latency_ll = gen_full(ll_tappy_data,'LL',"LatencyTime")
full_latency_lr = gen_full(lr_tappy_data,'LR',"LatencyTime")
full_latency_rl = gen_full(rl_tappy_data,'RL',"LatencyTime")
full_latency_rr = gen_full(rr_tappy_data,'RR',"LatencyTime")

full = full_hold_left.\
    merge(full_hold_right, on="ID", how="outer").\
    merge(full_latency_ll, on="ID", how="outer").\
    merge(full_latency_lr, on="ID", how="outer").\
    merge(full_latency_rl, on="ID", how="outer").\
    merge(full_latency_rr, on="ID", how="outer")

# Drop NA values (importatn to do this BEFORE merging with users data)
before = len(full)
full=full.dropna()
print('Dropped {} rows with NA values.'.format(before - len(full)))

full = full.merge(users_data, on="ID")



Xvars=list( set(full_hold_left.columns.tolist()+
                full_hold_right.columns.tolist()+
                full_latency_ll.columns.tolist() +
                full_latency_lr.columns.tolist() +
                full_latency_rl.columns.tolist() +
                full_latency_rr.columns.tolist())
                - {"ID"})


X = full[Xvars]
y = full["Parkinsons"]



vclf = VotingClassifier(estimators=[('0',GradientBoostingClassifier()),
                                    ('0.5',AdaBoostClassifier()),
                                    ('1',RandomForestClassifier()),
                                    ('2',SVC(probability=True)),
                                   #  ('3',MLPClassifier()),
                                    ('4',LogisticRegression()),
                                 #   ('5',NuSVC(probability=True)),
                                    ('6',KNeighborsClassifier()),
                                  #  ('7',DecisionTreeClassifier()),
                                   # ('8',QuadraticDiscriminantAnalysis())
                                  ],
                        voting='soft',)


clf=RandomForestClassifier()
print(np.mean(cross_val_score(clf, X, y, cv=10)))
print("ensemble: ",np.mean(cross_val_score(vclf, X, y, cv=10)))


def get_random_test_split_accuracy(clf, X, y, test_percentage, apply_dim_reduction=None, n_components=None):
    test_size = int(test_percentage * len(y))
    indices = [i for i in range(len(y))]
    test_indices = np.random.choice(indices, size=test_size)
    train_indices = [i for i in indices if i not in test_indices]
    train_X = X.iloc[train_indices]
    train_y = y.iloc[train_indices]
    test_X = X.iloc[test_indices]
    test_y = y.iloc[test_indices]
    if apply_dim_reduction:
        if apply_dim_reduction == 'lda':
            lda = LinearDiscriminantAnalysis()
            lda.fit(train_X, train_y)
            train_X = lda.transform(train_X)
            test_X = lda.transform(test_X)
        elif apply_dim_reduction == 'pca':
            if n_components is None:
                print('Error: specify n_component for PCA...')
                raise ValueError
            pca = PCA(n_components=n_components)
            pca.fit(train_X, train_y)
            train_X = pca.transform(train_X)
            test_X = pca.transform(test_X)
    clf.fit(train_X,train_y)
    return clf.score(test_X, test_y)

best_dim = -1
best_score = -1

for nc in range(1,len(X.columns)):
    print('testing reduction to dim=',nc)
    warnings.filterwarnings("ignore") #TODO: dont supress warnings..
    score = np.mean([get_random_test_split_accuracy(vclf, X,y,0.2,apply_dim_reduction='pca', n_components=nc) for _ in range(30)])
    warnings.resetwarnings()
    if score > best_score:
        best_score=score
        best_dim=nc


def percnt90(series):
    if len(series)==0:
        return 0
    return np.percentile(series, 90)


def percnt80(series):
    if len(series)==0:
        return 0
    return np.percentile(series, 80)


def percnt70(series):
    if len(series)==0:
        return 0
    return np.percentile(series, 70)


def percnt60(series):
    if len(series)==0:
        return 0
    return np.percentile(series, 60)


def percnt40(series):
    if len(series)==0:
        return 0
    return np.percentile(series, 40)


def percnt20(series):
    if len(series)==0:
        return 0
    return np.percentile(series, 20)


def percnt10(series):
    if len(series)==0:
        return 0
    return np.percentile(series, 10)
