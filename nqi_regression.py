import pandas as pd
import numpy as np
import random
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from constants import *


def list_diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

mit_nqi_features = pd.read_csv(MIT_NQI_FEATURES)
mit_users = pd.read_csv(MIT_USERS_INPUT)

data = mit_nqi_features.merge(mit_users, on="ID").dropna().reset_index().drop("index", axis=1)

PREDICTION_COLUMNS = list_diff(data.columns, ["UDPRS", "Parkinsons", 'binIndex', 'ID'])

scaler = StandardScaler()
min_max_scaler = MinMaxScaler()
data[PREDICTION_COLUMNS] = pd.DataFrame(scaler.fit_transform(data[PREDICTION_COLUMNS]))
data["UDPRS"] = min_max_scaler.fit_transform(data["UDPRS"].values.reshape(-1,1))

X = data[PREDICTION_COLUMNS]
y = data["UDPRS"]


ENSEMBLE_SIZE = 200

# TODO: consider the hyper parameters of the regression
ensemble = BaggingRegressor(base_estimator=SVR(),
                            n_estimators=ENSEMBLE_SIZE,
                            bootstrap=True,
                            n_jobs=1)

ensemble.fit(X=X, y=y)


d11 = data[data.ID == 11].drop(["binIndex","ID","UDPRS","Parkinsons"], axis=1)
d11stan = pd.DataFrame(scaler.transform(d11))

