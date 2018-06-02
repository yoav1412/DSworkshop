import pandas as pd
import numpy as np
import random

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import BaggingRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostRegressor, \
    RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.tree import DecisionTreeClassifier

from constants import *


def list_diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

mit_nqi_features = pd.read_csv(MIT_NQI_FEATURES)
mit_users = pd.read_csv(MIT_USERS_INPUT)

data = mit_nqi_features.merge(mit_users, on="ID").dropna().reset_index().drop("index", axis=1)

PREDICTION_COLUMNS = list_diff(data.columns, ["UDPRS", "Parkinsons", 'binIndex', 'ID'])

scaler = StandardScaler()
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))


# Split the unified MIT dataset to train and test according to ID's:
train_mit_data = data[data.ID <= 100].copy()
test_mit_data = data[data.ID >= 1000].copy()

train_mit_X = train_mit_data[PREDICTION_COLUMNS]
train_mit_y = train_mit_data["UDPRS"]

train_mit_X = pd.DataFrame(scaler.fit_transform(train_mit_X))
train_mit_y = min_max_scaler.fit_transform(train_mit_y.values.reshape(-1,1))


test_mit_X = test_mit_data[PREDICTION_COLUMNS]
test_mit_y = test_mit_data["UDPRS"]
test_mit_X = pd.DataFrame(scaler.transform(test_mit_X))
test_mit_y = min_max_scaler.transform(test_mit_y.values.reshape(-1,1))



ENSEMBLE_SIZE = 500

# TODO: consider the hyper parameters of the regression
ensemble = BaggingRegressor(base_estimator=SVR(),
                            n_estimators=ENSEMBLE_SIZE,
                            bootstrap=True,
                            n_jobs=1)
ensemble = GradientBoostingRegressor(n_estimators=1000)

ensemble.fit(X=train_mit_X, y=np.ravel(train_mit_y))

mit_train_nqi_predictions = ensemble.predict(train_mit_X)

train_mit_data["predicted_nqi"] = mit_train_nqi_predictions
mit_train_final_df = train_mit_data.groupby("ID")["Parkinsons", "predicted_nqi"].mean().reset_index()

# predict nqi for every subject in the test-set (as mean of predicted nqi of all the subject's time-windows):
mit_test_nqi_predictions = ensemble.predict(test_mit_X)
test_mit_data["predicted_nqi"] = mit_test_nqi_predictions
mit_test_final_df = test_mit_data.groupby("ID")["Parkinsons", "predicted_nqi"].mean().reset_index()

# We now use the nqi score predicted by the regression ensemble t in order to predict Parkinsons:
clf = LogisticRegression()


clf.fit(mit_train_final_df["predicted_nqi"].values.reshape(-1, 1), mit_train_final_df["Parkinsons"])

print(mit_test_final_df.groupby("Parkinsons")["predicted_nqi"].describe())

predicted_test_probs = clf.predict_proba(mit_test_final_df["predicted_nqi"].values.reshape(-1, 1))[:,1]
mit_test_set_auc = roc_auc_score(y_true=mit_test_final_df["Parkinsons"], y_score=predicted_test_probs)
print("AUC on test set: ", mit_test_set_auc)
np.mean(cross_val_score(clf,
                        X =mit_test_final_df["predicted_nqi"].values.reshape(-1, 1),
                        y=mit_test_final_df["Parkinsons"],
                        scoring="roc_auc",
                        cv=10))

# Sanity check with a dummy-classifier:
dummy_clf = DummyClassifier(strategy='stratified')
dummy_clf.fit(mit_train_final_df["predicted_nqi"].values.reshape(-1, 1), mit_train_final_df["Parkinsons"])
dummy_test_predicted_probs = dummy_clf.predict_proba(mit_test_final_df["predicted_nqi"].values.reshape(-1, 1))[:,1]
dummy_auc = roc_auc_score(y_true=mit_test_final_df["Parkinsons"], y_score=dummy_test_predicted_probs)