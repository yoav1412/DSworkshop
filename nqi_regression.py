# import pandas as pd
# import numpy as np
# import random
#
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPRegressor, MLPClassifier
# from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
# from sklearn.svm import SVR, SVC, NuSVC
# from sklearn.ensemble import BaggingRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostRegressor, \
#     RandomForestRegressor, VotingClassifier, RandomForestClassifier, AdaBoostClassifier
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import cross_val_score, GridSearchCV
# from sklearn.metrics import roc_auc_score
# from sklearn.dummy import DummyClassifier, DummyRegressor
# from sklearn.tree import DecisionTreeClassifier
# from auxiliary_functions import *
# from constants import *
#
#
#
#
# mit_nqi_features = pd.read_csv(MIT_NQI_FEATURES)
# mit_users = pd.read_csv(MIT_USERS_INPUT)
#
# mit_data = mit_nqi_features.merge(mit_users, on="ID").dropna().reset_index().drop("index", axis=1)
#
# PREDICTION_COLUMNS = list_diff(mit_data.columns, ["UPDRS", "Parkinsons", 'binIndex', 'ID', 'count_nonzero'])
#
#
#
# # Split the unified MIT dataset to train and test according to ID's:
# train_mit_data = mit_data[mit_data.ID <= 100].copy()
# test_mit_data = mit_data[mit_data.ID >= 1000].copy()
#
#
#
# def nqi_regression_and_pd_classification(train_data, test_data):
#     scaler = StandardScaler()
#     min_max_scaler = MinMaxScaler(feature_range=(0, 1))
#
#     train_X = train_data[PREDICTION_COLUMNS]
#     train_y = train_data["UPDRS"]
#
#     train_X = pd.DataFrame(scaler.fit_transform(train_X))
#     train_y = min_max_scaler.fit_transform(train_y.values.reshape(-1,1))
#
#     test_X = test_data[PREDICTION_COLUMNS]
#     test_X = pd.DataFrame(scaler.transform(test_X))
#
#     ENSEMBLE_SIZE = 200
#
#     # TODO:
#     """
#     explanation here about how ding exactly whar the article did (including filtering bins with less than
#     30 holdTimes) yelds not so good auc, but still in the CI given in the article. when we don't filter, and use
#     gbtree we get better results
#     """
#     ensemble = BaggingRegressor(base_estimator=SVR(C=0.94, epsilon=0.052),
#                                 n_estimators=ENSEMBLE_SIZE,
#                                 bootstrap=True,
#                                 n_jobs=1)
#     ensemble = GradientBoostingRegressor(n_estimators=1000)
#
#     # Optimize regressor parameters:
#     param_space = {
#         'n_estimators' : [100,500,1000],
#         'loss':['ls', 'lad', 'huber', 'quantile'],
#         'learning_rate':[0.001,0.01,0.1],
#         'max_depth':[1,2,3,5,7,8],
#         #'alpha':[0.3,0.5,0.9],
#         #'max_leaf_nodes':[None, 3, 5, 10, 20, 50, 100]
#     }
#
#     #grid_searcher = GridSearchCV(ensemble, param_grid=param_space, n_jobs=-1, cv=3, verbose=3, scoring="r2")
#     #grid_searcher.fit(X=train_mit_X, y=np.ravel(train_mit_y))
#     #ensemble.set_params(**grid_searcher.best_params_)
#
#     ensemble.fit(X=train_X, y=np.ravel(train_y))
#
#     train_nqi_predictions = ensemble.predict(train_X)
#
#     train_data["predicted_nqi"] = train_nqi_predictions
#     train_final_df = train_data.groupby("ID")["Parkinsons", "predicted_nqi"].mean().reset_index()
#
#     # predict nqi for every subject in the test-set (as mean of predicted nqi of all the subject's time-windows):
#     test_nqi_predictions = ensemble.predict(test_X)
#     test_data["predicted_nqi"] = test_nqi_predictions
#     test_final_df = test_data.groupby("ID")["Parkinsons", "predicted_nqi"].mean().reset_index()
#
#     # We now use the nqi score predicted by the regression ensemble t in order to predict Parkinsons:
#     clf = LogisticRegression()
#
#     clf = VotingClassifier(estimators=[('0',GradientBoostingClassifier(n_estimators=1000)),
#                                         ('0.5',AdaBoostClassifier()),
#                                         ('1',RandomForestClassifier(n_estimators=1000)),
#                                         ('2',SVC(probability=True)),
#                                          #('3',MLPClassifier()),
#                                         ('4',LogisticRegression()),
#                                        # ('5',NuSVC(probability=True)),
#                                        # ('6',KNeighborsClassifier()),
#                                       #  ('7',DecisionTreeClassifier()),
#                                         ('8',QuadraticDiscriminantAnalysis())
#                                       ],
#                             voting='soft')
#
#
#     clf.fit(train_final_df["predicted_nqi"].values.reshape(-1, 1), train_final_df["Parkinsons"])
#
#     predicted_test_probs = clf.predict_proba(test_final_df["predicted_nqi"].values.reshape(-1, 1))[:,1]
#     test_set_auc = roc_auc_score(y_true=test_final_df["Parkinsons"], y_score=predicted_test_probs)
#
#
#     # Sanity check with a dummy-classifier:
#     dummy_clf = DummyClassifier(strategy='stratified')
#     dummy_clf.fit(train_final_df["predicted_nqi"].values.reshape(-1, 1), train_final_df["Parkinsons"])
#     dummy_test_predicted_probs = dummy_clf.predict_proba(test_final_df["predicted_nqi"].values.reshape(-1, 1))[:,1]
#     dummy_auc = roc_auc_score(y_true=test_final_df["Parkinsons"], y_score=dummy_test_predicted_probs)
#
#     print("AUC on test set: ", test_set_auc," \ndummy AUC (samity check) :", dummy_auc)
#
#     return test_final_df
# nqi_regression_and_pd_classification(train_mit_data, test_mit_data)
#
# # Now we'll train on all of the MIT dataset. and test on the Kaggle dataset:
# # kaggle_nqi_features = pd.read_csv(KAGGLE_NQI_FEATURES)
# # kaggle_users_data = pd.read_csv(KAGGLE_USERS_INPUT)
# # kaggle_data = kaggle_nqi_features.merge(kaggle_users_data, on="ID").dropna().reset_index().drop(["index"], axis=1)
# # # Remove patients taking Levadopa or patients with non-mild Parkinson's:
# # kaggle_data = kaggle_data[kaggle_data.Levadopa == False]
# # kaggle_data = kaggle_data[kaggle_data.Parkinsons == False | ((kaggle_data.Parkinsons == True) & (kaggle_data.Impact == "Mild"))]
# # nqi_regression_and_pd_classification(train_data=mit_data, test_data=kaggle_data)
#
# # However trying to use only the same feature to simply predict PD, gives reasonable results:
# # cv_auc = np.mean(cross_val_score(estimator=GradientBoostingClassifier(),
# #                          X=kaggle_data[PREDICTION_COLUMNS],
# #                          y=kaggle_data["Parkinsons"],
# #                          cv=5,
# #                          scoring="roc_auc",
# #                          n_jobs=1))
# # print(cv_auc)