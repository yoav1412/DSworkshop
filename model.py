import pandas as pd
import numpy as np
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

# The summary statistics used in the original article:
ARTICLE_EXPLANATORY_VARIABLES =['L_HoldTime_mean', 'L_HoldTime_std', 'L_HoldTime_kurtosis', 'L_HoldTime_skew',
 'R_HoldTime_mean', 'R_HoldTime_std', 'R_HoldTime_kurtosis', 'R_HoldTime_skew', 'LL_LatencyTime_mean', 'LL_LatencyTime_std',
 'LL_LatencyTime_kurtosis', 'LL_LatencyTime_skew', 'LR_LatencyTime_mean', 'LR_LatencyTime_std', 'LR_LatencyTime_kurtosis',
 'LR_LatencyTime_skew', 'RL_LatencyTime_mean', 'RL_LatencyTime_std', 'RL_LatencyTime_kurtosis', 'RL_LatencyTime_skew',
 'RR_LatencyTime_mean', 'RR_LatencyTime_std', 'RR_LatencyTime_kurtosis', 'RR_LatencyTime_skew', 'mean_diff_LR_RL_LatencyTime',
 'mean_diff_LL_RR_LatencyTime', 'mean_diff_L_R_HoldTime']


def evaluate_classifier(clf, X, y, cross_validation_folds=10, round_result_to=4):
    """
    :param clf: a classifier that inherrits from sklearn's BaseClassifier
    :param X: pandas df of explanatory variables
    :param y: target column
    :param cross_validation_folds: number of folds for k-fold cross-validation
    :return: prints train accuracy on the entire dataset, and the test accuracy calculated with k-fold cross validation
    """
    cv_gen = KFold(n_splits=10)
    test_accuracy = np.mean(cross_val_score(estimator=clf, X=X, y=y, cv=cv_gen))
    clf.fit(X, y)
    train_accuracy = clf.score(X, y)
    res = namedtuple("accuracy", "test train test_score train_score")
    res.test = "Test accuracy ({}-fold cross validation):".format(cross_validation_folds)+str(round(test_accuracy, round_result_to))
    res.train = "Train accuracy:"+str(round(train_accuracy, round_result_to))
    res.test_score = test_accuracy
    res.train_score = train_accuracy
    return res


data = pd.read_csv(constants.DATA_FOLDER + r"\\final.csv")


# Clean the data according to criteria stated in the article:
data = data[data.total_count >= 2000]  # take only users with more than 2000 keystrokes
data = data[data.Levadopa == False]
data = data[data.Parkinsons == False | ( (data.Parkinsons == True) & (data.Impact == "Mild"))]

# TODO: there is one user missing - in the article there were 53, we have only 52. need to verify why

########################################################
###################### TRY 1 ###########################
########################################################
# applying several classifiers to the raw data with the variables used in the article, without further processing:
X = data[ARTICLE_EXPLANATORY_VARIABLES]
y = data["Parkinsons"]
classifiers = [LogisticRegression(),
               RandomForestClassifier(),
               AdaBoostClassifier(),
               KNeighborsClassifier(),
               GradientBoostingClassifier(),
               SVC(kernel='rbf'),
               ]
# for clf in classifiers:
#     accuracy = evaluate_classifier(clf, X, y, cross_validation_folds=10)
#     print(str(clf).split("(")[0]+":")
#     print("\t"+accuracy.train)
#     print("\t" + accuracy.test)
# # TODO: comment on results (models seem very overfitted, performance is basically random, etc.)
#
# ########################################################
# ###################### TRY 2 ###########################
# ########################################################
# # We will now try to normalize the data, and cast it to a lower dimension with PCA.
# # For every dimension in (1,...,original_dim), we run all models and look for the best test score among those models:
#
# def plot_labeled_data_1d(reduced_X, y, title, group_labels =("diagnosed", "not_diagnosed")):
#     pdt = [reduced_X[i] for i in range(len(reduced_X)) if y.values[i] == True]
#     pdf = [reduced_X[i] for i in range(len(reduced_X)) if y.values[i] == False]
#     a = plt.scatter(pdf, [0 for i in range(len(pdf))])
#     b = plt.scatter(pdt, [0 for i in range(len(pdt))], color="red")
#     plt.title(title)
#     plt.legend([b, a], group_labels)
#     plt.show()
#
# def plot_labeled_data_2d(reduced_X, y, title, group_labels =("diagnosed", "not_diagnosed")):
#     x1_pd_true = [reduced_X[i][0] for i in range(len(y)) if y.values[i] == True]
#     x1_pd_false = [reduced_X[i][0] for i in range(len(y)) if y.values[i] == False]
#     x2_pd_true = [reduced_X[i][1] for i in range(len(y)) if y.values[i] == True]
#     x2_pd_false = [reduced_X[i][1] for i in range(len(y)) if y.values[i] == False]
#     b = plt.scatter(x1_pd_true, x2_pd_true, color='red')
#     a = plt.scatter(x1_pd_false, x2_pd_false, color='blue')
#     plt.title(title)
#     plt.legend([b, a], group_labels)
#     plt.show()
#
# def plot_labeled_data_3d(reduced_X, y, title, group_labels =("diagnosed", "not_diagnosed")):
#     x1_pd_true = [reduced_X[i][0] for i in range(len(y)) if y.values[i] == True]
#     x1_pd_false = [reduced_X[i][0] for i in range(len(y)) if y.values[i] == False]
#     x2_pd_true = [reduced_X[i][1] for i in range(len(y)) if y.values[i] == True]
#     x2_pd_false = [reduced_X[i][1] for i in range(len(y)) if y.values[i] == False]
#     x3_pd_true = [reduced_X[i][2] for i in range(len(y)) if y.values[i] == True]
#     x3_pd_false = [reduced_X[i][2] for i in range(len(y)) if y.values[i] == False]
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     a = ax.scatter(x1_pd_false, x2_pd_false, x3_pd_false, c='blue')
#     b = ax.scatter(x1_pd_true, x2_pd_true, x3_pd_true, c='red')
#     plt.title(title)
#     plt.legend([b, a], group_labels)
#     plt.show()
#
# scaler = StandardScaler()
# normalized_X = scaler.fit_transform(X)
# original_dim = len(normalized_X[0])
# tested_dimensions = []
# accuracies = []
# for dim in [i for i in range(1, original_dim+1)]:
#     if dim != original_dim:
#         pca = PCA(n_components=dim)
#         reduced_X = pca.fit_transform(normalized_X, y.values)
#     else:
#         reduced_X = normalized_X
#     best_accuracy = namedtuple("best_accuracy", "clf_name test_accuracy train_accuracy")
#     best_accuracy.test_accuracy = -1  #init
#     for clf in classifiers:
#         accuracy = evaluate_classifier(clf, reduced_X, y, cross_validation_folds=10)
#         if accuracy.test_score > best_accuracy.test_accuracy:
#             best_accuracy.test_accuracy = accuracy.test_score
#             best_accuracy.clf_name = str(clf).split("(")[0]
#     tested_dimensions.append(dim)
#     accuracies.append(best_accuracy.test_accuracy)
# plt.title("Best test accuracy by dimension (PCA)")
# plt.xlabel("Dimension")
# plt.ylabel("Test accuracy rate")
# plt.plot(tested_dimensions, accuracies) #TODO: add explanations to plot.
# # We can see that dimensionality reduction did not help.
#
#
# # Visualizing the data in 3d, 2d and 1d after PCA:
# pca = PCA(n_components=3)
# reduced_X = pca.fit_transform(normalized_X)
# plot_labeled_data_3d(reduced_X, y, "Casting to 3D using Principal Component Analysis")
#
#
# pca = PCA(n_components=2)
# reduced_X = pca.fit_transform(normalized_X)
# plot_labeled_data_2d(reduced_X, y, "Casting to 2D using Principal Component Analysis")
#
#
# pca = PCA(n_components=1)
# reduced_X = pca.fit_transform(normalized_X)
# plot_labeled_data_1d(reduced_X, y, "Casting to 1D using Principal Component Analysis")
#

########################################################
###################### TRY 3 ###########################
########################################################
# We will now apply normalizaion and reduction to 1d wth LDA, in accordance with the original article:
lda = LinearDiscriminantAnalysis()
scaler = StandardScaler()
normalized_X = scaler.fit_transform(X)

indices = [i for i in range(52)]
tr_indices = [i for i in range(30)]  # np.random.choice(indices,size=46, replace=False)
ts_indices = [i for i in indices if i not in tr_indices]  # [i for i in range(31,52)]#list(e for e in indices if e not in tr_indices)

trainX = X.iloc[tr_indices]
trainX = StandardScaler().fit_transform(trainX)
trainy = y.iloc[tr_indices]
testX = X.iloc[ts_indices]
testX = StandardScaler().fit_transform(testX)
testy = y.iloc[ts_indices]

trainX = lda.fit_transform(trainX, trainy)
testX = lda.transform(testy)

reduced_X =lda.fit_transform(normalized_X, y)
#plot_labeled_data_1d(reduced_X, y, "Casting to 1D using Linear Discriminant Analysis")

best_accuracy = namedtuple("best_accuracy", "clf_name test_accuracy train_accuracy")
best_accuracy.test_accuracy = -1  #init
for clf in classifiers:
    accuracy = evaluate_classifier(clf, reduced_X, y, cross_validation_folds=10)
    clf_name = str(clf).split("(")[0]
    print(clf_name + ":")
    print("\t"+accuracy.train)
    print("\t" + accuracy.test)
    if accuracy.test_score > best_accuracy.test_accuracy:
        best_accuracy.test_accuracy = accuracy.test_score
        best_accuracy.clf_name = clf_name
#
# # We will try to optimize the learning-parameters of the one of the more complex models:
# clf = GradientBoostingClassifier()
# param_grid = {
# "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8],
# "n_estimators":[50, 100, 200, 400, 800],
# "max_depth":[1, 2, 4, 6, 8, 10]
# }
# grid_searcher = GridSearchCV(clf, param_grid, n_jobs=cpu_count()-1, cv=10)
# grid_searcher.fit(reduced_X, y)
# print("best score: ", grid_searcher.best_score_, "\nbest params: ", grid_searcher.best_params_)
#
#
# # We will now tr to optimize K for the KNN model:
# param_grid = {'n_neighbors':[k for k in range(1, 35+1)]}
# grid_searcher = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=cpu_count()-1, cv=10)
# grid_searcher.fit(reduced_X, y)
# k_values = dict(grid_searcher.cv_results_)["param_n_neighbors"].tolist()
# test_accuracy = dict(grid_searcher.cv_results_)["mean_test_score"].tolist()
# plt.plot(k_values, test_accuracy)
# plt.title("kNN test accuracy by num. of neighbors")
# plt.xlabel("k")
# plt.ylabel("Test accuracy")
# plt.show()
#
#
# #TODO: add an explanation here, noting that the best classifier is KNN which has low train accuracy. seems like the other more complex models are verfitting, therefore we can now try to divide the data into two groups like in the article (to reduce num of X vars and prevent overfitting)
#
# ########################################################
# ###################### TRY 4 ###########################
# ########################################################
# # We will now try to add some additional explanatory variables that were not used in the original article:
# # We add percentiles summary statistics and entropy:
# PERCENTILES = [c for c in data.columns.values if "perc" in c and "FlightTime" not in c]
# ENTROPY = [c for c in data.columns.values if "entropy" in c and "FlightTime" not in c]
# X = data[ARTICLE_EXPLANATORY_VARIABLES + PERCENTILES + ENTROPY]
# y = data["Parkinsons"]
#
# normalized_X = scaler.fit_transform(X)
# reduced_X =lda.fit_transform(normalized_X, y)
#
# best_accuracy = namedtuple("best_accuracy", "clf_name test_accuracy train_accuracy")
# best_accuracy.test_accuracy = -1  #init
# for clf in classifiers:
#     accuracy = evaluate_classifier(clf, reduced_X, y, cross_validation_folds=10)
#     clf_name = str(clf).split("(")[0]
#     print(clf_name + ":")
#     print("\t"+accuracy.train)
#     print("\t" + accuracy.test)
#     if accuracy.test_score > best_accuracy.test_accuracy:
#         best_accuracy.test_accuracy = accuracy.test_score
#         best_accuracy.clf_name = clf_name
# # We can see that with the new variables a 90+ % accuracy can be reached.
#
# # We will now try to tune the parameters of the bet classifier:
# clf = GradientBoostingClassifier()
# param_grid = {
# "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8],
# "n_estimators":[50, 100, 200, 400, 800],
# "max_depth":[1, 2, 4, 6, 8, 10]
# }
# grid_searcher = GridSearchCV(clf, param_grid, n_jobs=cpu_count()-1, cv=10, verbose=5)


########################################################
###################### TRY 5 ###########################
########################################################

# We will now try to split the explanatory variable into two groups, Hold variables and Latency variables (depending
# on the orignal column they were created by). We will then assign a probability for each group seperately, and output
# a final probability as a weighted average of these two probabilities.

ALL_VARIABLES = ARTICLE_EXPLANATORY_VARIABLES #+ PERCENTILES + ENTROPY
HOLD_VARIABLES = [v for v in ALL_VARIABLES if "HoldTime" in v]
LATENCY_VARIABLES = [v for v in ALL_VARIABLES if "LatencyTime" in v]





from sklearn.pipeline import Pipeline
from TwoGroupsWeightedModel import TwoGroupsWeightedModel


vclf = VotingClassifier(estimators=[#('0',GradientBoostingClassifier()),
                                    #('0.5',AdaBoostClassifier()),
                                    ('1',RandomForestClassifier()),
                                    ('2',SVC(probability=True)),
                                     ('3',MLPClassifier()),
                                    ('4',LogisticRegression()),
                                    ('5',NuSVC(probability=True)),
                                    ('6',KNeighborsClassifier()),
                                    ('7',DecisionTreeClassifier()),
                                    ('8',QuadraticDiscriminantAnalysis())
                                  ],
                        voting='soft')

def weighting_function(hold_probs, latency_probs):
    """TODO: doc. maybe use this to implement the weihgting func in the article, which is not good bcz it yields probs > 1
    :param hold_probs:
    :param latency_probs:
    :return:
    """
    return (hold_probs + 0.5*(1-1.2) + 1.2*latency_probs)/2.0


tg = TwoGroupsWeightedModel(underlying_estimator_module_and_class="sklearn.pipeline Pipeline",
                            transformer_module_and_class=None,#"sklearn.discriminant_analysis LinearDiscriminantAnalysis",
                            group1_var_names=HOLD_VARIABLES,
                            group2_var_names=LATENCY_VARIABLES,
                            #weighting_function=weighting_function,
                            group1_weight=0.5, group2_weight=0.5, classification_threshold=0.5,
                            steps=[("normalization", MinMaxScaler()),
                                   ("lda", LinearDiscriminantAnalysis()),
                                 # ("pcs", PCA(n_components=7)),
                                   ("clf", vclf)])



# tr_indices = [i for i in range(30)]  # np.random.choice(indices,size=46, replace=False)
# ts_indices = [32, 33]  # [i for i in range(31,52)]#list(e for e in indices if e not in tr_indices)
#
# tg.fit(X.iloc[tr_indices],y.iloc[tr_indices])
# probs = tg.predict_proba(X.iloc[ts_indices])

print(np.mean(cross_val_score(tg, X, y, cv=10)))

print("h")

raise ValueError


## sanity check:

hold =X[HOLD_VARIABLES]
latency = X[LATENCY_VARIABLES]

NFOLDS = 1
accuracies = []

for i in range(NFOLDS):
    indices = [i for i in range(52)]
    tr_indices = np.random.choice(indices, size=40)  # np.random.choice(indices,size=46, replace=False)
    ts_indices = [i for i in indices if i not in tr_indices]  # [i for i in range(31,52)]#list(e for e in indices if e not in tr_indices)
    print("tr:",tr_indices)
    print("ts:",ts_indices)

    y_train = y.iloc[tr_indices]
    y_test = y.iloc[ts_indices]

    h_train = hold.iloc[tr_indices]
    h_test = hold.iloc[ts_indices]

    l_train = latency.iloc[tr_indices]
    l_test = latency.iloc[ts_indices]





    # h_train = StandardScaler().fit_transform(h_train)
    # h_lda = LinearDiscriminantAnalysis()
    # h_train = h_lda.fit_transform(h_train, y_train)
    # h_test = StandardScaler().fit_transform(h_test)
    # h_test = h_lda.transform(h_test)
    #
    # l_train = StandardScaler().fit_transform(l_train)
    # l_lda = LinearDiscriminantAnalysis()
    # l_train =l_lda.fit_transform(l_train, y_train)
    # l_test = StandardScaler().fit_transform(l_test)
    # l_test =l_lda.transform(l_test)


    h_clf = Pipeline(steps=[("normalization", StandardScaler()),
                                   ("lda", LinearDiscriminantAnalysis()),
                                   ("clf", GradientBoostingClassifier())])
    h_clf.fit(h_train, y_train)
    h_probs = h_clf.predict_proba(h_test)

    l_clf = Pipeline(steps=[("normalization", StandardScaler()),
                                   ("lda", LinearDiscriminantAnalysis()),
                                   ("clf", GradientBoostingClassifier())])
    l_clf.fit(l_train, y_train)
    l_probs = l_clf.predict_proba(l_test)

    tot_probs = 0.5*(h_probs + l_probs)
    print(tot_probs[:,1])
    # tot_preds = tot_probs > 0.5
    # tot_preds = tot_preds[:,1]
    # accuracy = sum(tot_preds == y_test) / float(len(y_test))
    # accuracies.append(accuracy)


print("g")