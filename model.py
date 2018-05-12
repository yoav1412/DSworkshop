import pandas as pd
import numpy as np
from sklearn.svm import SVC

import constants
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import namedtuple
from sklearn.decomposition import PCA
from os import cpu_count
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier


# The summary statistics used in the original article:
ARTICLE_EXPLANATORY_VARIABLES =['L_HoldTime_mean', 'L_HoldTime_std', 'L_HoldTime_kurtosis', 'L_HoldTime_skew',
 'R_HoldTime_mean', 'R_HoldTime_std', 'R_HoldTime_kurtosis', 'R_HoldTime_skew', 'LL_LatencyTime_mean', 'LL_LatencyTime_std',
 'LL_LatencyTime_kurtosis', 'LL_LatencyTime_skew', 'LR_LatencyTime_mean', 'LR_LatencyTime_std', 'LR_LatencyTime_kurtosis',
 'LR_LatencyTime_skew', 'RL_LatencyTime_mean', 'RL_LatencyTime_std', 'RL_LatencyTime_kurtosis', 'RL_LatencyTime_skew',
 'RR_LatencyTime_mean', 'RR_LatencyTime_std', 'RR_LatencyTime_kurtosis', 'RR_LatencyTime_skew', 'mean_diff_LR_RL_LatencyTime',
 'mean_diff_LL_RR_LatencyTime']
HOLD_VARIABLES = [var for var in ARTICLE_EXPLANATORY_VARIABLES if "Hold" in var]


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
for clf in classifiers:
    accuracy = evaluate_classifier(clf, X, y, cross_validation_folds=10)
    print(str(clf).split("(")[0]+":")
    print("\t"+accuracy.train)
    print("\t" + accuracy.test)
# TODO: comment on results (models seem very overfitted, performance is basically random, etc.)

########################################################
###################### TRY 2 ###########################
########################################################
# We will now try to normalize the data, and cast it to a lower dimension with PCA.
# For every dimension in (1,...,original_dim), we run all models and look for the best test score among those models:

def plot_labeled_data_1d(reduced_X, y, title, group_labels =("diagnosed", "not_diagnosed")):
    pdt = [reduced_X[i] for i in range(len(reduced_X)) if y.values[i] == True]
    pdf = [reduced_X[i] for i in range(len(reduced_X)) if y.values[i] == False]
    a = plt.scatter(pdf, [0 for i in range(len(pdf))])
    b = plt.scatter(pdt, [0 for i in range(len(pdt))], color="red")
    plt.title(title)
    plt.legend([b, a], group_labels)
    plt.show()

def plot_labeled_data_2d(reduced_X, y, title, group_labels =("diagnosed", "not_diagnosed")):
    x1_pd_true = [reduced_X[i][0] for i in range(len(y)) if y.values[i] == True]
    x1_pd_false = [reduced_X[i][0] for i in range(len(y)) if y.values[i] == False]
    x2_pd_true = [reduced_X[i][1] for i in range(len(y)) if y.values[i] == True]
    x2_pd_false = [reduced_X[i][1] for i in range(len(y)) if y.values[i] == False]
    b = plt.scatter(x1_pd_true, x2_pd_true, color='red')
    a = plt.scatter(x1_pd_false, x2_pd_false, color='blue')
    plt.title(title)
    plt.legend([b, a], group_labels)
    plt.show()

def plot_labeled_data_3d(reduced_X, y, title, group_labels =("diagnosed", "not_diagnosed")):
    x1_pd_true = [reduced_X[i][0] for i in range(len(y)) if y.values[i] == True]
    x1_pd_false = [reduced_X[i][0] for i in range(len(y)) if y.values[i] == False]
    x2_pd_true = [reduced_X[i][1] for i in range(len(y)) if y.values[i] == True]
    x2_pd_false = [reduced_X[i][1] for i in range(len(y)) if y.values[i] == False]
    x3_pd_true = [reduced_X[i][2] for i in range(len(y)) if y.values[i] == True]
    x3_pd_false = [reduced_X[i][2] for i in range(len(y)) if y.values[i] == False]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    a = ax.scatter(x1_pd_false, x2_pd_false, x3_pd_false, c='blue')
    b = ax.scatter(x1_pd_true, x2_pd_true, x3_pd_true, c='red')
    plt.title(title)
    plt.legend([b, a], group_labels)
    plt.show()

scaler = StandardScaler()
normalized_X = scaler.fit_transform(X)
original_dim = len(normalized_X[0])
tested_dimensions = []
accuracies = []
for dim in [i for i in range(1, original_dim+1)]:
    if dim != original_dim:
        pca = PCA(n_components=dim)
        reduced_X = pca.fit_transform(normalized_X, y.values)
    else:
        reduced_X = normalized_X
    best_accuracy = namedtuple("best_accuracy", "clf_name test_accuracy train_accuracy")
    best_accuracy.test_accuracy = -1  #init
    for clf in classifiers:
        accuracy = evaluate_classifier(clf, reduced_X, y, cross_validation_folds=10)
        if accuracy.test_score > best_accuracy.test_accuracy:
            best_accuracy.test_accuracy = accuracy.test_score
            best_accuracy.clf_name = str(clf).split("(")[0]
    tested_dimensions.append(dim)
    accuracies.append(best_accuracy.test_accuracy)
plt.title("Best test accuracy by dimension (PCA)")
plt.xlabel("Dimension")
plt.ylabel("Test accuracy rate")
plt.plot(tested_dimensions, accuracies) #TODO: add explanations to plot.
# We can see that dimensionality reduction did not help.


# Visualizing the data in 3d, 2d and 1d after PCA:
pca = PCA(n_components=3)
reduced_X = pca.fit_transform(normalized_X)
plot_labeled_data_3d(reduced_X, y, "Casting to 3D using Principal Component Analysis")


pca = PCA(n_components=2)
reduced_X = pca.fit_transform(normalized_X)
plot_labeled_data_2d(reduced_X, y, "Casting to 2D using Principal Component Analysis")


pca = PCA(n_components=1)
reduced_X = pca.fit_transform(normalized_X)
plot_labeled_data_1d(reduced_X, y, "Casting to 1D using Principal Component Analysis")


########################################################
###################### TRY 3 ###########################
########################################################
# We will now apply normalizaion and reduction to 1d wth LDA, in accordance with the original article:
lda = LinearDiscriminantAnalysis()
reduced_X =lda.fit_transform(normalized_X, y)
plot_labeled_data_1d(reduced_X, y, "Casting to 1D using Linear Discriminant Analysis")

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

# We will try to optimize the learning-parameters of the one of the more complex models:
clf = GradientBoostingClassifier()
param_grid = {
"learning_rate": [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8],
"n_estimators":[50, 100, 200, 400, 800],
"max_depth":[1, 2, 4, 6, 8, 10]
}
grid_searcher = GridSearchCV(clf, param_grid, n_jobs=cpu_count()-1, cv=10)
grid_searcher.fit(reduced_X, y)
print("best score: ", grid_searcher.best_score_, "\nbest params: ", grid_searcher.best_params_)


# We will now tr to optimize K for the KNN model:
param_grid = {'n_neighbors':[k for k in range(1, 35+1)]}
grid_searcher = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=cpu_count()-1, cv=10)
grid_searcher.fit(reduced_X, y)
k_values = dict(grid_searcher.cv_results_)["param_n_neighbors"].tolist()
test_accuracy = dict(grid_searcher.cv_results_)["mean_test_score"].tolist()
plt.plot(k_values, test_accuracy)
plt.title("kNN test accuracy by num. of neighbors")
plt.xlabel("k")
plt.ylabel("Test accuracy")
plt.show()


#TODO: add an explanation here, noting that the best classifier is KNN which has low train accuracy. seems like the other more complex models are verfitting, therefore we can now try to divide the data into two groups like in the article (to reduce num of X vars and prevent overfitting)

########################################################
###################### TRY 4 ###########################
########################################################
# We will now try to add some additional explanatory variables that were not used in the original article:
# We add percentiles summary statistics and entropy:
PERCENTILES = [c for c in data.columns.values if "perc" in c and "FlightTime" not in c]
ENTROPY = [c for c in data.columns.values if "entropy" in c and "FlightTime" not in c]
X = data[ARTICLE_EXPLANATORY_VARIABLES + PERCENTILES + ENTROPY]
y = data["Parkinsons"]

normalized_X = scaler.fit_transform(X)
reduced_X =lda.fit_transform(normalized_X, y)

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
# We can see that with the new variables a 90+ % accuracy can be reached.

# We will now try to tune the parameters of the bet classifier:
clf = GradientBoostingClassifier()
param_grid = {
"learning_rate": [0.001,0.01,.05,0.1,0.3,0.5,0.8],
"n_estimators":[50, 100, 200, 400, 800],
"max_depth":[1,2,4,6,8,10]
}
grid_searcher = GridSearchCV(clf, param_grid, n_jobs=cpu_count()-1, cv=10, verbose=5)


########################################################
###################### TRY 5 ###########################
########################################################

# We will now try to split the explanatory variable into two groups, Hold variables and Latency variables (depending
# on the orignal column they were created by). We will then assign a probability for each group seperately, and output
# a final probability as a weighted average of these two probabilities.

ALL_VARIABLES = ARTICLE_EXPLANATORY_VARIABLES + PERCENTILES + ENTROPY
HOLD_VARIABLES = [v for v in ALL_VARIABLES if "HoldTime" in v]
LATENCY_VARIABLES = [v for v in ALL_VARIABLES if "LatencyTime" in v]

hold_x = data[HOLD_VARIABLES]
latency_x = data[LATENCY_VARIABLES]

class TwoGroupsWeightedModel(BaseEstimator):
    def __init__(self, underlying_estimator_f, group1_var_names, group2_var_names, **kwargs):
        self.group1_var_names = group1_var_names
        self.group2_var_names = group2_var_names
        self.underlying_estimator_f = underlying_estimator_f
        self.underlying_estimator_params_dict = {k:v for k,v in kwargs.items()}
        self.underlying_estimator = self.underlying_estimator_f(**kwargs)

    def fit(self, X, y):
        group1_X = X[self.group1_var_names]
        group2_X = X[self.group2_var_names]

        self.group1_estimator = self.underlying_estimator.fit(group1_X, y)
        print("fit done")
        print("train score:")
        print(self.group1_estimator.score(group1_X,y))

    def score(self, X, y):
        return self.group1_estimator.score(X[self.group1_var_names], y)

X = data[ARTICLE_EXPLANATORY_VARIABLES]
y = data["Parkinsons"]
tg = TwoGroupsWeightedModel(KNeighborsClassifier, ["L_HoldTime_mean"], ['L_HoldTime_kurtosis'], n_neighbors=6)
param_grid = {'underlying_estimator':[KNeighborsClassifier],
              'group1_var_names': [["L_HoldTime_mean"]],
              'group2_var_names':[['L_HoldTime_kurtosis']],
              "n_neighbors" : [2,6]}
evaluate_classifier(tg, X, y)
