import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import namedtuple
from sklearn.decomposition import PCA


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


data = pd.read_csv(os.getcwd()+r"\\Data\\final.csv")


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
classifiers = [LogisticRegression(), RandomForestClassifier(), AdaBoostClassifier(), KNeighborsClassifier(), GradientBoostingClassifier()]
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
plt.plot(tested_dimensions, accuracies) #TODO: add explanations to plot.
# We can see that dimensionality reduction did not help.


# Visualizing the data in 2d and 1d after PCA:
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

#TODO: add an explanation here, noting that the best classifier is KNN which has low train accuracy. seems like the other more complex models are verfitting, therefore we can now try to divide the data into two groups like in the article (to reduce num of X vars and prevent overfitting)

########################################################
###################### TRY 4 ###########################
########################################################
# We will now try to add some additional explanatory variables that were not used in the original article:
# We add percentiles summary statistics and entropy:
PERCENTILES = [c for c in data.columns.values if "perc" in c]
ENTROPY = [c for c in data.columns.values if "entropy" in c]
X = data[ARTICLE_EXPLANATORY_VARIABLES + PERCENTILES + ENTROPY]
y = data["Parkinsons"]

normalized_X = scaler.fit_transform(X)
reduced_X =lda.fit_transform(normalized_X, y)
