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
    res = namedtuple("accuracy", "test train")
    res.test = "Test accuracy ({}-fold cross validation):".format(cross_validation_folds)+str(round(test_accuracy, round_result_to))
    res.train = "Train accuracy:"+str(round(train_accuracy, round_result_to))
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
# TODO: comment on results (models semm very overfitted, performance is basically random, etc.)


pca = PCA(n_components=2)
scaler = StandardScaler()
X= scaler.fit_transform(X)
reduced_X = pca.fit_transform(X)



classifiers = [LogisticRegression(), RandomForestClassifier(), AdaBoostClassifier(), KNeighborsClassifier(), GradientBoostingClassifier()]
for clf in classifiers:
    accuracy = evaluate_classifier(clf, reduced_X, y, cross_validation_folds=10)
    print(str(clf).split("(")[0]+":")
    print("\t"+accuracy.train)
    print("\t" + accuracy.test)

# TODO: we can show following plot that shows that the data do not seem seperated in 2d after PCA
x1_pd_true = [reduced_X[i][0] for i in range(len(y)) if y.values[i]==True]
x1_pd_false = [reduced_X[i][0] for i in range(len(y)) if y.values[i]==False]
x2_pd_true = [reduced_X[i][1] for i in range(len(y)) if y.values[i]==True]
x2_pd_false = [reduced_X[i][1] for i in range(len(y)) if y.values[i]==False]
plt.scatter(x1_pd_true,x2_pd_true, color='red')
plt.scatter(x1_pd_false,x2_pd_false, color='blue')
plt.show()
