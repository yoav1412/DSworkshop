from collections import namedtuple
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler


def get_best_roc(models, train_X, train_y, test_X, test_y):
    """
    :param models: a collection of sklearn estimators
    :param train_X: training features
    :param train_y: training target
    :param test_X:  test features
    :param test_y:  test targer
    :return: fits all models in the collection to train data, tests them on the test data with respect to AUC metric,
                and returns the best AUC score and the best estimator.
    """
    best_score = -1 * float('inf')
    for model in models:
        model.fit(train_X, np.ravel(train_y))
        predicted_probs = model.predict_proba(test_X)[:, 1]
        scr = roc_auc_score(y_true=test_y, y_score=predicted_probs)
        if scr > best_score:
            best_score = scr
            best_model = model
    return best_score, best_model


def evaluate_classifier_cv(clf, X, y, cross_validation_folds=10, random_split=True, round_result_to=4,
                           scoring='accuracy'):
    """
    :param clf: a classifier that inherrits from sklearn's BaseClassifier
    :param X: pandas df of explanatory variables
    :param y: target column
    :param cross_validation_folds: number of folds for k-fold cross-validation
    :return: prints train accuracy on the entire dataset, and the test accuracy calculated with k-fold cross validation
    """
    cv_gen = KFold(n_splits=cross_validation_folds, shuffle=random_split)
    test_accuracy = np.mean(cross_val_score(estimator=clf, X=X, y=y, cv=cv_gen, scoring=scoring))
    clf.fit(X, y)
    train_accuracy = clf.score(X, y)
    res = namedtuple("accuracy", "test train test_score train_score")
    res.test = "Test accuracy ({}-fold cross validation):".format(cross_validation_folds) + str(
        round(test_accuracy, round_result_to))
    res.train = "Train accuracy:" + str(round(train_accuracy, round_result_to))
    res.test_score = test_accuracy
    res.train_score = train_accuracy
    return res


def split_to_train_test_and_apply_scaling_and_lda_dim_reduction(X, y, train_percentage):
    """
    :param train_percentage: what percentage of the data will be used for train (the rest - for test)
    :return: retunrs train and test sets, after applying sklearn's standard-scalar and LDA dimensionality reduction.

    This function is used to explicitly avoid using sklearn's LDA API for dim-reduction, which we find confusing and
    believe is the source for the error in the original article.
    """
    lda = LinearDiscriminantAnalysis()
    scaler = StandardScaler()
    indices = [i for i in range(len(y))]
    train_indices = np.random.choice(indices, size=int(train_percentage * len(y)), replace=False)
    test_indices = [i for i in indices if i not in train_indices]

    train_X = scaler.fit_transform(X.iloc[train_indices])
    train_y = np.ravel(y.iloc[train_indices].values)

    test_X = scaler.transform(X.iloc[test_indices])
    test_y = np.ravel(y.iloc[test_indices].values)

    train_X = lda.fit_transform(train_X, train_y)
    test_X = lda.transform(test_X)

    return train_X, train_y, test_X, test_y


def get_labeled_data_1d(reduced_X, y):
    """
    :param reduced_X: feature matrix reduced to 1D by some dimensionality reduction technique
    :param y: class labels for the data
    :return: returns a tuple of the 'true'/'false' labels and their respective values. The result can be directly
            plotted to view the labeled data with partition to classes.
    """
    pdt = [reduced_X[i] for i in range(len(reduced_X)) if y.values[i] == True]
    pdf = [reduced_X[i] for i in range(len(reduced_X)) if y.values[i] == False]
    return pdf, [0 for i in range(len(pdf))], pdt, [0 for i in range(len(pdt))]


def get_labeled_data_2d(reduced_X, y):
    """
    :param reduced_X: feature matrix reduced to 2D by some dimensionality reduction technique
    :param y: class labels for the data
    :return: returns a tuple of the 'true'/'false' labels and their respective values. The result can be directly
            plotted to view the labeled data with partition to classes.
    """
    x1_pd_true = [reduced_X[i][0] for i in range(len(y)) if y.values[i] == True]
    x1_pd_false = [reduced_X[i][0] for i in range(len(y)) if y.values[i] == False]
    x2_pd_true = [reduced_X[i][1] for i in range(len(y)) if y.values[i] == True]
    x2_pd_false = [reduced_X[i][1] for i in range(len(y)) if y.values[i] == False]
    return x1_pd_true, x2_pd_true, x1_pd_false, x2_pd_false


def get_labeled_data_3d(reduced_X, y):
    """
    :param reduced_X: feature matrix reduced to 3D by some dimensionality reduction technique
    :param y: class labels for the data
    :return: returns a tuple of the 'true'/'false' labels and their respective values. The result can be directly
            plotted in 3D to view the labeled data with partition to classes.
    """
    x1_pd_true = [reduced_X[i][0] for i in range(len(y)) if y.values[i] == True]
    x1_pd_false = [reduced_X[i][0] for i in range(len(y)) if y.values[i] == False]
    x2_pd_true = [reduced_X[i][1] for i in range(len(y)) if y.values[i] == True]
    x2_pd_false = [reduced_X[i][1] for i in range(len(y)) if y.values[i] == False]
    x3_pd_true = [reduced_X[i][2] for i in range(len(y)) if y.values[i] == True]
    x3_pd_false = [reduced_X[i][2] for i in range(len(y)) if y.values[i] == False]
    return x1_pd_false, x2_pd_false, x3_pd_false, x1_pd_true, x2_pd_true, x3_pd_true


def plot_dimensionality_reduction(_1d_res, _2d_res, _3d_res):
    """
    :return: given the labeled data in 1\2\3 dimensions, plots the data in the relevant space with class labels.
    """
    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(1, 3, 1, title="Casting to 1D using PCA")
    ax1.scatter(_1d_res[0], _1d_res[1], color='red')
    ax1.scatter(_1d_res[2], _1d_res[3], color='blue')
    ax1.legend(("diagnosed", "not_diagnosed"))

    ax2 = fig.add_subplot(1, 3, 2, title="Casting to 2D using PCA")
    ax2.scatter(_2d_res[0], _2d_res[1], color='red')
    ax2.scatter(_2d_res[2], _2d_res[3], color='blue')
    ax2.legend(("diagnosed", "not_diagnosed"))

    ax3 = fig.add_subplot(1, 3, 3, projection='3d', title="Casting to 3D using PCA")
    ax3.title.set_position([0.5, 1])
    ax3.scatter(_3d_res[0], _3d_res[1], _3d_res[2], color='red')
    ax3.scatter(_3d_res[3], _3d_res[4], _3d_res[5], color='blue')
    ax3.legend(("diagnosed", "not_diagnosed"))


def list_diff(first, second):
    """
    :return: all items in first list that are not in second list.
    """
    second = set(second)
    return [item for item in first if item not in second]


def plot_multiple_roc_curves(data, title):
    """
    :param data: a list of (a,b,c) tuples, where 'a' is the legend text for one roc line, 'b' is the true y values,
                    and 'c' is the predicted y values.
    :param title: the title for the whole plot
    :return: plot multiple ROC curves on one canvas according to the data
    """
    import matplotlib.pyplot as plt
    plt.title(title)

    legends = []
    for legend, y_true, y_score in data:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr)
        legends.append(legend)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(legends, loc='lower right')
    plt.show()
