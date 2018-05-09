from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from model import classifiers, evaluate_classifier, ARTICLE_EXPLANATORY_VARIABLES
import constants
import pandas as pd
from matplotlib import pyplot as plt

# #####################################################################################################
# ############# For each threshold level, get the best accuracy rate and plot the results #############
# #####################################################################################################

thresholds = [100, 200, 500, 1000, 2000, 2500, 3000, 5000]
best_accuracies_for_threshold = []

for threshold in thresholds:
    # filter the data
    data = pd.read_csv(constants.DATA_FOLDER + r"\\final.csv")
    data = data[data.total_count >= threshold]  # take only users with more than 2000 keystrokes
    data = data[data.Levadopa == False]
    data = data[data.Parkinsons == False | ((data.Parkinsons == True) & (data.Impact == "Mild"))]

    X = data[ARTICLE_EXPLANATORY_VARIABLES]
    y = data["Parkinsons"]

    # normalize and apply LDA
    lda = LinearDiscriminantAnalysis()
    scaler = StandardScaler()
    normalized_X = scaler.fit_transform(X)
    X_after_lda = lda.fit_transform(normalized_X, y)

    # get best test accuracy from within the set of classifiers
    accuracies = []
    for clf in classifiers:
        accuracy = evaluate_classifier(clf, X_after_lda, y, cross_validation_folds=10).test_score
        accuracies.append(accuracy)
    best_accuracies_for_threshold.append(max(accuracies))


plt.plot(thresholds, best_accuracies_for_threshold)
plt.title("Best test accuracy by num. of observations threshold (>X)")
plt.xlabel("Threshold")
plt.ylabel("Best accuracy rate")
plt.show()

# Can be easily shown that 2000 is the optimum.