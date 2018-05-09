import pandas as pd
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, NuSVC
from sklearn.base import clone, TransformerMixin
from matplotlib import pyplot as plt
import constants


def get_accuracy(test_preds, test_Y):
    errs = 0
    for i in range(len(test_y)):
        if test_y.values[i] != test_preds[i]:
            errs += 1
    return 1-errs/float(len(test_y))


class SubsetTransformer(TransformerMixin):
    def __init__(self, start=0, end=None):
        self.start = start
        self.end = end

    def fit(self, *_):
        return self

    def transform(self, X, *_):
        if self.end is None:
            return X.iloc[:, self.start:]
        else:
            return X.iloc[:, self.start: self.end]


select_left = SubsetTransformer(0, 9)
select_right = SubsetTransformer(9, -1)
scale = StandardScaler()





full_set = pd.DataFrame.from_csv(constants.DATA_FOLDER + r"\\full_set_fixed.csv")

#Normalize data:
scaler = StandardScaler()
normalized = scaler.fit_transform(full_set)
normalized = pd.DataFrame(index=full_set.index, data=normalized, columns=full_set.columns)
#normalized.Parkinsons = normalized.Parkinsons.apply(lambda x : 1 if x>0 else 0)
normalized.Parkinsons = full_set.Parkinsons.apply(lambda x: x)
full_set = normalized


# TODO: make sure all original variables ae here
hold_variables = ['mean_L', 'mean_R', 'std_L', 'std_R', 'skew_L', 'skew_R', 'kurtosis_L',
       'kurtosis_R', 'mean_hold_diff']

latency_variables= ['mean_LL', 'mean_LR', 'mean_RL',
       'mean_RR', 'std_LL', 'std_LR', 'std_RL', 'std_RR', 'skew_LL', 'skew_LR',
       'skew_RL', 'skew_RR', 'kurtosis_LL', 'kurtosis_LR', 'kurtosis_RL',
       'kurtosis_RR', 'mean_LR_RL_diff', 'mean_LL_RR_diff']



train_X, test_X, train_y, test_y = train_test_split(full_set.iloc[:, :-1], full_set.iloc[:, -1],
                                                    stratify=full_set.iloc[:, -1], test_size=0.35)


hold_train = train_X[hold_variables]
latency_train = train_X[latency_variables]
hold_test = test_X[hold_variables]
latency_test = test_X[latency_variables]


lda = LinearDiscriminantAnalysis()

hold_train = lda.fit_transform(hold_train, train_y)
hold_test = lda.fit_transform(hold_test, test_y)

latency_train = lda.fit_transform(latency_train, train_y)
latency_test = lda.fit_transform(latency_test, test_y)

hold_clfs = [RandomForestClassifier(), SVC(probability=True),
             MLPClassifier(), LogisticRegression(),
             NuSVC(probability=True), KNeighborsClassifier(), ]
latency_clfs = [RandomForestClassifier(), SVC(probability=True),
             MLPClassifier(), LogisticRegression(),
             NuSVC(probability=True), KNeighborsClassifier(), ]


def get_hold_test_prob_predictions(clf):
    """
    fits given classifier and calculaes test prediction on hold test set
    """
    clf.fit(hold_train, train_y)
    index_of_PD_prob = list(clf.classes_).index(1)  # see predict_proba doc
    preds = [p[index_of_PD_prob] for p in clf.predict_proba(hold_test)]
    return preds

def get_latency_test_prob_predictions(clf):
    """
    fits given classifier and calculaes test prediction on latency test set
    """
    clf.fit(latency_train, train_y)
    index_of_PD_prob = list(clf.classes_).index(1)  # see predict_proba doc
    preds = [p[index_of_PD_prob] for p in clf.predict_proba(latency_test)]
    return preds


hold_probs_list = [get_hold_test_prob_predictions(hold_clf) for hold_clf in hold_clfs]
latency_probs_list = [get_latency_test_prob_predictions(latency_clf) for latency_clf in latency_clfs]

final_probs = []
for i in range(len(test_X)):
    PDprob = 0
    assert len(hold_probs_list) == len(latency_probs_list)
    num_models = len(hold_probs_list)

    for probs_list in hold_probs_list: #iterate every hold model
        PDprob += probs_list[i]
    for probs_list in latency_probs_list:  # iterate every latency model
        PDprob += ( 0.5 + 1.2*( probs_list[i] - 0.5) ) #TODO: seems to work just as well with a simple weight, see next line
        #PDprob += 1.2 * (probs_list[i])
    PDprob = PDprob / float(2*num_models)
    final_probs.append(PDprob)

print([(round(p,2), label) for p, label in zip(final_probs, test_y)])

final_preds = [1 if p>=0.5 else 0 for p in final_probs]
print("accuracy: ", get_accuracy(final_preds, test_y))