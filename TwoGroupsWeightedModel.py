from sklearn.base import BaseEstimator, ClassifierMixin
import importlib
from copy import deepcopy


class TwoGroupsWeightedModel(BaseEstimator, ClassifierMixin):
    """
    A wrapper model that splits the X data into two groups, each using a subset of the X space explanatory variables.
    It then applies the "underlying estimator" on each group separately, and assigns every observations a
    probability that is a weighted mixture of the probability assigned by underlying estimator to each group.
    """
    def __init__(self, underlying_estimator_module_and_class, group1_var_names, group2_var_names,
                 steps, group1_weight=0.5, group2_weight=0.5,
                 weighting_function=None, classification_threshold=0.5, **kwargs):
        """
        :param underlying_estimator_class_name: a string holding the class name of the classifier to be
                                                used on each group. Has to be given as a string because of a bug in
                                                sklearn, see line 243 sklearn's base.py
        :param group1_var_names: List of explanatory variables to be used in constructing Group1.
        :param group2_var_names: List of explanatory variables to be used in constructing Group2.
        :param group1_weight: Group 1's weight for final probability mixture.
        :param group2_weight: Group 2's weight for final probability mixture.
        :param weighting_function: Optional, use a custom weighting function instead of simple weights.
        :param classification_threshold: probability threshold for classification as class '1'.
        :param kwargs: All init parameters for the underlying classifier.
        """
        self.steps = steps
        self.group1_var_names = group1_var_names
        self.group2_var_names = group2_var_names
        self.underlying_estimator_params_dict = {k:v for k,v in kwargs.items()}
        self.underlying_estimator_module_and_class = underlying_estimator_module_and_class
        self.underlying_estimator = \
            self._instantiate_class_from_module(self.underlying_estimator_module_and_class.split(" ")[0],
                                            self.underlying_estimator_module_and_class.split(" ")[1])

        self.weighting_function = weighting_function
        self.group1_weight = group1_weight
        self.group2_weight = group2_weight
        self.classification_threshold = classification_threshold
        self.group1_X, self.group2_X = None, None

    def fit(self, X, y):
        if (self.group1_X is None) or (self.group2_X is None): # i.e, no transformation was applied
            self.group1_X = X[self.group1_var_names]
            self.group2_X = X[self.group2_var_names]
        self.group1_estimator = deepcopy(self.underlying_estimator.fit(self.group1_X, y))
        self.group2_estimator = deepcopy(self.underlying_estimator.fit(self.group2_X, y))

    def predict_proba(self, X):
        group1_predictions = self.group1_estimator.predict_proba(X[self.group1_var_names])
        group2_predictions = self.group2_estimator.predict_proba(X[self.group2_var_names])
        if self.weighting_function:
            return self.weighting_function(group1_predictions, group2_predictions)
        else:
            if self.group1_weight + self.group2_weight != 1:
                print("Warning: group weights ({},{}) do not sum up to 1.".format(self.group1_weight, self.group2_weight))
            return self.group1_weight*group1_predictions + self.group2_weight*group2_predictions

    def predict(self, X):
        probs = self.predict_proba(X)
        probs_for_class_1 = probs[:,1]
        class_predictions = probs_for_class_1 > self.classification_threshold
        return class_predictions

    def _instantiate_class_from_module(self, module_name, class_name):
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        instance = class_(self.steps)
        return instance
