from sklearn.base import BaseEstimator, ClassifierMixin
from utils import *


class Estimator(BaseEstimator, ClassifierMixin):
    """Classifier"""

    def __init__(self, metric="f1"):
        """
        Initialization of the classifier
        """
        self.alpha = None
        self.metric = metric

    def fit(self, X, y):
        idx = simplify_labels(y)

        self.mu = np.nanmean(X * idx, axis=0)
        self.var = np.nanvar(X * idx, axis=0)

        return self

    def predict(self, X):
        try:
            getattr(self, "mu")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        prediction = np.zeros(X.shape)
        prediction[np.absolute(X - self.mu) >= self.alpha * (self.var + 2)] = 1

        return prediction

    def score(self, X, y, sample_weight=None):
        prediction = self.predict(X)
        PE = pixel_evaluation(y, prediction)
        if self.metric == 'f1':
            return(f1_score(PE))
        elif self.metric == 'precision':
            return(f1_score(PE))
        elif self.metric == 'recall':
            return(f1_score(PE))
        elif self.metric == 'accuracy':
            return (sum(self.predict(X)))
        else:
            raise RuntimeError("Invalid metric")

    def set_alpha(self, alpha):
        self.alpha = alpha

