from sklearn.base import BaseEstimator, ClassifierMixin
from utils import *


class Estimator(BaseEstimator, ClassifierMixin):
    """Classifier"""

    def __init__(self, X_res=None, y_pred=None, metric="f1"):
        """
        Initialization of the classifier
        """
        self.metric = metric
        self.X_res = X_res
        self.y_pred = y_pred

    def fit(self, X, y=None):
        if y is not None:
            y = simplify_labels(y)
            self.mu = np.nanmean(X * y, axis=0)
            self.var = np.nanvar(X * y, axis=0)
        else:
            self.mu = np.nanmean(X, axis=0)
            self.var = np.nanvar(X, axis=0)

        return self

    def predict(self, X):
        try:
            getattr(self, "mu")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        self.X_res = np.zeros(X.shape)
        self.X_res[np.absolute(X - self.mu) >= self.alpha * (self.var + 2)] = 1

        return self.X_res

    def score(self, y, X=None, sample_weight=None):
        y = build_mask(y)
        if X is not None and self.X_res is None:
            self.predict(X)
        elif X is None and self.X_res is None:
            RuntimeError("Can't compute score")
        PE = pixel_evaluation(self.X_res, y)
        if self.metric == 'f1':
            return f1_score(PE)
        elif self.metric == 'precision':
            return precision(PE)
        elif self.metric == 'recall':
            return recall(PE)
        elif self.metric == 'fpr':
            return fpr_metric(PE)
        elif self.metric == 'tpr':
            return tpr_metric(PE)
        else:
            raise RuntimeError("Invalid metric")

    def set_alpha(self, alpha):
        self.alpha = alpha
