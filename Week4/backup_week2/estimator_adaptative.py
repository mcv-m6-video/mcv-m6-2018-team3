from utils import *
from estimator import Estimator

class EstimatorAdaptative(Estimator):
    """Adaptative classifier"""

    def __init__(self, X_res=None, y_pred=None, metric='f1', alpha=4, rho=0.5):
        """
        Initialization of the classifier
        """
        super().__init__(X_res=X_res, y_pred=y_pred, metric=metric)
        self.alpha = alpha
        self.rho = rho

    def fit(self, x, y=None):
        if y is not None:
            y = simplify_labels(y)
            mu = np.nanmean(x * y, axis=0)
            var = np.nanvar(x * y, axis=0)
        else:
            mu = np.nanmean(x, axis=0)
            var = np.nanvar(x, axis=0)

        RHO = np.ones(x.shape[1:3])*self.rho
        mus = []
        for i in range(0, x.shape[0]):
            frame = x[i, :, :]
            out = np.abs(frame - mu) >= self.alpha * (np.sqrt(var) + 2)
            out = out.astype(np.uint8)
            mu_old = mu
            var_old = var
            mu = (out*mu) + ((RHO * frame) + ((1.0 - RHO) * mu)) * (1 - out)
            mus.append(mu)
            var = (out * var) + ((RHO * (frame - mu) ** 2) + ((1.0 - RHO) * var)) * (1 - out)
            if y is not None:
                mu[np.where(np.isnan(y[i, :, :]))] = mu_old[np.where(np.isnan(y[i, :, :]))]
                var[np.where(np.isnan(y[i, :, :]))] = var_old[np.where(np.isnan(y[i, :, :]))]

        self.mu = mu
        self.var = var
        return self

    def set_rho(self, rho):
        self.rho = rho

def week2_masks(X_est, X_pred, rho, alpha):
    est = EstimatorAdaptative(alpha=alpha, rho=rho)
    est.fit(X_est)
    return est.predict(X_pred)

def evaluate(X_res, y_pred, metric="f1"):
    est = EstimatorAdaptative(X_res=X_res, y_pred=y_pred, metric=metric)
    return est.score(y_pred)