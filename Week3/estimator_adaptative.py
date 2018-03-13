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
        RHO = np.ones(x.shape[1:3])
        mu = np.zeros(x.shape[1:3])
        for i in range(0, x.shape[0]):
            frame = x[i, :, :]
            RHO[np.where(mu != 0)] = self.rho
            mu_old = mu
            mu = RHO * frame + (1 - RHO) * mu
            if y is not None:
                mu[np.where(np.isnan(y[i, :, :]))] = mu_old[np.where(np.isnan(y[i, :, :]))]

        RHO = np.ones(x.shape[1:3])
        var = np.zeros(x.shape[1:3])
        for i in range(0, x.shape[0]):
            frame = x[i, :, :]
            RHO[np.where(var != 0)] = self.rho
            var_old = var
            var = RHO * (frame - mu) ** 2 + (1 - RHO) * var
            if y is not None:
                var[np.where(np.isnan(y[i, :, :]))] = var_old[np.where(np.isnan(y[i, :, :]))]
        self.mu = mu
        self.var = var
        return self

    def set_rho(self, rho):
        self.rho = rho