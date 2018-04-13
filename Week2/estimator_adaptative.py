from utils import *
from estimator import Estimator

class EstimatorAdaptative(Estimator):
    """Adaptative classifier"""

    def __init__(self, metric='f1', alpha=4, rho=0.5):
        """
        Initialization of the classifier
        """
        super().__init__(metric=metric)
        self.alpha = alpha
        self.rho = rho


    def predict(self, X):
        try:
            getattr(self, "mu")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        mu = self.mu
        var = self.var
        RHO = np.ones(X.shape[1:3])*self.rho
        predictions = np.zeros(X.shape, dtype=np.uint8)
        for i in range(0, X.shape[0]):

            #Predict frame
            img = X[i, :, :]
            pred = np.abs(img - mu) >= self.alpha * (np.sqrt(var) + 2)
            pred = pred.astype(np.uint8)
            predictions[i] = pred

            #At each prediction: adapt mean (mu) and variance (var)
            mu = (pred * mu) + ((RHO * img) + ((1.0 - RHO) * mu)) * (1 - pred)
            var = (pred * var) + ((RHO * (img - mu) ** 2) + ((1.0 - RHO) * var)) * (1 - pred)

        return predictions

    def set_rho(self, rho):
        self.rho = rho