from estimator_adaptative import EstimatorAdaptative
from utils import *

class rgbEstimatorAdaptative(EstimatorAdaptative):

    def __init__(self, metric='f1'):
        super().__init__(metric=metric)

    def fit(self, x, y):
        assert len(x.shape) == 4, RuntimeError("Expected RGB images")
        assert len(x.shape) == 4, RuntimeError("Expected RGB images")

        y = simplify_labels(y)
        RHO = np.ones(x.shape[1:4])
        mu = np.zeros(x.shape[1:4])
        for i in range(0, x.shape[0]):
            frame = x[i, :, :]
            RHO[np.where(mu != 0)] = self.rho
            mu_old = mu
            mu = RHO * frame + (1 - RHO) * mu
            mu[np.where(np.isnan(y[i, :, :]))] = mu_old[np.where(np.isnan(y[i, :, :]))]

        RHO = np.ones(x.shape[1:4])
        var = np.zeros(x.shape[1:4])
        for i in range(0, x.shape[0]):
            frame = x[i, :, :]
            RHO[np.where(var != 0)] = self.rho
            var_old = var
            var = RHO * (frame - mu) ** 2 + (1 - RHO) * var
            var[np.where(np.isnan(y[i, :, :]))] = var_old[np.where(np.isnan(y[i, :, :]))]
        self.mu = mu
        self.var = var
        return self

    def predict(self, x):
        assert len(x.shape) == 4, RuntimeError("Expected RGB images")
        frame_prediction = []
        predictions = []
        for i in range(0, x.shape[0]):
            for ch in range(3):
                ch_prediction = np.zeros(x[0, :, :, ch].shape)
                ch_prediction[np.absolute(x[i, :, :, ch] - self.mu[:,:,ch]) >= self.alpha * (self.var[:,:,ch] + 2)] = 1
                frame_prediction.append(ch_prediction)
            predictions.append(np.prod(frame_prediction, axis=0))

        return np.array(predictions)
