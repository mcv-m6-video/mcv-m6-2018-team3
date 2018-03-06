from estimator import Estimator
from utils import *

class rgbEstimator(Estimator):

    def __init__(self, ):
        super().__init__()

    def fit(self, X, y):
        assert len(X.shape) == 4, RuntimeError("Expected RGB images")
        assert len(X.shape) == 4, RuntimeError("Expected RGB images")

        idx = simplify_labels(y)
        self.mu = np.array([np.nanmean(X[:,:,:,ch] * idx, axis=0) for ch in range(3)])
        self.var = np.array([np.nanvar(X[:, :, :, ch] * idx, axis=0) for ch in range(3)])

        return self

    def predict(self, X):
        assert len(X.shape) == 4, RuntimeError("Expected RGB images")

        prediction = []

        for ch in range(3):
            ch_prediction = np.zeros(X[:,:,:,ch].shape)
            ch_prediction[np.absolute(X[:,:,:,ch] - self.mu[ch]) >= self.alpha * (self.var[ch] + 2)] = 1
            prediction.append(ch_prediction)

        return np.prod(prediction, axis=0)