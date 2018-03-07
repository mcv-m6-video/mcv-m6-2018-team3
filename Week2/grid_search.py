from sklearn.model_selection import ParameterGrid

class GridSearch:

    def __init__(self, estimator, param_grid):
        """
        Initialization
        """
        self.estimator = estimator
        self.param_grid=param_grid

    def fitAndPredict(self, X_est, X_pred, y_est, y_pred):
        params = list(ParameterGrid(self.param_grid))
        self.results = list()
        for param in params:
            print(param)
            self.estimator.set_alpha(param['alpha'])
            self.estimator.set_rho(param['rho'])
            self.estimator.fit(X_est, y_est)
            self.results.append(self.estimator.score(X_pred, y_pred))
