from utils import load_data, fit
import cv2
import numpy as np

data_path = '../../databases'

# Sequence 1: Highway

data_id = 'highway'

estimation_range = np.array([1050, 1200])
prediction_range = np.array([1201, 1350])

[X_est, y_est] = load_data(data_path, data_id, estimation_range, grayscale=True)
[X_pred, y_pred] = load_data(data_path, data_id, prediction_range, grayscale=True)

background_model = fit(X_est, y_est)
