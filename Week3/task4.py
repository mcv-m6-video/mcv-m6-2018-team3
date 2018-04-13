from utils import *
from estimator_adaptative import week2_masks, evaluate
from task3 import task3
from morphology import Dilatation, Closing, Opening
from hole_filling import hole_filling, hole_filling2




data_path = '../../databases'
PlotsDirectory = '../plots/Week3/task4/'

if not os.path.exists(PlotsDirectory):
    os.makedirs(PlotsDirectory)

names = ['highway', 'fall', 'traffic']
estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]

def task4(X_est, X_pred, rho, alpha, apply = True):

    mask = week2_masks(X_est, X_pred, rho, alpha)

    if apply:
        shadows = MOG2(X_pred)
        mask[np.where(shadows == 1)] = 0

    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))


    mask = Closing(mask, kernel_closing)
    mask = hole_filling2(mask, connectivity=8, visualize=False)
    mask = Opening(mask, kernel_opening)


    return mask

def main():
    data_path = '../../databases'
    PlotsDirectory = '../plots/Week3/task1/'

    if not os.path.exists(PlotsDirectory):
        os.makedirs(PlotsDirectory)

    names = ['highway', 'fall', 'traffic']
    estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
    prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]
    #alpha = [{'min': 4, 'max': 20, 'step': 1.5}, {'min': 1, 'max': 10, 'step': 1}, {'min': 1, 'max': 20, 'step': 1.5}]
    #ro = [{'min': 1, 'max': 10, 'step': 1}, {'min': 1, 'max': 10, 'step': 1}, {'min': 1, 'max': 10, 'step': 1}]

    params = { 'highway': {'alpha': 2.89, 'rho': 0.21},
               'fall': {'alpha': 3.2, 'rho': 0.05},
               'traffic': {'alpha': 3.55, 'rho': 0.16}}

    for i in range(len(names)):
        [X_est, y_est] = load_data(data_path, names[i], estimation_range[i], grayscale=True)
        [X_pred, y_pred] = load_data(data_path, names[i], prediction_range[i], grayscale=True)


        masks = task4(X_est, X_pred, params['traffic']['rho'], params['traffic']['alpha'], True)
        masksno = task4(X_est, X_pred, params['traffic']['rho'], params['traffic']['alpha'], False)

        print(names[i] + ": F1 score with shadow = " + str(evaluate(masks, y_pred, 'f1')))
        print(names[i] + ": F1 score without shadow = " + str(evaluate(masksno, y_pred, 'f1')))

if __name__ == "__main__":
    main()
