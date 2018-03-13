import cv2
import sys
from utils import *

def Erosion(images, kernel, iterations=1):

    erosion = []

    for i in range(images.shape[0]):
        e = cv2.erode(images[i], kernel, iterations=iterations)
        erosion.append(e)

    return np.array(erosion, dtype=np.uint8)


def Dilatation(images, kernel, iterations=1):

    dilation = []

    for i in range(images.shape[0]):
        d = cv2.dilate(images[i], kernel, iterations=iterations)
        dilation.append(d)

    return np.array(dilation, dtype=np.uint8)


def Opening(images, kernel):

    opening = []

    for i in range(images.shape[0]):
        o = cv2.morphologyEx(images[i], cv2.MORPH_OPEN, kernel)
        opening.append(o)

    return np.array(opening, dtype=np.uint8)


def Closing(images, kernel):

    closing = []

    for i in range(images.shape[0]):
        c = cv2.morphologyEx(images[i], cv2.MORPH_CLOSE, kernel)
        closing.append(c)

    return np.array(closing, dtype=np.uint8)


def Gradient(images, kernel):
    gradient = []

    for i in range(images.shape[0]):
        gradient = cv2.morphologyEx(images[i], cv2.MORPH_GRADIENT, kernel)
        gradient.append(c)

    return np.array(gradient, dtype=np.uint8)


def Top_Hat(images, kernel):
    tophat = []

    for i in range(images.shape[0]):
        th = cv2.morphologyEx(images[i], cv2.MORPH_TOPHAT, kernel)
        tophat.append(th)

    return np.array(tophat, dtype=np.uint8)


def Black_Hat(images, kernel):
    blackhat = []

    for i in range(images.shape[0]):
        bh = cv2.morphologyEx(images[i], cv2.MORPH_BLACKHAT, kernel)
        blackhat.append(th)

    return np.array(blackhat, dtype=np.uint8)




# ================== TESTING ================
# data_path = '../../databases'
# PlotsDirectory = '../plots/Week3/task1/'
#
# if not os.path.exists(PlotsDirectory):
#     os.makedirs(PlotsDirectory)
#
# names = ['highway', 'fall', 'traffic']
# estimation_range = [np.array([1050, 1200]), np.array([1460, 1510]), np.array([950, 1000])]
# prediction_range = [np.array([1201, 1350]), np.array([1511, 1560]), np.array([1001, 1050])]
#
# i=0
# [X_est, y_est] = load_data(data_path, names[i], estimation_range[i], grayscale=True)
# [X_pred, y_pred] = load_data(data_path, names[i], prediction_range[i], grayscale=True)
#
#
# kernel = np.ones((5,5),np.uint8)
# th, im_th = cv2.threshold(X_pred[100], 58, 255, cv2.THRESH_BINARY_INV)
#
# result = Erosion(im_th, kernel, 1)
# cv2.imshow("Original image", X_pred[100])
# cv2.imshow("Binary image", im_th)
# cv2.imshow("Erosion image", result)
# cv2.waitKey(0)