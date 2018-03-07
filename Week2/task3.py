# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html
import numpy as np
import cv2
from utils import *
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

#load_data, pixel_evaluation, precision, recall, f1_score, build_mask

#Upload evaluation Frames
data_path = '../../databases'
data_id = 'fall'


prediction_range = np.array([1050, 1350]) # highway: ([1050, 1350]) , fall: ([1460, 1560]) , traffic: ([950, 1050])
[X_pred, y_pred] = load_data(data_path, data_id, prediction_range, grayscale=True)

def MOG(X_pred, y_pred, alpha):
    #BackgroundSubtractorMOG
    fgbgMOG = cv2.bgsegm.createBackgroundSubtractorMOG(history=25)
    maskMOG=[]

    for frame in X_pred:
        fgmask = fgbgMOG.apply(frame, learningRate=alpha)
        maskMOG.append(fgmask)

        aux = np.array(fgmask)
        aux2 = np.divide(aux, 255)

        cv2.imshow('image', aux2)
        cv2.waitKey(delay=7)


    maskMOG=np.array(maskMOG,dtype=np.uint8)
    maskMOG=np.divide(maskMOG,255)

    y_pred = build_mask(y_pred)

    #highway: (maskMOG[150:301], y_pred[150:301])
    #fall: (maskMOG[50:101], y_pred[50:101])
    #traffic: (maskMOG[50:101], y_pred[50:101])

    print("roc auc score : ", roc_auc_score(y_pred[50:101], X_pred[50:101]))
    EvaluationMOG=pixel_evaluation(maskMOG[150:301], y_pred[150:301])


    return EvaluationMOG


lr = np.array([0.00001, 0.0001, 0.001,0.01,0.1]) # highway: ([0.00001, 0.0001, 0.001,0.01,0.1]) , fall: ([0.01, 0.001])
RecallMOG = np.zeros(len(lr))
FPRMOG = np.zeros(len(lr))
PrecisionMOG = np.zeros(len(lr))
F1scoreMOG = np.zeros(len(lr))

AOMOG=[]
for idx, alpha in enumerate(lr):
    print(alpha)
    eMOG=MOG(X_pred, y_pred, alpha)
    print("eMOG[TP, TN, FP, FN, TF]\n", eMOG)


    RecallMOG[idx] = recall(eMOG)
    #print("RecallMOG\n", RecallMOG)

    FPRMOG[idx] = FPR(eMOG)
    #print("FPRMOG\n", FPRMOG)

    PrecisionMOG[idx] = precision(eMOG)
    F1scoreMOG[idx] = f1_score(eMOG)


print("RecallMOG\n", RecallMOG)
print("FPRMOG\n", FPRMOG)
print("PrecisionMOG\n", PrecisionMOG)
print("F1 score\n", F1scoreMOG)

#plt.plot(FPRMOG, RecallMOG)
#plt.show()

AUC=metrics.auc(FPRMOG,RecallMOG,reorder=True)
print("AUC : ",AUC)

plt.figure(1)
plt.plot(FPRMOG, RecallMOG, 'b', label='ROC')
plt.title("ROC")
plt.xlabel("FPR = FP / (FP + TN)")
plt.ylabel("Recall")
plt.savefig('ROC_'+data_id, bbox_inches='tight')
plt.close()
