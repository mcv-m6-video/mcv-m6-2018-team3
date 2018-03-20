import cv2
import numpy as np

def gunner_farneback(prvs, curr):
    flow = cv2.calcOpticalFlowFarneback(prvs, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow