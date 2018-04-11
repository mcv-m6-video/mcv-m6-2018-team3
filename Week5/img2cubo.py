import numpy as np
import cv2
import os
from utils import *


N = 101
cubo = np.zeros([N, 240, 320])

path = 'test'
files = sorted(os.listdir(path))


for i in range(0, N):
    img = cv2.imread('test/' + files[i], 0)
    #height, width = img.shape[:2]
    res = cv2.resize(img, (320, 240), interpolation=cv2.INTER_CUBIC)
    cubo[i,:,:] = res

np.save('output.npy', cubo)
write_images2(cubo, "borrar", 'output_')



# =============================

cubo2 = np.zeros([N, 240, 320])

path = 'test_GT'
files = sorted(os.listdir(path))

for i in range(0, N):
    img = cv2.imread('test_GT/' + files[i], 0)
    res = cv2.resize(img, (320, 240), interpolation=cv2.INTER_CUBIC)
    cubo2[i,:,:] = res

np.save('GToutput.npy', cubo2)
write_images2(cubo2, "borrar", 'GToutput_')



