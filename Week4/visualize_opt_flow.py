import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import visual_of

if __name__ == "__main__":
    gt_name = "../databases/data_stereo_flow/training/flow_noc/000045_10.png"
    im_name = "../databases/data_stereo_flow/training/image_0/000045_10.png"

    gt = cv2.imread(gt_name, -1)
    im = cv2.imread(im_name)

    gtx = (np.array(gt[:, :, 1], dtype=float) - (2 ** 15)) / 64.0
    gty = (np.array(gt[:, :, 2], dtype=float) - (2 ** 15)) / 64.0
    gtz = np.array(gt[:, :, 0], dtype=bool)

    # im_of = visual_of(im, gtx, gty, gtz, overlap=0.9, wsize=300, mult=1, thickness=1) #normalized
    # im_of = visual_of(im, gtx, gty, gtz, overlap=0.45, wsize=50, mult=1, thickness=1) #normalized2
    # im_of = visual_of(im, gtx, gty, gtz, overlap=0.85, wsize=180, mult=9, thickness=2) #custom1
    im_of = visual_of(im, gtx, gty, gtz, overlap=0.3, wsize=40, mult=25, thickness=2)  # custom2

    plt.imshow(im_of)
    plt.show()