import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import visual_of
from gunner_farneback import gunner_farneback
from block_matching import get_block_matching

def visualize_custom(seq, dataset, names):
    img = cv2.imread(names[1])
    prvs = cv2.imread(names[1], 0)
    curr = cv2.imread(names[2], 0)

    #Parameters
    offset = 24
    block = 50

    test = get_block_matching(curr, prvs, block, block, offset, offset)
    testx = (np.array(test[:, :, 1], dtype=float)) / offset
    testy = (np.array(test[:, :, 2], dtype=float)) / offset
    testz = np.array(test[:, :, 0], dtype=bool)

    # im_of = visual_of(im, testx, testy, testz, overlap=0.9, wsize=300, mult=1, thickness=1) #normalized
    # im_of = visual_of(im, testx, testy, testz, overlap=0.45, wsize=50, mult=1, thickness=1) #normalized2
    # im_of = visual_of(im, testx, testy, testz, overlap=0.85, wsize=180, mult=9, thickness=2) #custom1
    im_of = visual_of(img, testx, testy, testz, overlap=0.3, wsize=40, mult=25, thickness=2)  # custom2

    plt.figure()
    plt.imshow(im_of)
    plt.title(seq + " " + dataset)
    plt.show()

def visualize_gunner_farneback(seq, dataset, names):
    img = cv2.imread(names[1])
    prvs = cv2.imread(names[1], 0)
    curr = cv2.imread(names[2], 0)

    test = gunner_farneback(prvs, curr)
    heigh = test.shape[0]
    width = test.shape[1]
    test = np.concatenate((test, np.ones((heigh, width, 1))), axis=2)
    testx = (np.array(test[:, :, 1], dtype=float)) / heigh
    testy = (np.array(test[:, :, 2], dtype=float)) / width
    testz = np.array(test[:, :, 0], dtype=bool)

    # im_of = visual_of(im, testx, testy, testz, overlap=0.9, wsize=300, mult=1, thickness=1) #normalized
    # im_of = visual_of(im, testx, testy, testz, overlap=0.45, wsize=50, mult=1, thickness=1) #normalized2
    # im_of = visual_of(im, testx, testy, testz, overlap=0.85, wsize=180, mult=9, thickness=2) #custom1
    im_of = visual_of(img, testx, testy, testz, overlap=0.3, wsize=40, mult=25, thickness=2)  # custom2

    plt.figure()
    plt.imshow(im_of)
    plt.title(seq + " "+ dataset)
    plt.show()
if __name__ == "__main__":
    # First pair
    gt45_name = "../../databases/data_stereo_flow/training/flow_noc/000045_10.png"
    test45_name1 = "../../databases/data_stereo_flow/training/image_0/000045_10.png"
    test45_name2 = "../../databases/data_stereo_flow/training/image_0/000045_11.png"
    names45 = [gt45_name, test45_name1, test45_name2]

    # Second pair
    gt157_name = "../../databases/data_stereo_flow/training/flow_noc/000157_10.png"
    test157_name1 = "../../databases/data_stereo_flow/training/image_0/000157_10.png"
    test157_name2 = "../../databases/data_stereo_flow/training/image_0/000157_11.png"
    names157 = [gt157_name, test157_name1, test157_name2]

    visualize_custom("45", "KITTI", names45)
    visualize_custom("157", "KITTI", names157)
    visualize_gunner_farneback("45", "KITTI", names45)
    visualize_gunner_farneback("157", "KITTI", names157)