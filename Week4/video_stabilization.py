import cv2
import numpy as np
from utils import *

from block_matching import get_block_matching


# def get_stabilization(prev_img, motion):
#
#     stabilized_prev_img = np.zeros(prev_img.shape)
#
#     for idx in range(prev_img.shape[0]):
#         for idy in range(prev_img.shape[1]):
#
#             #if motion[idx, idy, 2] is not 0:
#             stabilized_prev_img[idx, idy] = prev_img[idx + int(motion[idx, idy, 0]), idy + int(motion[idx, idy, 1])]
#
#     return stabilized_prev_img



# Backward prediction
def video_stabilization(sequence, block_size_x, block_size_y, search_area_x, search_area_y, compensation = 'backward', grayscale=True, resize=None):

    #resize (x, y)

    N = len(sequence)

    prev_img = sequence[0]
    #if not grayscale:
    #    prev_img = rgb2gray(prev_img)

    if resize is not None:
        sequence_stabilized = np.zeros([sequence.shape[0], resize[1], resize[0], 3])
    else:
        sequence_stabilized = np.zeros(sequence.shape)

    for idx in range(1, N):
        print(idx, N)
        curr_img = sequence[idx]


        if not grayscale:
            color_curr_frame = np.copy(curr_img)
            curr_img = rgb2gray(curr_img)
            prev_img = rgb2gray(prev_img)

        if resize is not None:
            curr_img = cv2.resize(curr_img, resize)
            prev_img = cv2.resize(prev_img, resize)
            aux_color = np.zeros([resize[1], resize[0], 3])
            aux_color[:, :, 0] = cv2.resize(color_curr_frame[:,:,0], resize)
            aux_color[:, :, 1] = cv2.resize(color_curr_frame[:, :, 1], resize)
            aux_color[:, :, 2] = cv2.resize(color_curr_frame[:, :, 2], resize)
            color_curr_frame = np.copy(aux_color)

        optical_flow = get_block_matching(curr_img, prev_img, block_size_x, block_size_y, search_area_x, search_area_y, compensation = 'backward')

        u = np.median(optical_flow[:,:,0])
        v = np.median(optical_flow[:,:,1])

        # translation matrix
        affine_H = np.float32([[1, 0, -u],
                               [0, 1, -v]])


        if grayscale:
            stabilized_frame = cv2.warpAffine(curr_img, affine_H, (curr_img.shape[1], curr_img.shape[0]))
        else:
            stabilized_frame = np.zeros(aux_color.shape)
            stabilized_frame[:, :, 0] = cv2.warpAffine(color_curr_frame[:,:,0], affine_H, (curr_img.shape[1], curr_img.shape[0]))
            stabilized_frame[:, :, 1] = cv2.warpAffine(color_curr_frame[:, :, 1], affine_H,
                                                     (curr_img.shape[1], curr_img.shape[0]))
            stabilized_frame[:, :, 2] = cv2.warpAffine(color_curr_frame[:, :, 2], affine_H,
                                                     (curr_img.shape[1], curr_img.shape[0]))

        # update the previous image to the estabilized current image
        sequence_stabilized[idx - 1] = stabilized_frame
        prev_img = stabilized_frame



    return  sequence_stabilized

# =================================
import os
from utils import *
import matplotlib.pyplot as plt

data_path = '../../databases'
PlotsDirectory = '../plots/Week4/task21/'

if not os.path.exists(PlotsDirectory):
    os.makedirs(PlotsDirectory)

name = 'traffic'
seq_range = np.array([950, 1050])


[seq, y] = load_data(data_path, name, seq_range, grayscale=True)


block_size_x, block_size_y, search_area_x, search_area_y = 5, 5, 10, 10
est_seq, est_gt = video_stabilization(seq, y, block_size_x, block_size_y, search_area_x, search_area_y, compensation = 'backward')

np.save(PlotsDirectory + 'traffic_stabilized_bloque5_area10.npy', est_seq)
np.save(PlotsDirectory + 'traffic_gt_stabilized_bloque5_area10.npy', est_gt)
write_images2(est_seq, PlotsDirectory, 'traffic_stabilized_bloque5_area10_')
write_images2(est_gt, PlotsDirectory, 'traffic_gt_stabilized_bloque5_area10_')

# sequence = seq
# N = 3  # len(sequence)+1
#
# prev_img = sequence[0]
#
# sequence_stabilized = np.copy(sequence)
#
# for idx in range(1, N):
#     print(idx, N)
#     curr_img = sequence[idx]
#
#     optical_flow = get_block_matching(curr_img, prev_img, block_size_x, block_size_y, search_area_x, search_area_y,
#                                       compensation='backward')
#
#     mean_x = np.mean(optical_flow[:, :, 0])
#     mean_y = np.mean(optical_flow[:, :, 1])
#     #print(mean_x)
#     #print(mean_y)
#
#     # optical_flow[:, :, 0] = np.around(optical_flow[:, :, 0], decimals=2)
#     # optical_flow[:, :, 1] = np.around(optical_flow[:, :, 1], decimals=2)
#     # unique_x, counts_x = np.unique(optical_flow[:, :, 0], return_counts=True)
#     # unique_y, counts_y = np.unique(optical_flow[:, :, 1], return_counts=True)
#
#     optical_flow[:, :, 0] = np.ones(optical_flow.shape[:2]) * mean_x
#     optical_flow[:, :, 1] = np.ones(optical_flow.shape[:2]) * mean_y
#
#     # optical_flow[:,:,0][abs(optical_flow[:,:,0])<(mean_x)] = 0
#     # optical_flow[:, :, 1][abs(optical_flow[:, :, 1])<(mean_y)] = 0
#
#     stabilized_frame = get_stabilization(prev_img, optical_flow)
#     prev_img = curr_img
#
#     sequence_stabilized[idx - 1, :, :] = stabilized_frame
#
#
# for i in range(N):
#     cv2.imshow('original', seq[i])
#     cv2.imshow('image', sequence_stabilized[i])
#     cv2.waitKey(delay=0)


#write_video(est_seq, "traffic_stabilized.avi")