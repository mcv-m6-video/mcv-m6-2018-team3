import cv2
import numpy as np

from block_matching import get_block_matching


def get_stabilization(prev_img, motion):
    """
    
    :param curr_img: grayscale frame t+1
    :param prev_img: grayscale frame t
    :param motion: optical_flow map, [motion_x, motion_y, valid]
    :return: stabilized_prev_img
    """
    
    stabilized_prev_img = np.zeros(prev_img.shape)
    
    for idx in range(prev_img.shape[0]):
        for idy in range(prev_img.shape[1]):
            
            if motion[idx, idy, 2] is not 0:
                stabilized_prev_img[idx, idy] = prev_img[idx + motion[idx, idy, 0], idy + motion[idx, idy, 1]]
    
    return stabilized_prev_img



# Backward prediction
def video_stabilization(sequence, block_size_x, block_size_y, search_area_x, search_area_y, compensation = 'backward'):


    N = len(sequence)+1

    prev_img = sequence[0]

    sequence_stabilized = np.copy(sequence)

    for idx in range(1, N):
        curr_img = sequence[idx]

        optical_flow = get_block_matching(curr_img, prev_img, block_size_x, block_size_y, search_area_x, search_area_y, compensation = 'backward')

        stabilized_frame = get_stabilization(curr_img, prev_img, optical_flow)
        prev_img = stabilized_frame

        sequence_stabilized [:, :, idx] = stabilized_frame


    return  sequence_stabilized
