import cv2
import numpy as np

from block_matching import get_block_matching


def get_stabilization(curr_img, prev_img, motion)

# Backward precision
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