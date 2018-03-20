import cv2
import numpy as np

def gunner_farneback(prvs, curr):

    # hsv = np.zeros_like(prvs)
    # hsv[..., 1] = 255
    #
    # flow = cv2.calcOpticalFlowFarneback(prvs, act, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow('frame2', bgr)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break
    # elif k == ord('s'):
    #     cv2.imwrite('opticalfb.png', frame2)
    #     cv2.imwrite('opticalhsv.png', bgr)
    # cap.release()
    # cv2.destroyAllWindows()

    return best_opt_flow