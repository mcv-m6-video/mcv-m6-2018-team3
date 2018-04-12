from scipy import ndimage, misc
import numpy as np
import matplotlib.pyplot as plt
import cv2

def RGBtoBGR(img):
    temp = img[:,:,0].copy()
    img[:,:,0] = img[:,:,2].copy()
    img[:,:,2] = temp
    return img

def speedlimit_emojis(sequence, image, speed, bb_pos, bb_shape):
    #-------------------CONFIGURATION------------------------
    #Highway
    sig = 50 #signal shape

    #Load emojis
    e1 = ndimage.imread("./emojis/1.png",mode='RGB')
    e2 = ndimage.imread("./emojis/2.png",mode='RGB')
    e3 = ndimage.imread("./emojis/3.png",mode='RGB')
    s70 = ndimage.imread("./emojis/s70.png", mode='RGB')

    e1 = misc.imresize(e1, (round(float(bb_shape[1]) * 0.2), round(float(bb_shape[1]) * 0.2), 3))
    e2 = misc.imresize(e2, (round(float(bb_shape[1])*0.2), round(float(bb_shape[1])*0.2), 3))
    e3 = misc.imresize(e3, (round(float(bb_shape[1]) * 0.2), round(float(bb_shape[1]) * 0.2), 3))
    s70 = misc.imresize(s70, (sig, sig, 3))

    e1 = RGBtoBGR(e1)
    e2 = RGBtoBGR(e2)
    e3 = RGBtoBGR(e3)
    s70 = RGBtoBGR(s70)

    # plt.figure()
    # plt.imshow(e2)
    # plt.show()

    if speed > 70:
        print('\n\n\n\n\n\n\n\n\n\n\n\n')
        print(bb_pos)
        bb_pos = (bb_pos[0], bb_pos[1] + 5)
        print(bb_pos)
        print(e2.shape)
        print(image[bb_pos[1]:bb_pos[1] + e2.shape[1], bb_pos[0]:bb_pos[0] + e2.shape[0]].shape)
        s1 = image[bb_pos[1]:bb_pos[1] + e2.shape[1], bb_pos[0]:bb_pos[0] + e2.shape[0]].shape
        s2 = e2.shape
        print('\n\n\n\n\n\n\n\n\n\n\n\n')

        if s1 == s2:
            image[bb_pos[1]:bb_pos[1] + e2.shape[1], bb_pos[0]:bb_pos[0] + e2.shape[0], :] = e2

    else:
        bb_pos = (bb_pos[0], bb_pos[1] + 5)
        s1 = image[bb_pos[1]:bb_pos[1] + e2.shape[1], bb_pos[0]:bb_pos[0] + e2.shape[0]].shape
        s2 = e2.shape

        if s1 == s2:
            image[bb_pos[1]:bb_pos[1] + e2.shape[1], bb_pos[0]:bb_pos[0] + e2.shape[0], :] = e1

    imsh = image.shape
    print(imsh)
    image[imsh[0]-sig-30 : imsh[0]-30, imsh[1]-sig-10: imsh[1]-10, :] = s70

    return image


def watermark(image, speed, x, y, hh, ww, high_shift=0, width_shift=0, alpha=1):

    image_rgb = image.copy()

    if speed > 70:
        watermark = cv2.imread("angry_bird3.png", cv2.IMREAD_UNCHANGED)
        watermark = cv2.resize(watermark, (ww, hh), interpolation=cv2.INTER_CUBIC)
        (wH, wW) = watermark.shape[:2]

        (B, G, R, A) = cv2.split(watermark)
        B = cv2.bitwise_and(B, B, mask=A)
        G = cv2.bitwise_and(G, G, mask=A)
        R = cv2.bitwise_and(R, R, mask=A)
        watermark = cv2.merge([B, G, R, A])

        (h, w) = image.shape[:2]
        image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])

        # construct an overlay that is the same size as the input
        # image, (using an extra dimension for the alpha transparency),
        # then add the watermark to the overlay in the bottom-right
        # corner
        overlay = np.zeros((h, w, 4), dtype="uint8")
        output = image.copy()

        if (y - wH - high_shift)>0 and (y - high_shift)>0 and (x - wW - width_shift)>0 and (x - width_shift)>0 :
            overlay[y - wH - high_shift:y - high_shift, x - wW - width_shift:x - width_shift] = watermark
            #overlay[h - wH - high_shift:h - high_shift, w - wW - width_shift:w - width_shift] = watermark

            # blend the two images together using transparent overlays
            output = image.copy()
            cv2.addWeighted(overlay, alpha, output, 1.0, 0, output)

        #cv2.imwrite("salida.png", output)
        image_rgb = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)
    else:
        watermark = cv2.imread("pacman.png", cv2.IMREAD_UNCHANGED)
        watermark = cv2.resize(watermark, (ww, hh), interpolation=cv2.INTER_CUBIC)
        (wH, wW) = watermark.shape[:2]

        (B, G, R, A) = cv2.split(watermark)
        B = cv2.bitwise_and(B, B, mask=A)
        G = cv2.bitwise_and(G, G, mask=A)
        R = cv2.bitwise_and(R, R, mask=A)
        watermark = cv2.merge([B, G, R, A])

        (h, w) = image.shape[:2]
        image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])

        # construct an overlay that is the same size as the input
        # image, (using an extra dimension for the alpha transparency),
        # then add the watermark to the overlay in the bottom-right
        # corner
        overlay = np.zeros((h, w, 4), dtype="uint8")
        output = image.copy()

        if (y - wH - high_shift)>0 and (y - high_shift)>0 and (x - wW - width_shift)>0 and (x - width_shift)>0 :
            overlay[y - wH - high_shift:y - high_shift, x - wW - width_shift:x - width_shift] = watermark
            #overlay[h - wH - high_shift:h - high_shift, w - wW - width_shift:w - width_shift] = watermark

            # blend the two images together using transparent overlays
            output = image.copy()
            cv2.addWeighted(overlay, alpha, output, 1.0, 0, output)

        #cv2.imwrite("salida.png", output)
        image_rgb = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)


    return image_rgb