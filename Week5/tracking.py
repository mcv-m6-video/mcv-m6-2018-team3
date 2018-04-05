import cv2
import numpy as np
from track import track
from utils import write_images2

# TODO: Check the thresholds (validate) & put in config file

thresh_dist = 50 # Highway = 50, Traffic = 80
thresh_consecutiveInvisible = 10
thresh_area = 5

tracker_type = 'kalman filter'

def getConnectedComponents(mask):
    connectivity = 4
    mask = mask * 255.
    mask = mask.astype("uint8")

    output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)

    nb_objects = output[0] - 1
    cc_map = output[1]
    bboxes = output[2]
    centroids = output[3]

    return nb_objects, cc_map, bboxes, centroids

def computeDistance(point1, point2):
    distance = pow((point1[0] - point2[0]) * 2 + (point1[1] - point2[1]) * 4, 0.5)
    return distance

def get_nearest_track(centroid, track_list):

    #predicted_centroids = [t.tracker.predict() for t in track_list]


    minDistance = 100
    track_index = -1
    for idx, t in enumerate(track_list):
        predicted_centroid = t.tracker.predict()
        distance = computeDistance(centroid, predicted_centroid)

        if distance < thresh_dist and distance < minDistance:
            minDistance = distance
            track_index = idx #index of menor distance

    return track_index

track_list = []
nb_tracks = 0

#X_res = [] #lo recivimos de IVAN (masks)
#Original_image = [] #lo recivimos de IVAN (original image)
X_res = np.load('masks.npy')
Original_image = np.load('original_images.npy')

found_index = []
output_tracking = []

for image, mask in zip(Original_image, X_res):
    nb_objects, cc_map, bboxes, centroids = getConnectedComponents(mask)

    for idx in np.unique(cc_map)[1:]:


        area = bboxes[idx][-1:]
        # Check if bbox area is valid

        if area < thresh_area:
            continue

        centroid = centroids[idx].astype('int')
        track_index = get_nearest_track(centroid, track_list)

        # TODO: Check if track_index is in found_index (there is already assigned)

        if track_index is -1:
            # create new track
            nb_tracks += 1
            newTrack = track(nb_tracks, bboxes[idx][:-1], centroid, area, tracker_type)
            track_list.append(newTrack)
        else:
            # Update track corresponding on track index
            track_list[track_index].centroid = centroid
            track_list[track_index].bbox = bboxes[idx][:-1]
            track_list[track_index].age += 1
            track_list[track_index].area.append(area)

            track_list[track_index].tracker.update(centroid)

            track_list[track_index].visible = True
            track_list[track_index].consecutiveInvisible = 0

            found_index.append(track_index)



    for idx, _ in enumerate(track_list):

        # Mark as False the existent tracks not found
        if idx not in found_index:
            track_list[idx].visible = False

        if track_list[idx].visible:
            track_list[idx].visible += 1
            # draw each bounding box into image
            color = [0, 255, 0]
            image = cv2.rectangle(image, (bboxes[idx][0], bboxes[idx][1]),
                                  (bboxes[idx][0] + bboxes[idx][2], bboxes[idx][1] + bboxes[idx][3]), color, 3)
        else:
            track_list[idx].consecutiveInvisible += 1
            if track_list[idx].consecutiveInvisible > thresh_consecutiveInvisible:
                track_list.remove(track_list[idx])

    get_nearest_track(cc_map, centroids)
    output_tracking.append(image)

# save images
output_tracking = np.array(output_tracking)
np.save('tracking_cube.npy', output_tracking)
write_images2(output_tracking, 'output', 'track_')


