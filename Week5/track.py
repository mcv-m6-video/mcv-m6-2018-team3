from kalman_filter import *

class track:
    def __init__(self, id, bbox, tracker_type):
        self.id = id
        self.centroid = [0, 0]
        self.bbox = bbox  # Define an initial bounding box, bbox = (287, 23, 86, 320)
        self.totalVisible = 0
        self.consecutiveInvisible = 0
        self.speed = 0

        self.tracker_type = tracker_type
        #filter
        if self.tracker_type == 'kalman filter':
            self.tracker = kalman_filter(id=0, startFrame=None, initialPosition=None)


