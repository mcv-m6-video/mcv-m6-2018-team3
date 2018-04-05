from kalman_filter import *

class track:
    def __init__(self, id, bbox, centroid, area, tracker_type):
        self.id = id
        self.centroid = centroid
        self.history_centroid = [centroid]
        self.bbox = bbox  # Define an initial bounding box, bbox = (287, 23, 86, 320)
        self.area = [area]
        self.totalVisible = 0
        self.consecutiveInvisible = 0
        self.speed = 0

        self.visible = True

        self.age = 0
        self.score = 0

        self.tracker_type = tracker_type
        #filter

        # TODO: Check tracker creation
        if self.tracker_type == 'kalman filter':
            self.tracker = kalman_filter(initialPosition=centroid)