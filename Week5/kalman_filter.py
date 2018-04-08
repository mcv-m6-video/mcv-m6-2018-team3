import numpy as np
import cv2

class kalman_filter:

    def __init__(self, initialPosition):

        self.currentPositionX = initialPosition[0]
        self.currentPositionY = initialPosition[1]
        self.prioriEstimateX = self.currentPositionX
        self.prioriEstimateY = self.currentPositionY
        self.posterioriErrorX = 0
        self.posterioriErrorY = 0
        self.prioriErrorX = 0
        self.prioriErrorY = 0
        self.gainX = 0
        self.gainY = 0
#        self.Q = 1e-5  # process variance
#        self.R = 0.1 ** 2  # estimate of measurement variance, change to see effect
        self.Q = 1e-6  # process variance
        self.R = 0.001 ** 2  # estimate of measurement variance, change to see effect


    def predict(self):
        return [self.currentPositionX, self.currentPositionY]

    def update(self, currentPosition):
        # Compute X update
        self.prioriErrorX = self.posterioriErrorX + self.Q
        self.gainX = self.prioriErrorX / (self.prioriErrorX + self.R)
        self.currentPositionX = self.prioriEstimateX + self.gainX * (currentPosition[0]-self.prioriEstimateX)
        self.posterioriErrorX = (1-self.gainX)*self.prioriErrorX
        self.prioriEstimateX = self.currentPositionX

        # Compute Y update
        self.prioriErrorY = self.posterioriErrorY + self.Q
        self.gainY = self.prioriErrorY / (self.prioriErrorY + self.R)
        self.currentPositionY = self.prioriEstimateY + self.gainY * (currentPosition[1]-self.prioriEstimateY)
        self.posterioriErrorY = (1-self.gainY)*self.prioriErrorY
        self.prioriEstimateY = self.currentPositionY

class kcf:
    def __init__(self, frame, bbox, tracker_type):
        # Define an initial bounding box
        self.bbox = bbox
        # Uncomment the line below to select a different bounding box
        #self.bbox = cv2.selectROI(frame, False)

        self.currentPositionX = int( bbox[0] + (bbox[2] / 2) )
        self.currentPositionY = int( bbox[1] + (bbox[3] / 2) )

        # Initialize tracker with first frame and bounding box
        if tracker_type == 'boosting':
            self.tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'mil':
            self.tracker = cv2.TrackerMIL_create()
        if tracker_type == 'kcf':
            self.tracker = cv2.TrackerKCF_create()
        if tracker_type == 'tld':
            self.tracker = cv2.TrackerTLD_create()
        if tracker_type == 'medianflow':
            self.tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'goturn':
            self.tracker = cv2.TrackerGOTURN_create()

        self.ok = self.tracker.init(frame, (bbox[0], bbox[1], bbox[2], bbox[3]))

    def predict(self):
        return [self.currentPositionX, self.currentPositionY]

    def update(self, frame):
        self.ok, self.bbox = self.tracker.update(frame)
        self.currentPositionX = int(self.bbox[0] + (self.bbox[2] / 2))
        self.currentPositionY = int(self.bbox[1] + (self.bbox[3] / 2))