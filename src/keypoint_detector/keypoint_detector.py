import cv2
from src.keypoint_detector.PoseModule import PoseDetector


class KeypointDetector(object):
    def __init__(self):
        self.detector = PoseDetector()

    def get_pose(self, frame):
        frame = self.detector.findPose(frame, draw=False)
        lm_list = self.detector.findPosition(frame, draw=False)
        cur_angle = None
        if lm_list:
            if not self.detector.left():
                cur_angle = self.detector.findAngle(frame, 12, 14, 16, draw=True)
            else:
                cur_angle = self.detector.findAngle(frame, 11, 13, 15, draw=True)
        return frame, cur_angle

