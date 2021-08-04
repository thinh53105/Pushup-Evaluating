import time
import cv2
import tkinter as tk
from tkinter.filedialog import askopenfilename
from threading import Thread
from src.keypoint_detector.keypoint_detector import KeypointDetector

root = tk.Tk()
root.withdraw()


class VideoStreamer(object):
    def __init__(self):
        self.default_frame = cv2.imread('sample_images/img001.png')
        self.source = 'default'
        self.video_path = None
        self.is_con = True
        self.stream = None
        self.frame = None
        self.keypoint_detector = KeypointDetector()
        self.open_stream(None)

    def open_stream(self, video_path):
        self.stream = cv2.VideoCapture(video_path)

    def open_file(self):
        self.source = 'file'
        filename = askopenfilename()
        if filename:
            self.open_stream(filename)

    def open_camera(self):
        self.source = 'camera'
        self.open_stream(0)

    def start(self):
        t = Thread(target=self.get, args=())
        t.daemon = True
        t.start()
        return self

    def get(self):
        while True:
            if not self.is_con:
                time.sleep(1)
                continue
            success, frame = self.stream.read()
            if frame is not None:
                self.frame = frame
            else:
                self.source = 'default'
                self.frame = None

    def get_frame(self):
        if self.frame is None:
            return self.default_frame, None
        return self.keypoint_detector.get_pose(self.frame.copy())

    def stop(self):
        self.is_con = False

    def is_stopping(self):
        return not self.is_con

